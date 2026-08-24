"""Executes Core-originated ``telegram.*`` calls against the Bot API.

This is the whole of the Gateway's "intelligence": it turns declarative intent into Telegram
calls. It never decides whether an action should happen — that is settled on the Core before the
call is made.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.types import BufferedInputFile, InlineKeyboardButton, InlineKeyboardMarkup
from pa_protocol import RpcError, errors, methods

from ..storage.models import GatewayStore
from .formatting import describe_stage, split_message

log = logging.getLogger(__name__)


class TelegramRenderer:
    def __init__(self, bot: Bot, store: GatewayStore, settings) -> None:
        self._bot = bot
        self._store = store
        self._settings = settings

    def handlers(self) -> dict[str, Any]:
        return {
            methods.TELEGRAM_SEND: self.send,
            methods.TELEGRAM_SEND_DOCUMENT: self.send_document,
            methods.TELEGRAM_EDIT: self.edit,
            methods.TELEGRAM_DELETE: self.delete,
            methods.TELEGRAM_ACTION: self.action,
            methods.TELEGRAM_CONFIRM: self.confirm,
            methods.JOB_PROGRESS: self.job_progress,
            methods.JOB_COMPLETED: self.job_completed,
            methods.JOB_FAILED: self.job_failed,
        }

    # ---- send ----------------------------------------------------------

    async def send(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = methods.TelegramSendParams.model_validate(raw)

        is_new, existing_message_id = await self._store.claim_delivery(
            params.delivery_id, methods.TELEGRAM_SEND
        )
        if not is_new:
            # At-least-once delivery means we will see this again after a reconnect. The user must
            # not see the message twice.
            log.info("duplicate delivery %s ignored", params.delivery_id)
            return methods.dump(
                methods.TelegramSendResult(message_id=existing_message_id, dedup=True)
            )

        try:
            message_id = await self._send_parts(
                chat_id=params.chat_id,
                text=params.text,
                parse_mode=params.parse_mode,
                reply_to_message_id=params.reply_to_message_id,
                silent=params.silent,
            )
        except TelegramForbiddenError as exc:
            await self._store.complete_delivery(
                params.delivery_id, chat_id=params.chat_id, message_id=None
            )
            raise RpcError(errors.TELEGRAM_BLOCKED, "telegram_blocked", {"detail": str(exc)})
        except Exception as exc:
            await self._store.release_delivery(params.delivery_id)
            raise RpcError(
                errors.TELEGRAM_SEND_FAILED, "telegram_send_failed", {"detail": str(exc)[:200]}
            )

        await self._store.complete_delivery(
            params.delivery_id, chat_id=params.chat_id, message_id=message_id
        )
        return methods.dump(methods.TelegramSendResult(message_id=message_id, dedup=False))

    async def send_document(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = methods.TelegramSendDocumentParams.model_validate(raw)

        is_new, existing_message_id = await self._store.claim_delivery(
            params.delivery_id, methods.TELEGRAM_SEND_DOCUMENT
        )
        if not is_new:
            log.info("duplicate document delivery %s ignored", params.delivery_id)
            return methods.dump(
                methods.TelegramSendDocumentResult(
                    message_id=existing_message_id, dedup=True
                )
            )

        filename = (params.filename or "document.md").replace("\\", "/").rsplit("/", 1)[-1]
        if not filename.strip():
            filename = "document.md"
        document = BufferedInputFile(
            params.content.encode("utf-8"), filename=filename
        )
        caption = (params.caption or "").strip() or None
        try:
            message = await self._call_with_retry(
                self._bot.send_document,
                chat_id=params.chat_id,
                document=document,
                caption=caption[:1024] if caption else None,
                parse_mode=(
                    "Markdown" if caption and params.parse_mode == "markdown" else None
                ),
                reply_to_message_id=params.reply_to_message_id,
                disable_notification=params.silent,
            )
        except TelegramForbiddenError as exc:
            await self._store.complete_delivery(
                params.delivery_id, chat_id=params.chat_id, message_id=None
            )
            raise RpcError(errors.TELEGRAM_BLOCKED, "telegram_blocked", {"detail": str(exc)})
        except Exception as exc:
            await self._store.release_delivery(params.delivery_id)
            raise RpcError(
                errors.TELEGRAM_SEND_FAILED, "telegram_send_failed", {"detail": str(exc)[:200]}
            )

        message_id = getattr(message, "message_id", None)
        await self._store.complete_delivery(
            params.delivery_id, chat_id=params.chat_id, message_id=message_id
        )
        return methods.dump(
            methods.TelegramSendDocumentResult(message_id=message_id, dedup=False)
        )

    async def _send_parts(
        self, *, chat_id: int, text: str, parse_mode: str,
        reply_to_message_id: int | None = None, silent: bool = False,
    ) -> int | None:
        parts = split_message(text, self._settings.telegram_message_limit)
        last_id: int | None = None
        for index, part in enumerate(parts):
            if not part.strip():
                continue
            message = await self._call_with_retry(
                self._bot.send_message,
                chat_id=chat_id,
                text=part,
                parse_mode="Markdown" if parse_mode == "markdown" else None,
                reply_to_message_id=reply_to_message_id if index == 0 else None,
                disable_notification=silent,
            )
            last_id = message.message_id
        return last_id

    async def _call_with_retry(self, fn, **kwargs):
        """Send, coping with Telegram's two most common refusals.

        A 429 is retried after the interval Telegram dictates. A Markdown parse failure is retried
        once as plain text, because an assistant reply containing an unbalanced asterisk should
        still reach the user rather than vanishing.
        """
        for attempt in range(3):
            try:
                return await fn(**kwargs)
            except TelegramRetryAfter as exc:
                if attempt == 2:
                    raise
                await asyncio.sleep(exc.retry_after + 0.5)
            except TelegramBadRequest as exc:
                if "parse" in str(exc).lower() and kwargs.get("parse_mode"):
                    log.info("markdown rejected by telegram, resending as plain text")
                    kwargs = {**kwargs, "parse_mode": None}
                    continue
                raise
        raise RuntimeError("unreachable")

    # ---- edit / delete ---------------------------------------------------

    async def edit(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = methods.TelegramEditParams.model_validate(raw)
        try:
            await self._bot.edit_message_text(
                chat_id=params.chat_id,
                message_id=params.message_id,
                text=params.text[: self._settings.telegram_message_limit],
                parse_mode="Markdown" if params.parse_mode == "markdown" else None,
            )
        except TelegramBadRequest as exc:
            # "message is not modified" is the expected outcome of a repeated progress update, not
            # an error worth propagating.
            if "not modified" in str(exc).lower():
                return methods.dump(methods.TelegramEditResult(edited=False))
            raise RpcError(
                errors.TELEGRAM_SEND_FAILED, "telegram_send_failed", {"detail": str(exc)[:200]}
            )
        return methods.dump(methods.TelegramEditResult(edited=True))

    async def delete(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = methods.TelegramDeleteParams.model_validate(raw)
        try:
            await self._bot.delete_message(params.chat_id, params.message_id)
        except TelegramBadRequest:
            # Already gone is the desired end state.
            return methods.dump(methods.TelegramDeleteResult(deleted=True))
        return methods.dump(methods.TelegramDeleteResult(deleted=True))

    async def action(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = methods.TelegramActionParams.model_validate(raw)
        mapping = {
            "typing": "typing",
            "record_voice": "record_voice",
            "upload_document": "upload_document",
        }
        try:
            await self._bot.send_chat_action(params.chat_id, mapping[params.action])
        except Exception:
            log.debug("chat action failed", exc_info=True)
        return {}

    # ---- confirmations -----------------------------------------------------

    async def confirm(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = methods.TelegramConfirmParams.model_validate(raw)

        is_new, existing_message_id = await self._store.claim_delivery(
            params.delivery_id, methods.TELEGRAM_CONFIRM
        )
        if not is_new:
            return methods.dump(
                methods.TelegramConfirmResult(message_id=existing_message_id, dedup=True)
            )

        tokens = await self._store.create_confirmation_tokens(
            action_id=params.action_id,
            user_id=params.user_id,
            chat_id=params.chat_id,
            choices=[action.id for action in params.actions],
        )
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text=action.label, callback_data=f"c:{tokens[action.id]}")
                    for action in params.actions
                ]
            ]
        )

        try:
            message = await self._bot.send_message(
                chat_id=params.chat_id, text=params.text, reply_markup=keyboard
            )
        except Exception as exc:
            await self._store.release_delivery(params.delivery_id)
            raise RpcError(
                errors.TELEGRAM_SEND_FAILED, "telegram_send_failed", {"detail": str(exc)[:200]}
            )

        await self._store.set_confirmation_message(params.action_id, message.message_id)
        await self._store.complete_delivery(
            params.delivery_id, chat_id=params.chat_id, message_id=message.message_id
        )
        return methods.dump(
            methods.TelegramConfirmResult(message_id=message.message_id, dedup=False)
        )

    # ---- job lifecycle -------------------------------------------------------

    async def job_progress(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Update one status message per job.

        Progress arrives far faster than Telegram tolerates edits, so this coalesces into a single
        message and throttles. Dropping an update is always acceptable — the next one supersedes
        it.
        """
        params = methods.JobProgressParams.model_validate(raw)
        if params.chat_id is None:
            return {}

        text = describe_stage(params.stage, params.detail, params.progress)
        existing = await self._store.get_status_message(params.job_id)

        if existing is None:
            try:
                message = await self._bot.send_message(chat_id=params.chat_id, text=text)
            except Exception:
                log.debug("could not post status message", exc_info=True)
                return {}
            await self._store.set_status_message(
                params.job_id, params.chat_id, message.message_id, text
            )
            return {}

        if existing["last_text"] == text:
            return {}
        if not self._edit_allowed(existing["last_edit_at"]):
            return {}

        try:
            await self._bot.edit_message_text(
                chat_id=existing["chat_id"], message_id=existing["message_id"], text=text
            )
            await self._store.touch_status_message(params.job_id, text)
        except TelegramBadRequest:
            log.debug("status edit rejected", exc_info=True)
        return {}

    def _edit_allowed(self, last_edit_at: str | None) -> bool:
        if not last_edit_at:
            return True
        from datetime import datetime, timezone

        try:
            previous = datetime.fromisoformat(last_edit_at)
        except ValueError:
            return True
        if previous.tzinfo is None:
            previous = previous.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - previous).total_seconds()
        return elapsed >= self._settings.status_edit_min_interval

    async def job_completed(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = methods.JobCompletedParams.model_validate(raw)
        await self._remove_status_message(params.job_id)
        return {}

    async def job_failed(self, raw: dict[str, Any]) -> dict[str, Any]:
        from .formatting import describe_error

        params = methods.JobFailedParams.model_validate(raw)
        existing = await self._store.get_status_message(params.job_id)
        message = describe_error(params.error.code)

        if existing is not None:
            try:
                await self._bot.edit_message_text(
                    chat_id=existing["chat_id"],
                    message_id=existing["message_id"],
                    text=message,
                )
            except TelegramBadRequest:
                log.debug("could not edit failed status message", exc_info=True)
            await self._store.clear_status_message(params.job_id)
        elif params.chat_id is not None:
            try:
                await self._bot.send_message(chat_id=params.chat_id, text=message)
            except Exception:
                log.debug("could not report job failure", exc_info=True)
        return {}

    async def _remove_status_message(self, job_id: str) -> None:
        """Delete the transient status message once the real reply has been sent."""
        existing = await self._store.get_status_message(job_id)
        if existing is None:
            return
        try:
            await self._bot.delete_message(existing["chat_id"], existing["message_id"])
        except Exception:
            log.debug("could not delete status message", exc_info=True)
        await self._store.clear_status_message(job_id)
