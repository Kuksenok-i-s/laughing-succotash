"""Confirmation flow.

Confirmation state lives on the Core. The Gateway only renders buttons and reports which one was
pressed; it never decides and never executes.

A pending action stores the exact validated arguments, so approval executes precisely what the
user was shown rather than a re-derivation of it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pa_protocol import methods, new_ulid

from ..storage.repositories import PendingAction, PendingActionRepository

log = logging.getLogger(__name__)


class ConfirmationService:
    def __init__(
        self,
        pending: PendingActionRepository,
        link,
        *,
        timeout_seconds: int = 900,
    ) -> None:
        self._pending = pending
        self._link = link
        self._timeout = timeout_seconds
        # action_id -> future resolved by the Gateway callback or by expiry.
        self._waiters: dict[str, asyncio.Future[str]] = {}

    async def request(
        self,
        *,
        user_id: str,
        chat_id: int | None,
        tool_name: str,
        arguments: dict[str, Any],
        operation_id: str,
        tier: str,
        prompt_text: str,
        job_id: str | None = None,
    ) -> bool:
        """Ask the user and block until they answer.

        Blocks the MCP tool call — and therefore that one agent turn — but nothing else: turns are
        serialised per conversation, so other users are unaffected. The wait is bounded and a
        timeout resolves to rejection, so an ignored prompt costs one turn rather than a session.
        """
        choice = await self.request_choice(
            user_id=user_id,
            chat_id=chat_id,
            tool_name=tool_name,
            arguments=arguments,
            operation_id=operation_id,
            tier=tier,
            prompt_text=prompt_text,
            job_id=job_id,
            actions=[
                methods.ConfirmAction(id="approve", label="Да", style="primary"),
                methods.ConfirmAction(id="reject", label="Отмена", style="secondary"),
            ],
        )
        return choice == "approve"

    async def request_choice(
        self,
        *,
        user_id: str,
        chat_id: int | None,
        tool_name: str,
        arguments: dict[str, Any],
        operation_id: str,
        tier: str,
        prompt_text: str,
        actions: list[methods.ConfirmAction],
        job_id: str | None = None,
    ) -> str | None:
        """Like :meth:`request`, but returns the button id (or ``None`` if refused/expired)."""
        action = await self._pending.create(
            user_id=user_id,
            chat_id=chat_id,
            job_id=job_id,
            tool_name=tool_name,
            arguments=arguments,
            operation_id=operation_id,
            tier=tier,
            prompt_text=prompt_text,
            ttl_seconds=self._timeout,
        )

        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._waiters[action.action_id] = future

        await self._link.send_event(
            methods.TELEGRAM_CONFIRM,
            methods.dump(
                methods.TelegramConfirmParams(
                    delivery_id=new_ulid(),
                    action_id=action.action_id,
                    user_id=user_id,
                    chat_id=chat_id or 0,
                    text=prompt_text,
                    actions=list(actions),
                    expires_at=action.expires_at,
                )
            ),
            user_id=user_id,
        )

        try:
            choice = await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
            log.info("confirmation %s expired for %s", action.action_id, tool_name)
            await self._pending.resolve(action.action_id, user_id, "expired")
            return None
        except asyncio.CancelledError:
            await self._pending.resolve(action.action_id, user_id, "rejected")
            raise
        finally:
            self._waiters.pop(action.action_id, None)

        if choice in {"reject", "expired", "cancel"}:
            return None
        return choice

    async def resolve(self, action_id: str, user_id: str, choice: str) -> str:
        """Apply the user's answer. Called by the ``confirmation.resolve`` RPC handler."""
        db_status = "rejected" if choice in {"reject", "expired", "cancel"} else "approved"
        status = await self._pending.resolve(action_id, user_id, db_status)
        if status != "applied":
            return status

        future = self._waiters.get(action_id)
        if future is not None and not future.done():
            future.set_result(choice)
        else:
            # Nobody is waiting: the Core restarted since the prompt was sent, so the tool call
            # that would have executed this no longer exists. Report it rather than silently
            # dropping the user's decision.
            log.warning(
                "confirmation %s resolved but its caller is gone (core restarted?)", action_id
            )
        return "applied"

    async def expire_overdue(self) -> list[PendingAction]:
        """Sweep timed-out confirmations. Driven by the scheduler tick."""
        expired = await self._pending.expire_overdue()
        for action in expired:
            future = self._waiters.get(action.action_id)
            if future is not None and not future.done():
                future.set_result("expired")
        return expired

    def abandon_all(self) -> None:
        """Refuse every outstanding confirmation. Used during shutdown."""
        for future in self._waiters.values():
            if not future.done():
                future.set_result("reject")
        self._waiters.clear()
