"""Inbound Telegram updates.

Every handler does the same three things: check the user, persist the interaction, nudge the
submitter. No handler waits for the assistant to finish — that work is a job on the Core, and its
result comes back later as a ``telegram.send``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message
from pa_protocol import RpcError, methods, new_ulid

from ..storage.models import GatewayStore
from .formatting import confirmation_expired_notice, describe_error

log = logging.getLogger(__name__)

HELP_TEXT = """Персональный ассистент.

Пишите текстом или голосом — я отвечу.

Команды:
/new — начать новый разговор
/cancel — отменить текущую задачу
/status — состояние системы
/transcribe — ответом на голосовое: только расшифровка, без обработки
/reminders — список напоминаний
/tasks — список задач
/help — эта справка

Длинную запись встречи можно прислать файлом: я расшифрую её и разберу
на решения, задачи и сроки. Ничего не выполню без подтверждения."""

# Telegram gives voice notes no filename; a sensible default keeps ffmpeg's format sniffing happy.
_DEFAULT_AUDIO_NAME = "voice.ogg"


def build_router(bot: Bot, store: GatewayStore, core, submissions, settings) -> Router:
    router = Router(name="assistant")

    def authorized(user_id: int | None) -> bool:
        """Advisory gate only.

        The Core re-checks every request against its own allowlist, so this is a courtesy that
        avoids waking the Mac mini for strangers — never the security boundary.
        """
        if not settings.allowed_users:
            return True
        return f"tg:{user_id}" in settings.allowed_users

    async def enqueue(
        message: Message, *, kind: str, text: str | None = None, command: str | None = None,
    ) -> str:
        request_id = new_ulid()
        payload = methods.dump(
            methods.AssistantSubmitParams(
                request_id=request_id,
                user_id=f"tg:{message.from_user.id}",
                chat_id=message.chat.id,
                message_id=message.message_id,
                kind=kind,
                text=text,
                command=command,
                client_time=message.date,
            )
        )
        await store.save_request(
            request_id=request_id,
            user_id=f"tg:{message.from_user.id}",
            chat_id=message.chat.id,
            message_id=message.message_id,
            kind=kind,
            payload=payload,
        )
        submissions.nudge()
        return request_id

    # ---- commands ----------------------------------------------------

    @router.message(CommandStart())
    async def on_start(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        await message.answer(HELP_TEXT)

    @router.message(Command("help"))
    async def on_help(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        await message.answer(HELP_TEXT)

    @router.message(Command("new"))
    async def on_new(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        if not core.connected:
            await message.answer(describe_error("not_ready"))
            return
        try:
            await core.call(
                methods.SESSION_RESET,
                methods.dump(
                    methods.SessionResetParams(
                        user_id=f"tg:{message.from_user.id}", request_id=new_ulid()
                    )
                ),
            )
        except RpcError as exc:
            await message.answer(describe_error(exc.message))
            return
        await message.answer("Начал новый разговор. Прошлый контекст больше не используется.")

    @router.message(Command("cancel"))
    async def on_cancel(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        await enqueue(message, kind="command", command="/cancel")
        await message.answer("Отменяю текущую задачу…")

    @router.message(Command("status"))
    async def on_status(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        queued = await store.pending_request_count()
        if not core.connected:
            await message.answer(
                "Gateway: online\n"
                "Core: disconnected\n"
                f"Отложенных запросов: {queued}"
            )
            return
        try:
            status = await core.call(methods.STATUS_GET, {})
        except RpcError as exc:
            await message.answer(describe_error(exc.message))
            return
        await message.answer(_render_status(status, queued))

    @router.message(Command("reminders"))
    async def on_reminders(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        await enqueue(message, kind="command", command="/reminders")

    @router.message(Command("tasks"))
    async def on_tasks(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        await enqueue(message, kind="command", command="/tasks")

    @router.message(Command("transcribe"))
    async def on_transcribe(message: Message) -> None:
        """Transcription only: no conversation, no tools.

        Requires a reply to an audio message so the intent is unambiguous.
        """
        if not authorized(message.from_user.id):
            return
        target = message.reply_to_message
        if target is None or not _audio_of(target):
            await message.answer(
                "Ответьте командой /transcribe на голосовое сообщение или аудиофайл."
            )
            return
        await _accept_audio(target, purpose="transcribe_only", trigger=message)

    # ---- content -----------------------------------------------------

    @router.message(F.text & ~F.text.startswith("/"))
    async def on_text(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        await enqueue(message, kind="text", text=message.text)

    @router.message(F.voice | F.audio | F.document | F.video_note)
    async def on_audio(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        if _audio_of(message) is None:
            return
        await _accept_audio(message, purpose="assistant", trigger=message)

    async def _accept_audio(message: Message, *, purpose: str, trigger: Message) -> None:
        media = _audio_of(message)
        if media is None:
            return

        size = getattr(media, "file_size", 0) or 0
        if size > settings.max_download_bytes:
            await trigger.answer("Файл слишком большой.")
            return

        request_id = new_ulid()
        filename = getattr(media, "file_name", None) or _DEFAULT_AUDIO_NAME
        target = settings.resolved_temp_dir / f"{request_id}_{_safe_name(filename)}"

        try:
            await trigger.bot.send_chat_action(message.chat.id, "typing")
            file = await trigger.bot.get_file(media.file_id)
            await trigger.bot.download_file(file.file_path, destination=target)
        except Exception:
            log.exception("failed to download audio from telegram")
            await trigger.answer("Не удалось скачать файл из Telegram.")
            target.unlink(missing_ok=True)
            return

        digest = await _sha256(target)
        actual_size = target.stat().st_size

        await store.save_upload(
            request_id=request_id,
            user_id=f"tg:{message.from_user.id}",
            chat_id=message.chat.id,
            message_id=message.message_id,
            file_path=target,
            filename=filename,
            content_type=getattr(media, "mime_type", None),
            size=actual_size,
            sha256=digest,
            duration_seconds=float(getattr(media, "duration", 0) or 0) or None,
            purpose=purpose,
        )
        # The submit references the upload; the Core starts work at audio.commit.
        await store.save_request(
            request_id=request_id,
            user_id=f"tg:{message.from_user.id}",
            chat_id=message.chat.id,
            message_id=message.message_id,
            kind="audio",
            payload={},
        )
        await store.mark_request_submitted(request_id, None)
        submissions.nudge()

    # ---- confirmation callbacks -----------------------------------------

    @router.callback_query(F.data.startswith("c:"))
    async def on_confirmation(query: CallbackQuery) -> None:
        token = query.data[2:]
        record = await store.resolve_confirmation_token(token)
        if record is None:
            await query.answer(confirmation_expired_notice(), show_alert=True)
            return
        if record["user_id"] != f"tg:{query.from_user.id}":
            # Someone else's button in a shared chat.
            await query.answer("Это подтверждение не для вас.", show_alert=True)
            return
        if record["status"] == "used":
            await query.answer("Уже обработано.")
            return

        if not core.connected:
            await query.answer(describe_error("not_ready"), show_alert=True)
            return

        try:
            result = await core.call(
                methods.CONFIRMATION_RESOLVE,
                methods.dump(
                    methods.ConfirmationResolveParams(
                        action_id=record["action_id"],
                        user_id=record["user_id"],
                        choice=record["choice"],
                    )
                ),
            )
        except RpcError as exc:
            await query.answer(describe_error(exc.message), show_alert=True)
            return

        await store.mark_confirmation_used(record["action_id"])
        status = (result or {}).get("status")
        await query.answer(
            {
                "applied": "Принято.",
                "already_resolved": "Уже обработано.",
                "expired": confirmation_expired_notice(),
                "unknown": "Действие не найдено.",
            }.get(status, "Готово.")
        )
        # Strip the keyboard so the decision cannot be pressed again.
        try:
            await query.message.edit_reply_markup(reply_markup=None)
        except Exception:
            log.debug("could not clear confirmation keyboard", exc_info=True)

    return router


def _audio_of(message: Message) -> Any | None:
    """Return the audio-bearing attachment, if any.

    Documents count only when their MIME type claims audio or video, so a PDF reply does not get
    pushed through Whisper.
    """
    if message.voice:
        return message.voice
    if message.audio:
        return message.audio
    if message.video_note:
        return message.video_note
    document = message.document
    if document is not None:
        mime = (document.mime_type or "").lower()
        if mime.startswith(("audio/", "video/")):
            return document
    return None


def _safe_name(filename: str) -> str:
    cleaned = "".join(c for c in filename if c.isalnum() or c in "._-")
    return cleaned[-80:] or "audio"


async def _sha256(path: Path) -> str:
    import asyncio

    def compute() -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    return await asyncio.get_running_loop().run_in_executor(None, compute)


def _render_status(status: dict[str, Any], queued_locally: int) -> str:
    core = status.get("core", {})
    cursor = status.get("cursor", {})
    stt = status.get("stt", {})
    scheduler = status.get("scheduler", {})
    jobs = status.get("jobs", {})
    lines = [
        "Gateway: online",
        f"Core: connected ({core.get('instance_id', '?')})",
        f"Cursor: {cursor.get('state', '?')}",
        f"Whisper: {stt.get('state', '?')} ({stt.get('model', '?')})",
        f"Scheduler: {scheduler.get('state', '?')}, напоминаний: {scheduler.get('pending_reminders', 0)}",
        f"Jobs: {jobs.get('running', 0)} running / {jobs.get('queued', 0)} queued",
    ]
    if queued_locally:
        lines.append(f"Отложено на Gateway: {queued_locally}")
    return "\n".join(lines)
