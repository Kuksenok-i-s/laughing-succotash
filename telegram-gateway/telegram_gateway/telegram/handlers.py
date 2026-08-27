"""Inbound Telegram updates.

Every handler does the same three things: check the user, persist the interaction, nudge the
submitter. No handler waits for the assistant to finish — that work is a job on the Core, and its
result comes back later as a ``telegram.send``.
"""

from __future__ import annotations

import asyncio
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
from .keyboard import BUTTON_COMMANDS, HIDE, hide_keyboard, main_keyboard
from .origin import attribution_of

log = logging.getLogger(__name__)

HELP_TEXT = """Персональный ассистент.

Пишите текстом или голосом — я отвечу.

Команды:
/new — начать новый разговор
/cancel — отменить текущую задачу
/status — состояние системы
/transcribe — ответом на голосовое: только расшифровка, без обработки
/ocr — ответом на фото: распознать рукописный текст ещё раз
/reminders — список напоминаний
/tasks — список задач
/journal — дневник за сегодня (или начать опрос)
/keyboard — показать кнопки
/help — эта справка

Кнопки внизу экрана делают то же самое. «Скрыть кнопки» убирает их.

Длинную запись встречи можно прислать файлом: я расшифрую её и разберу
на решения, задачи и сроки.

Фотографию можно прислать как есть: если на ней текст — распознаю и предложу
задачи; если нет — коротко опишу, что на снимке. Несколько фото одним альбомом
обрабатываются вместе, как одна смысловая пачка.

Ссылку на YouTube:
• видео, плейлист или канал — кнопками «Конспект» или «Скачать видео»;
• «конспект» — расшифровка, тезисы и (для плейлиста/канала) общий обзор;
• «скачай» — выкачка видео на диск.
Ничего не выполню без подтверждения."""

# Telegram gives voice notes no filename; a sensible default keeps ffmpeg's format sniffing happy.
_DEFAULT_AUDIO_NAME = "voice.ogg"
_DEFAULT_IMAGE_NAME = "photo.jpg"
_ALBUM_DEBOUNCE_SECONDS = 1.5


def build_router(bot: Bot, store: GatewayStore, core, submissions, settings) -> Router:
    router = Router(name="assistant")
    album_buffers: dict[str, list[Message]] = {}
    album_timers: dict[str, asyncio.Task] = {}

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
        sender, source = attribution_of(message)
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
                sender=sender,
                source=source,
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
        await message.answer(HELP_TEXT, reply_markup=main_keyboard())

    @router.message(Command("help"))
    async def on_help(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        await message.answer(HELP_TEXT, reply_markup=main_keyboard())

    @router.message(Command("keyboard"))
    async def on_keyboard(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        await message.answer("Кнопки на месте.", reply_markup=main_keyboard())

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

    @router.message(Command("journal"))
    async def on_journal(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        await enqueue(message, kind="command", command="/journal")

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

    @router.message(Command("ocr"))
    async def on_ocr(message: Message) -> None:
        """Re-run handwriting OCR on a replied photo or image document."""
        if not authorized(message.from_user.id):
            return
        target = message.reply_to_message
        if target is None or _image_of(target) is None:
            await message.answer(
                "Ответьте командой /ocr на фотографию или изображение-файл."
            )
            return
        await _accept_image(target, trigger=message)

    _COMMANDS = {
        "new": on_new,
        "cancel": on_cancel,
        "reminders": on_reminders,
        "tasks": on_tasks,
        "journal": on_journal,
        "status": on_status,
        "help": on_help,
    }

    # ---- reply-keyboard labels ---------------------------------------
    # Telegram delivers these as ordinary text, not as slash commands.

    @router.message(F.text == HIDE)
    async def on_hide_keyboard(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        await message.answer(
            "Кнопки скрыты. /keyboard вернёт их.",
            reply_markup=hide_keyboard(),
        )

    @router.message(F.text.in_(set(BUTTON_COMMANDS)))
    async def on_menu_button(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        name = BUTTON_COMMANDS[message.text]
        await _COMMANDS[name](message)

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
        if _image_of(message) is not None:
            await _accept_image(message, trigger=message)
            return
        if _audio_of(message) is None:
            return
        await _accept_audio(message, purpose="assistant", trigger=message)

    @router.message(F.photo)
    async def on_photo(message: Message) -> None:
        if not authorized(message.from_user.id):
            return
        if message.media_group_id:
            await _buffer_album_photo(message)
            return
        await _accept_image(message, trigger=message)

    async def _buffer_album_photo(message: Message) -> None:
        group_id = message.media_group_id
        assert group_id is not None
        album_buffers.setdefault(group_id, []).append(message)
        previous = album_timers.get(group_id)
        if previous is not None and not previous.done():
            previous.cancel()

        async def flush_later() -> None:
            try:
                await asyncio.sleep(_ALBUM_DEBOUNCE_SECONDS)
            except asyncio.CancelledError:
                return
            batch = album_buffers.pop(group_id, [])
            album_timers.pop(group_id, None)
            if not batch:
                return
            await _accept_album(batch)

        album_timers[group_id] = asyncio.create_task(flush_later())

    async def _accept_album(messages: list[Message]) -> None:
        ordered = sorted(messages, key=lambda item: item.message_id)
        album_id = new_ulid()
        part_count = len(ordered)
        for index, message in enumerate(ordered):
            await _accept_image(
                message,
                trigger=message,
                album_id=album_id,
                part_index=index,
                part_count=part_count,
            )

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
        sender, source = attribution_of(message)

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
            attribution={"sender": methods.dump(sender), "source": methods.dump(source)},
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

    async def _accept_image(
        message: Message,
        *,
        trigger: Message,
        album_id: str | None = None,
        part_index: int | None = None,
        part_count: int | None = None,
    ) -> None:
        media = _image_of(message)
        if media is None:
            return

        size = getattr(media, "file_size", 0) or 0
        if size > settings.max_download_bytes:
            await trigger.answer("Файл слишком большой.")
            return

        request_id = new_ulid()
        filename = getattr(media, "file_name", None) or _DEFAULT_IMAGE_NAME
        content_type = getattr(media, "mime_type", None) or "image/jpeg"
        target = settings.resolved_temp_dir / f"{request_id}_{_safe_name(filename)}"
        caption = (message.caption or "").strip() or None

        try:
            await trigger.bot.send_chat_action(message.chat.id, "typing")
            file = await trigger.bot.get_file(media.file_id)
            await trigger.bot.download_file(file.file_path, destination=target)
        except Exception:
            log.exception("failed to download image from telegram")
            await trigger.answer("Не удалось скачать файл из Telegram.")
            target.unlink(missing_ok=True)
            return

        digest = await _sha256(target)
        actual_size = target.stat().st_size
        sender, source = attribution_of(message)

        await store.save_upload(
            request_id=request_id,
            user_id=f"tg:{message.from_user.id}",
            chat_id=message.chat.id,
            message_id=message.message_id,
            file_path=target,
            filename=filename,
            content_type=content_type,
            size=actual_size,
            sha256=digest,
            purpose="ocr",
            caption=caption,
            attribution={"sender": methods.dump(sender), "source": methods.dump(source)},
            album_id=album_id,
            part_index=part_index,
            part_count=part_count,
        )
        payload: dict[str, Any] = {}
        if caption:
            payload["caption"] = caption
        if album_id:
            payload["album_id"] = album_id
            payload["part_index"] = part_index
            payload["part_count"] = part_count
        await store.save_request(
            request_id=request_id,
            user_id=f"tg:{message.from_user.id}",
            chat_id=message.chat.id,
            message_id=message.message_id,
            kind="image",
            payload=payload,
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


def _image_of(message: Message) -> Any | None:
    """Return the largest photo size or an image document, if any."""
    if message.photo:
        return message.photo[-1]
    document = message.document
    if document is not None:
        mime = (document.mime_type or "").lower()
        if mime.startswith("image/"):
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
        f"OCR: {status.get('ocr', {}).get('state', '?')} ({status.get('ocr', {}).get('model', '-')})",
        f"Scheduler: {scheduler.get('state', '?')}, напоминаний: {scheduler.get('pending_reminders', 0)}",
        f"Jobs: {jobs.get('running', 0)} running / {jobs.get('queued', 0)} queued",
    ]
    if queued_locally:
        lines.append(f"Отложено на Gateway: {queued_locally}")
    return "\n".join(lines)
