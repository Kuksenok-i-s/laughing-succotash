"""Inbound Telegram updates, driven through the real aiogram router.

The handlers are the one place where Telegram's object model meets ours, so the updates here are
built as aiogram types and dispatched for real rather than being called as plain functions.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import (
    Audio,
    CallbackQuery,
    Chat,
    Document,
    File,
    Message,
    Update,
    User,
    Voice,
)
from pa_protocol import RpcError, errors, methods, new_ulid

from telegram_gateway.telegram.handlers import build_router

CHAT_ID = 777
USER_ID = 1


class RecordingCore:
    def __init__(self) -> None:
        self.connected = True
        self.calls: list[tuple[str, dict]] = []
        self.result: dict = {}
        self.error: Exception | None = None

    async def call(self, method: str, params: dict, *, timeout=None):
        self.calls.append((method, params))
        if self.error is not None:
            raise self.error
        return self.result


class RecordingSubmissions:
    def __init__(self) -> None:
        self.nudges = 0

    def nudge(self) -> None:
        self.nudges += 1


class HandlerBot(Bot):
    """A real ``Bot`` with the network intercepted at the one place everything funnels through.

    ``message.answer()`` does not call ``Bot.send_message``; it builds a method object and invokes
    the bot, so the interception has to happen in ``__call__`` or nothing is caught.
    """

    def __init__(self, token: str, audio: bytes = b"opus") -> None:
        super().__init__(token=token)
        self.answers: list[str] = []
        self.callback_answers: list[str] = []
        self.markup_cleared = 0
        self._audio = audio

    async def __call__(self, method, request_timeout=None):
        name = type(method).__name__
        if name == "SendMessage":
            self.answers.append(method.text)
            return _message(message_id=9000 + len(self.answers), text=method.text, bot=self)
        if name == "AnswerCallbackQuery":
            self.callback_answers.append(method.text or "")
            return True
        if name == "EditMessageReplyMarkup":
            self.markup_cleared += 1
            return True
        if name == "GetFile":
            return File(
                file_id=method.file_id,
                file_unique_id=method.file_id,
                file_size=len(self._audio),
                file_path=f"voice/{method.file_id}.ogg",
            )
        if name == "GetMe":
            return User(id=99, is_bot=True, first_name="bot", username="assistant_bot")
        if name in ("SendChatAction", "EditMessageText", "DeleteMessage"):
            return True
        raise AssertionError(f"unexpected Bot API call: {name}")

    async def download_file(self, file_path, destination=None, **kwargs):
        Path(destination).write_bytes(self._audio)
        return destination


def _message(*, message_id: int = 1, text: str | None = None, bot=None, **fields) -> Message:
    message = Message(
        message_id=message_id,
        date=datetime.now(timezone.utc),
        chat=Chat(id=CHAT_ID, type="private"),
        from_user=User(id=USER_ID, is_bot=False, first_name="Илья"),
        text=text,
        **fields,
    )
    return message.as_(bot) if bot is not None else message


@pytest.fixture
def bot() -> HandlerBot:
    return HandlerBot("123:fake")


@pytest.fixture
async def wired(bot, store, settings):
    core = RecordingCore()
    submissions = RecordingSubmissions()
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(bot, store, core, submissions, settings))
    try:
        yield dispatcher, core, submissions
    finally:
        await bot.session.close()


async def feed(dispatcher, bot, **update_fields) -> None:
    await dispatcher.feed_update(bot, Update(update_id=1, **update_fields))


# ---- text and commands ----------------------------------------------------


async def test_a_text_message_is_queued_for_the_core(wired, bot, store) -> None:
    dispatcher, _core, submissions = wired

    await feed(dispatcher, bot, message=_message(text="что у меня завтра?"))

    pending = await store.pending_requests()
    assert len(pending) == 1
    assert pending[0].payload["text"] == "что у меня завтра?"
    assert pending[0].user_id == "tg:1"
    # Persisted first, then the submitter is woken: an outage cannot lose the message.
    assert submissions.nudges == 1


async def test_the_handler_does_not_wait_for_an_answer(wired, bot) -> None:
    """No reply is sent inline; the answer arrives later as a telegram.send from the Core."""
    dispatcher, core, _ = wired

    await feed(dispatcher, bot, message=_message(text="привет"))

    assert bot.answers == []
    assert core.calls == []


async def test_an_unknown_user_is_ignored(wired, bot, store) -> None:
    dispatcher, _core, _ = wired
    stranger = Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=Chat(id=CHAT_ID, type="private"),
        from_user=User(id=999, is_bot=False, first_name="Кто-то"),
        text="пусти",
    ).as_(bot)

    await feed(dispatcher, bot, message=stranger)

    assert await store.pending_requests() == []
    assert bot.answers == []


async def test_start_and_help_answer_locally(wired, bot, store) -> None:
    """These need no Core: answering from the Gateway keeps help working during an outage."""
    dispatcher, core, _ = wired

    await feed(dispatcher, bot, message=_message(text="/start"))
    await feed(dispatcher, bot, message=_message(message_id=2, text="/help"))

    assert len(bot.answers) == 2
    assert core.calls == []
    assert await store.pending_requests() == []


async def test_new_asks_the_core_to_reset_the_session(wired, bot) -> None:
    dispatcher, core, _ = wired

    await feed(dispatcher, bot, message=_message(text="/new"))

    assert [method for method, _ in core.calls] == [methods.SESSION_RESET]
    assert "новый разговор" in bot.answers[0]


async def test_new_while_the_core_is_down_says_so(wired, bot) -> None:
    dispatcher, core, _ = wired
    core.connected = False

    await feed(dispatcher, bot, message=_message(text="/new"))

    assert core.calls == []
    assert "недоступно" in bot.answers[0]


async def test_cancel_is_queued_like_any_other_command(wired, bot, store) -> None:
    dispatcher, _core, _ = wired

    await feed(dispatcher, bot, message=_message(text="/cancel"))

    pending = await store.pending_requests()
    assert pending[0].payload["command"] == "/cancel"
    assert pending[0].payload["kind"] == "command"


async def test_status_shows_both_sides(wired, bot) -> None:
    dispatcher, core, _ = wired
    core.result = {
        "core": {"instance_id": "home-macmini"},
        "cursor": {"state": "ready"},
        "stt": {"state": "ready", "model": "large-v3"},
        "scheduler": {"state": "ready", "pending_reminders": 2},
        "jobs": {"running": 1, "queued": 2},
    }

    await feed(dispatcher, bot, message=_message(text="/status"))

    rendered = bot.answers[0]
    assert "Core: connected (home-macmini)" in rendered
    assert "Jobs: 1 running / 2 queued" in rendered
    assert "token" not in rendered.lower()


async def test_status_works_without_the_core(wired, bot, store) -> None:
    dispatcher, core, _ = wired
    core.connected = False
    await store.save_request(
        request_id=new_ulid(), user_id="tg:1", chat_id=CHAT_ID, message_id=1,
        kind="text", payload={},
    )

    await feed(dispatcher, bot, message=_message(text="/status"))

    assert "Core: disconnected" in bot.answers[0]
    assert "Отложенных запросов: 1" in bot.answers[0]


# ---- audio ---------------------------------------------------------------


def _voice(message_id: int = 5, bot=None) -> Message:
    return _message(
        message_id=message_id,
        bot=bot,
        voice=Voice(
            file_id="voice-1", file_unique_id="v1", duration=12, mime_type="audio/ogg",
            file_size=len(b"opus"),
        ),
    )


async def test_a_voice_note_is_downloaded_and_queued(wired, bot, store) -> None:
    dispatcher, _core, submissions = wired

    await feed(dispatcher, bot, message=_voice(bot=bot))

    uploads = await store.pending_uploads()
    assert len(uploads) == 1
    assert uploads[0].purpose == "assistant"
    assert uploads[0].file_path.read_bytes() == b"opus"
    # The digest is computed on the Gateway so the Core can prove the bytes arrived intact.
    assert uploads[0].sha256 == hashlib.sha256(b"opus").hexdigest()
    assert submissions.nudges == 1


async def test_an_audio_file_is_accepted_too(wired, bot, store) -> None:
    dispatcher, _core, _ = wired
    message = _message(
        message_id=6,
        bot=bot,
        audio=Audio(
            file_id="audio-1", file_unique_id="a1", duration=3600,
            mime_type="audio/mpeg", file_name="meeting.mp3", file_size=4,
        ),
    )

    await feed(dispatcher, bot, message=message)

    uploads = await store.pending_uploads()
    assert uploads[0].filename == "meeting.mp3"


async def test_a_pdf_is_not_pushed_through_whisper(wired, bot, store) -> None:
    dispatcher, _core, _ = wired
    message = _message(
        message_id=7,
        bot=bot,
        document=Document(
            file_id="doc-1", file_unique_id="d1", mime_type="application/pdf",
            file_name="contract.pdf", file_size=4,
        ),
    )

    await feed(dispatcher, bot, message=message)

    assert await store.pending_uploads() == []


async def test_a_file_over_the_limit_is_refused_before_downloading(
    wired, bot, store, settings
) -> None:
    dispatcher, _core, _ = wired
    message = _message(
        message_id=8,
        bot=bot,
        audio=Audio(
            file_id="huge", file_unique_id="h1", duration=99999,
            file_size=settings.max_download_bytes + 1,
        ),
    )

    await feed(dispatcher, bot, message=message)

    assert await store.pending_uploads() == []
    assert "слишком большой" in bot.answers[0]


async def test_transcribe_marks_the_upload_as_transcription_only(wired, bot, store) -> None:
    dispatcher, _core, _ = wired
    target = _voice(message_id=10, bot=bot)
    command = _message(message_id=11, text="/transcribe", bot=bot, reply_to_message=target)

    await feed(dispatcher, bot, message=command)

    uploads = await store.pending_uploads()
    assert uploads[0].purpose == "transcribe_only"


async def test_transcribe_without_a_reply_explains_itself(wired, bot, store) -> None:
    dispatcher, _core, _ = wired

    await feed(dispatcher, bot, message=_message(text="/transcribe", bot=bot))

    assert await store.pending_uploads() == []
    assert "Ответьте командой" in bot.answers[0]


# ---- confirmation callbacks ----------------------------------------------


async def _register(store, choice: str = "approve", user_id: str = "tg:1") -> str:
    action_id = new_ulid()
    tokens = await store.create_confirmation_tokens(
        action_id=action_id, user_id=user_id, chat_id=CHAT_ID, choices=[choice]
    )
    return tokens[choice]


def _query(token: str, bot=None, user_id: int = USER_ID) -> CallbackQuery:
    query = CallbackQuery(
        id="q1",
        from_user=User(id=user_id, is_bot=False, first_name="Илья"),
        chat_instance="ci",
        data=f"c:{token}",
        message=_message(message_id=42, text="Создать встречу?", bot=bot),
    )
    return query.as_(bot) if bot is not None else query


async def test_a_button_press_becomes_a_confirmation_resolve(wired, bot, store) -> None:
    dispatcher, core, _ = wired
    core.result = {"status": "applied"}
    token = await _register(store)

    await feed(dispatcher, bot, callback_query=_query(token, bot=bot))

    method, params = core.calls[0]
    assert method == methods.CONFIRMATION_RESOLVE
    assert params["choice"] == "approve"
    assert bot.callback_answers == ["Принято."]
    # The keyboard is stripped so the same decision cannot be pressed twice.
    assert bot.markup_cleared == 1


async def test_someone_elses_button_is_refused(wired, bot, store) -> None:
    dispatcher, core, _ = wired
    token = await _register(store, user_id="tg:2")

    await feed(dispatcher, bot, callback_query=_query(token, bot=bot))

    assert core.calls == []
    assert "не для вас" in bot.callback_answers[0]


async def test_an_unknown_token_is_reported_as_expired(wired, bot) -> None:
    dispatcher, core, _ = wired

    await feed(dispatcher, bot, callback_query=_query("nonexistent", bot=bot))

    assert core.calls == []
    assert "истёк" in bot.callback_answers[0]


async def test_a_second_press_is_not_forwarded_again(wired, bot, store) -> None:
    dispatcher, core, _ = wired
    core.result = {"status": "applied"}
    token = await _register(store)

    await feed(dispatcher, bot, callback_query=_query(token, bot=bot))
    await feed(dispatcher, bot, callback_query=_query(token, bot=bot))

    assert len(core.calls) == 1
    assert bot.callback_answers[-1] == "Уже обработано."


async def test_a_press_during_an_outage_is_not_silently_dropped(wired, bot, store) -> None:
    """Approving must never appear to succeed while the Core cannot hear it."""
    dispatcher, core, _ = wired
    core.connected = False
    token = await _register(store)

    await feed(dispatcher, bot, callback_query=_query(token, bot=bot))

    assert core.calls == []
    assert "недоступно" in bot.callback_answers[0]
    # Still usable once the Core is back.
    assert await store.resolve_confirmation_token(token) is not None


async def test_a_core_error_is_shown_and_the_token_stays_usable(wired, bot, store) -> None:
    dispatcher, core, _ = wired
    core.error = RpcError(errors.NOT_READY, "not_ready")
    token = await _register(store)

    await feed(dispatcher, bot, callback_query=_query(token, bot=bot))

    assert bot.callback_answers[0] == "Ядро сейчас недоступно. Запрос сохранён и будет обработан после переподключения."
    assert (await store.resolve_confirmation_token(token))["status"] != "used"
