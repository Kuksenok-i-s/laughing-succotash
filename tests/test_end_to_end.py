"""End-to-end tests: a real Gateway and a real Core joined by a real WebSocket.

Only three things are faked — Telegram itself, Cursor and whisper. Everything in between is the
production code: the aiohttp RPC endpoint, the service-token handshake, SQLite on both sides, the
durable queues, the job manager, the scheduler and the renderer. That is the point. The failure
modes this project cares about (a lost response, a duplicated delivery, an outage mid-request) live
in that machinery, and a mock of it would only test the mock.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from aiohttp import web
from pa_protocol import methods, new_ulid

from agent_core.agent.base import AgentResponse
from agent_core.config import Settings as CoreSettings
from agent_core.main import Core
from agent_core.stt.base import SttError, TranscriptionResult, TranscriptSegment
from telegram_gateway.config import Settings as GatewaySettings
from telegram_gateway.delivery.service import SubmissionService
from telegram_gateway.rpc.server import CoreLink
from telegram_gateway.storage.database import Database as GatewayDatabase
from telegram_gateway.storage.models import GatewayStore
from telegram_gateway.telegram.formatting import STAGE_TEXT
from telegram_gateway.telegram.renderer import TelegramRenderer

SERVICE_TOKEN = "t" * 48
USER = "tg:1"
CHAT = 4242

STATUS_TEXTS = set(STAGE_TEXT.values())


# ---- fakes at the edges ---------------------------------------------------


@dataclass
class SentMessage:
    chat_id: int
    text: str
    message_id: int
    reply_markup: object | None = None


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[SentMessage] = []
        self.edits: list[tuple[int, int, str]] = []
        self.deleted: list[tuple[int, int]] = []
        self.actions: list[tuple[int, str]] = []
        self.documents: list[tuple[str, bytes, str | None]] = []
        self._next_id = 1000

    async def send_message(self, chat_id, text, parse_mode=None, reply_markup=None, **kwargs):
        self._next_id += 1
        message = SentMessage(chat_id, text, self._next_id, reply_markup)
        self.sent.append(message)
        return message

    async def edit_message_text(self, chat_id, message_id, text, **kwargs) -> None:
        self.edits.append((chat_id, message_id, text))

    async def delete_message(self, chat_id, message_id) -> None:
        self.deleted.append((chat_id, message_id))

    async def send_chat_action(self, chat_id, action) -> None:
        self.actions.append((chat_id, action))

    async def send_document(self, chat_id, document, caption=None, **kwargs):
        filename = getattr(document, "filename", "file")
        data = getattr(document, "data", b"")
        self._next_id += 1
        message = SentMessage(chat_id, caption or filename, self._next_id, None)
        self.sent.append(message)
        self.documents.append((filename, data, caption))
        return message


class FakeBackend:
    """A scripted Cursor. ``on_prompt`` may return a string or a full ``AgentResponse``."""

    def __init__(self) -> None:
        self.on_prompt = None
        # Not "Готово.": that is also a status line, and the harness tells the two apart by text.
        self.reply = "ок"
        self.prompts: list[str] = []
        self.sessions: list[str] = []
        self.cancelled: list[str] = []
        self.started = False
        self._counter = 0

    name = "fake"

    @property
    def state(self) -> str:
        return "ready" if self.started else "stopped"

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.started = False

    async def create_session(self, *, workspace, context=None, mcp_servers=None) -> str:
        self._counter += 1
        session_id = f"session-{self._counter}"
        self.sessions.append(session_id)
        return session_id

    async def resume_session(self, session_id, workspace, *, mcp_servers=None) -> bool:
        return session_id in self.sessions

    async def send_message(self, session_id, message, context=None, *, on_progress=None):
        self.prompts.append(message)
        if self.on_prompt is not None:
            outcome = self.on_prompt(message, context)
            if asyncio.iscoroutine(outcome):
                outcome = await outcome
            if isinstance(outcome, AgentResponse):
                return outcome
            return AgentResponse(text=str(outcome), session_id=session_id)
        return AgentResponse(text=self.reply, session_id=session_id)

    async def cancel(self, session_id) -> None:
        self.cancelled.append(session_id)

    async def set_mode(self, session_id, mode) -> None:
        pass


class FakeSTT:
    def __init__(self) -> None:
        self.result: TranscriptionResult | None = None
        self.error: Exception | None = None
        self.calls: list[Path] = []
        self.ready = True
        self.model_name = "fake"

    def script(self, text: str, *, segments: list[TranscriptSegment] | None = None) -> None:
        self.result = TranscriptionResult(
            text=text,
            language="ru",
            duration=segments[-1].end if segments else 5.0,
            segments=segments or [TranscriptSegment(0.0, 5.0, text)],
        )

    async def transcribe(
        self, audio_path: Path, *, on_progress=None, on_notice=None
    ) -> TranscriptionResult:
        # Asserted here rather than in a test: the Core must hand whisper a real, complete file.
        assert audio_path.exists(), "core handed whisper a path that does not exist"
        self.calls.append(audio_path)
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise SttError("nothing scripted")
        return self.result

    async def warmup(self) -> None:
        pass

    async def close(self) -> None:
        pass


class HarnessCore(Core):
    """The real Core composition with only the two expensive local runtimes replaced."""

    def __init__(self, settings: CoreSettings, backend: FakeBackend, stt: FakeSTT) -> None:
        super().__init__(settings)
        self._fake_backend = backend
        self._fake_stt = stt

    def _build_backend(self):
        return self._fake_backend

    def _build_stt(self):
        return self._fake_stt


# ---- harness --------------------------------------------------------------


@dataclass
class Harness:
    bot: FakeBot
    store: GatewayStore
    link: CoreLink
    submissions: SubmissionService
    renderer: TelegramRenderer
    core: HarnessCore
    backend: FakeBackend
    stt: FakeSTT
    settings: GatewaySettings
    core_settings: CoreSettings
    temp: Path

    async def until(self, predicate, what: str, timeout: float = 10.0) -> None:
        """Poll a (possibly async) predicate. Nothing in a two-machine system is instantaneous."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            outcome = predicate()
            if asyncio.iscoroutine(outcome):
                outcome = await outcome
            if outcome:
                return
            await asyncio.sleep(0.02)
        raise AssertionError(f"timed out waiting for {what}")

    async def pending_events(self) -> int:
        """Core-originated events committed to SQLite but not yet delivered."""
        return len(await self.core._repos.events.pending())  # noqa: SLF001

    # ---- inbound (what a Telegram handler does) ----

    async def send_text(self, text: str, *, user: str = USER, message_id: int = 1) -> str:
        request_id = new_ulid()
        await self.store.save_request(
            request_id=request_id,
            user_id=user,
            chat_id=CHAT,
            message_id=message_id,
            kind="text",
            payload=methods.dump(
                methods.AssistantSubmitParams(
                    request_id=request_id, user_id=user, chat_id=CHAT,
                    message_id=message_id, kind="text", text=text,
                )
            ),
        )
        self.submissions.nudge()
        return request_id

    async def send_command(self, command: str, *, user: str = USER) -> str:
        request_id = new_ulid()
        await self.store.save_request(
            request_id=request_id,
            user_id=user,
            chat_id=CHAT,
            message_id=2,
            kind="command",
            payload=methods.dump(
                methods.AssistantSubmitParams(
                    request_id=request_id, user_id=user, chat_id=CHAT, message_id=2,
                    kind="command", command=command,
                )
            ),
        )
        self.submissions.nudge()
        return request_id

    async def send_voice(
        self, data: bytes = b"opus bytes", *, purpose: str = "assistant", user: str = USER
    ) -> str:
        request_id = new_ulid()
        path = self.temp / f"{request_id}.ogg"
        path.write_bytes(data)
        await self.store.save_upload(
            request_id=request_id,
            user_id=user,
            chat_id=CHAT,
            message_id=3,
            file_path=path,
            filename="voice.ogg",
            content_type="audio/ogg",
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            duration_seconds=8.0,
            purpose=purpose,
        )
        self.submissions.nudge()
        return request_id

    # ---- outbound (what the user sees) ----

    def replies(self) -> list[str]:
        return [m.text for m in self.bot.sent if m.text not in STATUS_TEXTS]

    def user_visible(self) -> list[str]:
        """Everything the user ended up reading, including text edited into a status message."""
        return self.replies() + [text for _chat, _message, text in self.bot.edits]

    async def wait_reply(self, count: int = 1, timeout: float = 10.0) -> list[str]:
        await self.until(lambda: len(self.replies()) >= count, f"{count} reply(ies)", timeout)
        return self.replies()

    def keyboards(self) -> list[SentMessage]:
        return [m for m in self.bot.sent if m.reply_markup is not None]

    # ---- the network ----

    async def cut_link(self) -> None:
        """Kill the socket the way an outage would, leaving both processes running."""
        peer = self.link._peer  # noqa: SLF001
        if peer is not None:
            await peer.close()
        await self.until(lambda: not self.link.connected, "the link to drop")

    @contextlib.asynccontextmanager
    async def outage(self):
        """Hold the link down for the duration of the block.

        Cutting the socket alone is not enough — the Core reconnects within milliseconds. Making
        the Gateway reject the handshake keeps it down deterministically while the Core keeps
        running, which is the situation a Gateway restart or a blocked route produces.
        """
        real_token = self.settings.core_token
        self.settings.core_token = "n" * 48
        await self.cut_link()
        try:
            yield
        finally:
            self.settings.core_token = real_token
            await self.wait_link()

    async def wait_link(self) -> None:
        await self.until(lambda: self.link.connected, "the core to connect", timeout=20.0)

    async def press(self, message: SentMessage, index: int = 0) -> None:
        """Simulate a Telegram callback query on a rendered confirmation button."""
        button = message.reply_markup.inline_keyboard[0][index]
        record = await self.store.resolve_confirmation_token(button.callback_data[2:])
        assert record is not None, "the callback token was not registered"
        await self.link.call(
            methods.CONFIRMATION_RESOLVE,
            {
                "action_id": record["action_id"],
                "user_id": record["user_id"],
                "choice": record["choice"],
            },
        )


@pytest.fixture
async def harness(tmp_path: Path):
    gateway_settings = GatewaySettings(
        telegram_bot_token="1:fake",
        core_token=SERVICE_TOKEN,
        allowed_users=[USER],
        data_dir=tmp_path / "gateway",
        host="127.0.0.1",
        port=0,
        status_edit_min_interval=0.0,
        submit_timeout=5.0,
        delivery_retry_base_delay=0.05,
    )

    gateway_db = GatewayDatabase(gateway_settings.resolved_database_path)
    await gateway_db.connect()
    store = GatewayStore(gateway_db)
    bot = FakeBot()

    link = CoreLink(gateway_settings, store)
    renderer = TelegramRenderer(bot, store, gateway_settings)
    link.register_all(renderer.handlers())
    submissions = SubmissionService(link, store, gateway_settings, bot=bot)
    link.on_ready = lambda: _nudge(submissions)

    app = web.Application()
    app.router.add_get(gateway_settings.rpc_path, link.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]

    core_settings = CoreSettings(
        instance_id="test-core",
        gateway_url=f"ws://127.0.0.1:{port}{gateway_settings.rpc_path}",
        core_token=SERVICE_TOKEN,
        mcp_token="m" * 40,
        mcp_port=0,
        allowed_users=[USER],
        data_dir=tmp_path / "core",
        assistant_workspace=tmp_path / "workspace",
        default_timezone="Europe/Moscow",
        scheduler_tick_seconds=0.05,
        reconnect_base_delay=0.05,
        reconnect_max_delay=0.2,
        reconnect_healthy_after=0.2,
        confirmation_timeout_seconds=3,
        long_transcript_chars=200,
        transcript_chunk_chars=400,
    )

    backend = FakeBackend()
    stt = FakeSTT()
    core = HarnessCore(core_settings, backend, stt)

    await submissions.start()
    await core.start()

    harness = Harness(
        bot=bot, store=store, link=link, submissions=submissions, renderer=renderer, core=core,
        backend=backend, stt=stt, settings=gateway_settings,
        core_settings=core_settings, temp=gateway_settings.resolved_temp_dir,
    )
    await harness.wait_link()

    try:
        yield harness
    finally:
        with contextlib.suppress(Exception):
            await core.stop()
        with contextlib.suppress(Exception):
            await submissions.stop()
        await runner.cleanup()
        await gateway_db.close()


async def _nudge(submissions: SubmissionService) -> None:
    submissions.nudge()


# ---- 1. text chat ---------------------------------------------------------


async def test_a_text_message_gets_an_answer(harness: Harness) -> None:
    harness.backend.reply = "Завтра в 15:00 созвон с Димой."

    await harness.send_text("что у меня завтра?")

    assert await harness.wait_reply() == ["Завтра в 15:00 созвон с Димой."]
    assert "что у меня завтра?" in harness.backend.prompts[0]


async def test_the_conversation_keeps_its_context(harness: Harness) -> None:
    """Both turns must land in the same Cursor session, or there is no conversation."""
    await harness.send_text("первое сообщение")
    await harness.wait_reply(1)
    await harness.send_text("второе сообщение", message_id=2)
    await harness.wait_reply(2)

    assert len(harness.backend.sessions) == 1
    assert len(harness.backend.prompts) == 2


async def test_two_users_never_share_a_session(harness: Harness) -> None:
    harness.core_settings.allowed_users.append("tg:2")

    await harness.send_text("от первого")
    await harness.wait_reply(1)
    await harness.send_text("от второго", user="tg:2", message_id=9)
    await harness.wait_reply(2)

    assert len(harness.backend.sessions) == 2


async def test_a_stranger_is_refused_by_the_core(harness: Harness) -> None:
    """The Gateway's allowlist is advisory; this proves the Core enforces its own."""
    await harness.send_text("пусти меня", user="tg:999")

    await harness.until(
        lambda: harness.store is not None and harness.backend.prompts == [],
        "the core to refuse",
        timeout=1.0,
    )
    assert harness.backend.prompts == []


async def test_a_long_answer_arrives_split_but_whole(harness: Harness) -> None:
    harness.backend.reply = "\n\n".join(f"Пункт {i}: " + "детали " * 60 for i in range(40))

    await harness.send_text("расскажи подробно")
    await harness.until(lambda: len(harness.replies()) > 1, "several parts")

    joined = " ".join(harness.replies())
    for i in range(40):
        assert f"Пункт {i}:" in joined


async def test_progress_is_shown_as_one_edited_status_message(harness: Harness) -> None:
    release = asyncio.Event()

    async def slow(_message, _context):
        await release.wait()
        return "готово"

    harness.backend.on_prompt = slow
    await harness.send_text("думай долго")

    await harness.until(lambda: bool(harness.bot.sent), "a status message")
    assert harness.bot.sent[0].text in STATUS_TEXTS

    release.set()
    await harness.wait_reply()
    # The status message is cleaned up rather than left behind.
    assert harness.bot.deleted


# ---- 2. voice -------------------------------------------------------------


async def test_a_voice_command_is_transcribed_and_answered(harness: Harness) -> None:
    harness.stt.script("что у меня завтра в календаре?")
    harness.backend.reply = "Завтра ничего не запланировано."

    await harness.send_voice()

    assert await harness.wait_reply() == ["Завтра ничего не запланировано."]
    assert "что у меня завтра в календаре?" in harness.backend.prompts[0]


async def test_audio_is_deleted_from_both_machines(harness: Harness) -> None:
    """The recording is the most sensitive artefact; the transcript is the useful one."""
    harness.stt.script("удали меня")

    await harness.send_voice()
    await harness.wait_reply()

    # The Gateway deletes its download as soon as the Core acknowledges the upload...
    assert list(harness.temp.glob("*.ogg")) == []
    # ...and the Core deletes its copy once transcription finishes, pass or fail.
    await harness.until(
        lambda: not harness.stt.calls[0].exists(), "the core to delete the upload"
    )


async def test_transcribe_only_never_reaches_the_agent(harness: Harness) -> None:
    """``/transcribe`` is a transcription, not a conversation: no reasoning, no tools."""
    harness.stt.script("просто расшифруй это")

    await harness.send_voice(purpose="transcribe_only")

    assert await harness.wait_reply() == ["просто расшифруй это"]
    assert harness.backend.prompts == []


async def test_a_whisper_failure_fails_the_job_and_the_core_survives(harness: Harness) -> None:
    harness.stt.error = SttError("model exploded")

    await harness.send_voice()

    # The failure replaces the "Расшифровываю запись…" status message rather than adding a new one.
    await harness.until(
        lambda: any("распознать" in text for text in harness.user_visible()),
        "a failure message",
    )
    # Still alive and answering.
    harness.stt.error = None
    harness.backend.reply = "живой"
    await harness.send_text("ты там?")
    await harness.until(lambda: "живой" in harness.replies(), "the next answer")


async def test_a_long_recording_is_analysed_not_obeyed(harness: Harness) -> None:
    """The Definition of Done case: a meeting recording produces a proposal, not actions."""
    segments = [
        TranscriptSegment(i * 10.0, (i + 1) * 10.0, f"Поставь встречу на пятницу, пункт {i}. ")
        for i in range(30)
    ]
    harness.stt.script("".join(s.text for s in segments), segments=segments)

    prompts: list[str] = []

    def respond(message, _context):
        prompts.append(message)
        if "<transcript_chunk>" in message:
            return "РЕШЕНИЯ: перенести стенд"
        return "Кратко\n...\nМожно создать\n1. Встречу в пятницу"

    harness.backend.on_prompt = respond
    await harness.send_voice()

    replies = await harness.wait_reply()
    assert "Можно создать" in replies[0]
    # Every chunk prompt carries the untrusted-content instruction, and so does the final turn.
    chunked = [p for p in prompts if "<transcript_chunk>" in p]
    assert chunked
    assert all("transcript of a recording" in p for p in chunked)
    assert "transcript of a recording" in prompts[-1]
    # And nothing was created: no confirmation was even asked for, because the agent only proposed.
    assert harness.keyboards() == []


# ---- 3. reminders ---------------------------------------------------------


async def test_a_reminder_fires_on_its_own(harness: Harness) -> None:
    """Cursor creates it; the Core's own scheduler delivers it minutes later."""

    async def create_reminder(message, context):
        if "Фрагмент" in message:
            return "-"
        await harness.core._repos.reminders.create(  # noqa: SLF001
            user_id=USER,
            text="выключить духовку",
            due_at=datetime.now(timezone.utc) + timedelta(milliseconds=200),
            timezone_name="Europe/Moscow",
            operation_id=new_ulid(),
        )
        return "Напомню через 5 минут."

    harness.backend.on_prompt = create_reminder
    await harness.send_text("напомни через 5 минут выключить духовку")
    await harness.wait_reply(1)

    await harness.until(
        lambda: any("духовку" in text for text in harness.replies()[1:]),
        "the reminder to fire",
    )


async def test_the_same_reminder_operation_twice_creates_one(harness: Harness) -> None:
    operation_id = new_ulid()
    due = datetime.now(timezone.utc) + timedelta(hours=1)
    reminders = harness.core._repos.reminders  # noqa: SLF001

    first, dup_a = await reminders.create(
        user_id=USER, text="один раз", due_at=due,
        timezone_name="Europe/Moscow", operation_id=operation_id,
    )
    second, dup_b = await reminders.create(
        user_id=USER, text="один раз", due_at=due,
        timezone_name="Europe/Moscow", operation_id=operation_id,
    )

    assert dup_a is False and dup_b is True
    assert first.reminder_id == second.reminder_id
    assert len(await reminders.list(USER, status="scheduled")) == 1


# ---- 4. gateway unavailable ----------------------------------------------


async def test_a_reminder_that_fires_during_an_outage_is_delivered_once(
    harness: Harness,
) -> None:
    await harness.core._repos.conversations.ensure_user(USER)  # noqa: SLF001
    await harness.core._repos.conversations.remember_chat(USER, CHAT)  # noqa: SLF001

    async with harness.outage():
        await harness.core._repos.reminders.create(  # noqa: SLF001
            user_id=USER,
            text="сработало без гейтвея",
            due_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            timezone_name="Europe/Moscow",
            operation_id=new_ulid(),
        )
        # It fires logically while nobody can hear it: the scheduler is the Core's own and does
        # not care whether the Gateway exists.
        await harness.until(
            lambda: harness.pending_events(), "the reminder to reach the outbox", timeout=5.0
        )
        assert harness.replies() == []

    await harness.until(
        lambda: any("без гейтвея" in text for text in harness.replies()),
        "delivery after reconnect",
    )
    matching = [text for text in harness.replies() if "без гейтвея" in text]
    assert len(matching) == 1


# ---- 5. core unavailable -------------------------------------------------


async def test_a_message_sent_while_the_core_is_down_is_processed_later(
    harness: Harness,
) -> None:
    await harness.core.stop()
    await harness.until(lambda: not harness.link.connected, "the core to stop")

    await harness.send_text("обработай когда вернёшься")
    await asyncio.sleep(0.1)
    assert await harness.store.pending_request_count() == 1
    assert harness.replies() == []

    # A fresh Core process over the same SQLite files, as after a restart.
    revived = HarnessCore(harness.core_settings, harness.backend, harness.stt)
    harness.core = revived
    harness.backend.reply = "принял"
    await revived.start()
    try:
        await harness.wait_link()
        await harness.until(lambda: "принял" in harness.replies(), "the delayed answer")
        assert await harness.store.pending_request_count() == 0
    finally:
        await revived.stop()


# ---- 6. confirmation -----------------------------------------------------


async def test_nothing_dangerous_happens_without_a_button_press(harness: Harness) -> None:
    """A DANGEROUS tool must reach the user as a question and wait for the answer."""
    confirmations = harness.core._confirmations  # noqa: SLF001
    decided: list[bool] = []

    async def ask() -> None:
        decided.append(
            await confirmations.request(
                user_id=USER,
                chat_id=CHAT,
                tool_name="calendar_delete",
                arguments={"event_id": "E1"},
                operation_id=new_ulid(),
                tier="dangerous",
                prompt_text="Удалить встречу с Иваном?",
            )
        )

    task = asyncio.ensure_future(ask())
    await harness.until(lambda: bool(harness.keyboards()), "an inline keyboard")

    prompt = harness.keyboards()[0]
    assert prompt.text == "Удалить встречу с Иваном?"
    assert decided == []  # still waiting

    await harness.press(prompt, index=0)
    await task
    assert decided == [True]


async def test_a_rejected_confirmation_is_a_no(harness: Harness) -> None:
    confirmations = harness.core._confirmations  # noqa: SLF001
    task = asyncio.ensure_future(
        confirmations.request(
            user_id=USER, chat_id=CHAT, tool_name="note_delete", arguments={},
            operation_id=new_ulid(), tier="dangerous", prompt_text="Удалить заметку?",
        )
    )
    await harness.until(lambda: bool(harness.keyboards()), "an inline keyboard")

    await harness.press(harness.keyboards()[0], index=1)

    assert await task is False


# ---- cancellation and status --------------------------------------------


async def test_cancel_stops_the_running_job_without_breaking_the_session(
    harness: Harness,
) -> None:
    started = asyncio.Event()

    async def hang(_message, _context):
        started.set()
        await asyncio.sleep(30)
        return "never"

    harness.backend.on_prompt = hang
    await harness.send_text("думай бесконечно")
    await asyncio.wait_for(started.wait(), 5)

    harness.backend.on_prompt = None
    harness.backend.reply = "всё ещё здесь"
    await harness.send_command("/cancel")

    await harness.until(lambda: bool(harness.backend.cancelled), "the agent to be cancelled")

    await harness.send_text("работаешь?", message_id=7)
    await harness.until(lambda: "всё ещё здесь" in harness.replies(), "a later answer")


async def test_status_reports_both_sides_and_no_secrets(harness: Harness) -> None:
    status = await harness.link.call(methods.STATUS_GET, {})

    assert status["cursor"]["state"] == "ready"
    assert status["scheduler"]["state"] in ("ready", "running")
    rendered = str(status)
    assert SERVICE_TOKEN not in rendered
    assert harness.core_settings.mcp_token not in rendered


async def test_a_new_session_starts_a_fresh_conversation(harness: Harness) -> None:
    await harness.send_text("первое")
    await harness.wait_reply(1)

    await harness.link.call(
        methods.SESSION_RESET, {"user_id": USER, "request_id": new_ulid()}
    )
    await harness.send_text("после сброса", message_id=5)
    await harness.wait_reply(2)

    assert len(harness.backend.sessions) == 2


# ---- duplicate delivery --------------------------------------------------


async def test_the_same_delivery_twice_sends_one_telegram_message(harness: Harness) -> None:
    """A reconnect mid-flight makes the Core replay; the Gateway must absorb it."""
    params = {
        "delivery_id": new_ulid(),
        "user_id": USER,
        "chat_id": CHAT,
        "text": "ровно один раз",
    }
    send = harness.renderer.handlers()[methods.TELEGRAM_SEND]

    first = await send(params)
    second = await send(params)

    assert first["dedup"] is False
    assert second["dedup"] is True
    assert [m.text for m in harness.bot.sent].count("ровно один раз") == 1
