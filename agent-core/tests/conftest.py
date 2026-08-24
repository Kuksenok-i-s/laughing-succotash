"""Shared fixtures and fakes for the Agent Core tests.

The fakes model the *semantics* of the real components rather than just recording calls: the fake
Gateway goes through the durable event log and dedupes on delivery_id exactly as the real link
does, so tests of "reminder fires while the Gateway is offline" exercise the real code path.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_core.agent.base import AgentContext, AgentResponse
from agent_core.config import Settings
from agent_core.storage.database import Database
from agent_core.storage.repositories import Repositories


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        instance_id="test-core",
        gateway_url="ws://localhost:8/rpc",
        core_token="x" * 40,
        mcp_token="y" * 40,
        allowed_users=["tg:1", "tg:2"],
        data_dir=tmp_path / "data",
        default_timezone="Europe/Moscow",
        scheduler_tick_seconds=0.05,
        confirmation_timeout_seconds=2,
        long_transcript_chars=200,
        transcript_chunk_chars=400,
    )


@pytest.fixture
async def db(settings: Settings) -> Database:
    database = Database(settings.resolved_database_path)
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture
async def repos(db: Database, settings: Settings) -> Repositories:
    return Repositories.build(db, settings.default_timezone)


class FakeGateway:
    """Stands in for ``GatewayLink``.

    Events go through the durable log, so ``online = False`` genuinely queues them and ``drain``
    genuinely replays them — including the refusal to enqueue a duplicate delivery_id.
    """

    def __init__(self, events) -> None:
        self._events = events
        self.online = True
        self.delivered: list[tuple[str, dict]] = []
        self.notifications: list[tuple[str, dict]] = []

    async def send_event(self, method, params, *, delivery_id=None, user_id=None) -> None:
        event = await self._events.enqueue(
            method, params, delivery_id=delivery_id, user_id=user_id
        )
        if event is None:
            return
        if self.online:
            await self._deliver(event.seq, method, params)

    async def _deliver(self, seq, method, params) -> None:
        self.delivered.append((method, params))
        await self._events.mark_sent(seq)

    async def drain(self) -> int:
        pending = await self._events.pending()
        for event in pending:
            await self._deliver(event.seq, event.method, event.params)
        return len(pending)

    async def notify(self, method, params) -> None:
        self.notifications.append((method, params))

    async def call(self, method, params, *, timeout=None):
        self.delivered.append((method, params))
        return {}

    def texts(self) -> list[str]:
        return [
            params["text"]
            for method, params in self.delivered
            if method == "telegram.send"
        ]


@pytest.fixture
def gateway(repos: Repositories) -> FakeGateway:
    return FakeGateway(repos.events)


class FakeBackend:
    """A scripted agent.

    ``on_prompt`` receives the prompt text and returns either a string reply or an
    ``AgentResponse``, which lets a test make the agent call a tool mid-turn.
    """

    def __init__(self, reply: str = "ok", on_prompt=None) -> None:
        self.reply = reply
        self.on_prompt = on_prompt
        self.prompts: list[tuple[str, str]] = []
        self.sessions: list[str] = []
        self.cancelled: list[str] = []
        self.started = False
        self._counter = 0

    @property
    def name(self) -> str:
        return "fake"

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
        self.prompts.append((session_id, message))
        if on_progress is not None:
            await on_progress("executing_tool", "fake tool")
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


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


def context_for(user_id: str = "tg:1", conversation_id: str = "conv-1", **fields) -> AgentContext:
    return AgentContext(
        user_id=user_id,
        conversation_id=conversation_id,
        now=datetime.now(timezone.utc),
        **fields,
    )
