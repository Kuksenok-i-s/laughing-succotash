"""Optional tests against the real Cursor Agent.

Skipped unless ``PA_REAL_CURSOR_TESTS=1``. An ordinary ``pytest`` run must never spend Cursor
usage, and CI must never depend on a logged-in Cursor install, so nothing here runs by default.

Run on the Mac mini before deploying, and again after any Cursor CLI upgrade:

    PA_REAL_CURSOR_TESTS=1 pytest tests/test_real_cursor.py -v

These are the claims from ``docs/cursor-acp.md`` that the rest of the Core is built on. If one of
them starts failing, the finding has changed and the code that relies on it needs revisiting —
which is exactly why they are assertions rather than prose.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from datetime import datetime, timezone

import pytest

from agent_core.agent.base import AgentContext, AgentError
from agent_core.agent.cursor_acp import CursorACPBackend

pytestmark = [
    pytest.mark.real_cursor,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("PA_REAL_CURSOR_TESTS") != "1",
        reason="set PA_REAL_CURSOR_TESTS=1 to run against the real Cursor Agent",
    ),
    pytest.mark.skipif(
        shutil.which(os.environ.get("CURSOR_AGENT_BINARY", "cursor-agent")) is None,
        reason="cursor-agent is not on PATH",
    ),
]

BINARY = os.environ.get("CURSOR_AGENT_BINARY", "cursor-agent")


@pytest.fixture
async def backend(tmp_path):
    agent = CursorACPBackend(
        BINARY,
        default_workspace=tmp_path,
        startup_timeout=120.0,
        prompt_timeout=300.0,
    )
    await agent.start()
    try:
        yield agent
    finally:
        await agent.close()


def context() -> AgentContext:
    return AgentContext(
        user_id="tg:test",
        conversation_id="conv-real",
        now=datetime.now(timezone.utc),
    )


async def test_the_binary_speaks_acp_and_reports_ready(backend) -> None:
    assert backend.state == "ready"


async def test_a_session_answers_a_question(backend, tmp_path) -> None:
    session = await backend.create_session(workspace=tmp_path, mcp_servers=[])

    response = await backend.send_message(
        session, "Ответь ровно одним словом: сколько будет два плюс два?", context()
    )

    assert response.text.strip()
    assert "4" in response.text or "четыре" in response.text.lower()


async def test_context_survives_within_a_session(backend, tmp_path) -> None:
    """The whole conversation model depends on this."""
    session = await backend.create_session(workspace=tmp_path, mcp_servers=[])

    await backend.send_message(
        session, "Запомни на этот разговор кодовое слово: КИРПИЧ. Ответь 'ок'.", context()
    )
    response = await backend.send_message(
        session, "Какое кодовое слово я просил запомнить? Ответь одним словом.", context()
    )

    assert "КИРПИЧ" in response.text.upper()


async def test_a_session_can_be_resumed_after_a_restart(tmp_path) -> None:
    """``session/load`` is what lets a conversation outlive the Core process."""
    first = CursorACPBackend(BINARY, default_workspace=tmp_path, startup_timeout=120.0)
    await first.start()
    try:
        session = await first.create_session(workspace=tmp_path, mcp_servers=[])
        await first.send_message(
            session, "Запомни число 8127. Ответь 'ок'.", context()
        )
    finally:
        await first.close()

    second = CursorACPBackend(BINARY, default_workspace=tmp_path, startup_timeout=120.0)
    await second.start()
    try:
        assert await second.resume_session(session, tmp_path, mcp_servers=[]) is True
        response = await second.send_message(
            session, "Какое число я просил запомнить? Ответь только числом.", context()
        )
        assert "8127" in response.text
    finally:
        await second.close()


async def test_cancellation_leaves_the_session_usable(backend, tmp_path) -> None:
    """``/cancel`` must not cost the user their conversation."""
    session = await backend.create_session(workspace=tmp_path, mcp_servers=[])

    task = asyncio.ensure_future(
        backend.send_message(
            session,
            "Напиши очень длинное эссе о истории почтовых голубей, минимум 3000 слов.",
            context(),
        )
    )
    await asyncio.sleep(5)
    await backend.cancel(session)

    with_result = await task
    assert with_result.cancelled or not with_result.text

    # The session still works, which is the property that matters.
    after = await backend.send_message(session, "Ответь одним словом: работаешь?", context())
    assert after.text.strip()


async def test_progress_reports_arrive_while_thinking(backend, tmp_path) -> None:
    session = await backend.create_session(workspace=tmp_path, mcp_servers=[])
    stages: list[str] = []

    async def on_progress(stage: str, detail: str | None) -> None:
        stages.append(stage)

    await backend.send_message(
        session,
        "Кратко объясни, что такое идемпотентность.",
        context(),
        on_progress=on_progress,
    )

    assert stages, "no progress was reported; the Telegram status message would never update"


async def test_an_unusable_binary_fails_loudly(tmp_path) -> None:
    """A missing Cursor must be an error the user can be told about, not a hang."""
    broken = CursorACPBackend(
        "definitely-not-a-real-binary", default_workspace=tmp_path, startup_timeout=5.0
    )
    with pytest.raises(AgentError):
        await broken.start()
