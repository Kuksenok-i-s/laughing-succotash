"""ACP client behaviour that does not need a live cursor-agent process."""

from __future__ import annotations

import asyncio
from typing import Any

from agent_core.agent.acp_client import AcpClient, acp_argv
from agent_core.agent.cursor_acp import CursorACPBackend, _TRAILING_IDLE, _chunk_text


class FakeAcp:
    running = True
    agent_capabilities: dict[str, Any] = {}

    def __init__(self, prompt) -> None:
        self._prompt = prompt
        self.cancelled: list[str] = []

    async def call(self, method: str, params: dict[str, Any], *, timeout: float | None = None):
        if method == "session/prompt":
            return await self._prompt(params)
        return {}

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        if method == "session/cancel":
            self.cancelled.append(params.get("sessionId", ""))


def _backend(tmp_path, client) -> CursorACPBackend:
    backend = CursorACPBackend(default_workspace=tmp_path, prompt_timeout=5.0)
    backend._client = client
    backend._state = "ready"
    backend._live_sessions.add("sess-1")
    return backend


async def _rpc(client: AcpClient, message: dict[str, Any]) -> dict[str, Any]:
    written: list[dict[str, Any]] = []

    async def write(payload: dict[str, Any]) -> None:
        written.append(payload)

    client._write = write  # type: ignore[method-assign]
    await client._handle_server_request(message)
    assert written, "ACP client sent no JSON-RPC reply"
    return written[0]


def test_acp_argv_pins_model_before_the_subcommand() -> None:
    assert acp_argv("cursor-agent") == ["cursor-agent", "acp"]
    assert acp_argv("cursor-agent", "cursor-grok-4.6-medium") == [
        "cursor-agent",
        "--model",
        "cursor-grok-4.6-medium",
        "acp",
    ]


def test_chunk_text_accepts_object_list_and_string() -> None:
    assert _chunk_text({"content": {"type": "text", "text": "привет"}}) == "привет"
    assert _chunk_text({"content": [{"type": "text", "text": "а"}, {"type": "text", "text": "б"}]}) == "аб"
    assert _chunk_text({"content": "готово"}) == "готово"
    assert _chunk_text({"content": None}) == ""


async def test_create_plan_is_accepted_not_empty_result() -> None:
    """An empty {} success is what hangs Cursor plan-mode turns."""
    seen: list[dict[str, Any]] = []

    async def on_plan(params: dict[str, Any]) -> None:
        seen.append(params)

    client = AcpClient()
    client.on_create_plan = on_plan
    reply = await _rpc(
        client,
        {
            "id": 7,
            "method": "cursor/create_plan",
            "params": {"name": "Шаги", "plan": "1. Сделать X"},
        },
    )
    assert reply["id"] == 7
    assert reply["result"]["outcome"]["outcome"] == "accepted"
    assert seen[0]["plan"] == "1. Сделать X"


async def test_ask_question_is_skipped_so_the_turn_unblocks() -> None:
    client = AcpClient()
    reply = await _rpc(
        client,
        {
            "id": 8,
            "method": "cursor/ask_question",
            "params": {
                "title": "Режим",
                "questions": [
                    {
                        "id": "q1",
                        "prompt": "Какой режим?",
                        "options": [{"id": "agent", "label": "Agent"}],
                    }
                ],
            },
        },
    )
    assert reply["result"]["outcome"]["outcome"] == "skipped"


async def test_unknown_blocking_method_is_method_not_found() -> None:
    client = AcpClient()
    reply = await _rpc(client, {"id": 9, "method": "cursor/nope", "params": {}})
    assert reply["error"]["code"] == -32601


async def test_late_chunks_after_end_turn_are_kept(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agent_core.agent.cursor_acp._TRAILING_IDLE", 0.15)
    monkeypatch.setattr("agent_core.agent.cursor_acp._TRAILING_MAX", 1.0)

    backend_holder: dict[str, CursorACPBackend] = {}

    async def prompt(params: dict[str, Any]) -> dict[str, Any]:
        session = params["sessionId"]

        async def late() -> None:
            await asyncio.sleep(0.05)
            await backend_holder["b"]._on_update(
                session,
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "поздний ответ"},
                },
            )

        asyncio.ensure_future(late())
        return {"stopReason": "end_turn"}

    backend = _backend(tmp_path, FakeAcp(prompt))
    backend_holder["b"] = backend

    response = await backend.send_message("sess-1", "привет")
    assert response.text == "поздний ответ"
    assert response.stop_reason == "end_turn"


async def test_create_plan_body_is_the_reply_when_no_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agent_core.agent.cursor_acp._TRAILING_IDLE", 0.05)
    monkeypatch.setattr("agent_core.agent.cursor_acp._TRAILING_MAX", 0.2)

    backend_holder: dict[str, CursorACPBackend] = {}

    async def prompt(params: dict[str, Any]) -> dict[str, Any]:
        await backend_holder["b"]._on_create_plan(
            {"sessionId": params["sessionId"], "name": "План", "plan": "1. Позвонить маме"}
        )
        return {"stopReason": "end_turn"}

    backend = _backend(tmp_path, FakeAcp(prompt))
    backend_holder["b"] = backend

    response = await backend.send_message("sess-1", "что дальше?")
    assert response.text == "1. Позвонить маме"


async def test_prompt_result_text_is_used_when_there_are_no_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agent_core.agent.cursor_acp._TRAILING_IDLE", 0.05)
    monkeypatch.setattr("agent_core.agent.cursor_acp._TRAILING_MAX", 0.2)

    async def prompt(_params: dict[str, Any]) -> dict[str, Any]:
        return {"stopReason": "end_turn", "text": "из result"}

    response = await _backend(tmp_path, FakeAcp(prompt)).send_message("sess-1", "hi")
    assert response.text == "из result"


def test_trailing_idle_is_short_enough_for_chat() -> None:
    assert _TRAILING_IDLE <= 0.5
