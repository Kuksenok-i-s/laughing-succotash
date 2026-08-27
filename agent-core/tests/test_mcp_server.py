"""End-to-end tests of the MCP server over real HTTP.

These go through the actual aiohttp app rather than calling handlers directly, because the parts
most likely to break — authorization, the session token that identifies the caller, argument
validation and the permission gate — all live in the request path.
"""

from __future__ import annotations

import json

import aiohttp
import pytest

from agent_core.agent.base import Provenance
from agent_core.calendar.local import LocalCalendarProvider
from agent_core.mcp.permissions import ToolContext
from agent_core.mcp.server import ContextRegistry, McpServer, ToolRegistry
from agent_core.mcp.tools import register_tools

TOKEN = "z" * 40


class RecordingConfirmations:
    def __init__(self, approve: bool = True) -> None:
        self.approve = approve
        self.requests: list[dict] = []

    async def request(self, **kwargs) -> bool:
        self.requests.append(kwargs)
        return self.approve


@pytest.fixture
async def mcp(repos):
    registry = ToolRegistry()
    contexts = ContextRegistry()
    confirmations = RecordingConfirmations()
    register_tools(
        registry, repos, calendar_provider=LocalCalendarProvider(repos.calendar)
    )
    server = McpServer(
        registry, contexts, repos.operations, confirmations,
        host="127.0.0.1", port=0, token=TOKEN,
    )
    await server.start()
    try:
        yield server, contexts, confirmations
    finally:
        await server.stop()


class Client:
    def __init__(self, server: McpServer, token: str) -> None:
        self._url = f"http://127.0.0.1:{server.port}/mcp/{token}"

    async def rpc(self, method: str, params=None, *, auth: str | None = TOKEN):
        headers = {"Authorization": f"Bearer {auth}"} if auth else {}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
                headers=headers,
            ) as response:
                return response.status, (
                    await response.json() if response.content_type == "application/json" else {}
                )

    async def call_tool(self, name: str, arguments: dict, **kwargs):
        status, body = await self.rpc(
            "tools/call", {"name": name, "arguments": arguments}, **kwargs
        )
        assert status == 200, body
        result = body["result"]
        payload = json.loads(result["content"][0]["text"])
        return payload, result.get("isError", False)


async def test_rejects_a_wrong_token(mcp) -> None:
    server, contexts, _ = mcp
    client = Client(server, contexts.issue_token("conv"))
    status, _ = await client.rpc("tools/list", auth="wrong")
    assert status == 401


async def test_lists_tools_with_schemas(mcp) -> None:
    server, contexts, _ = mcp
    client = Client(server, contexts.issue_token("conv"))
    status, body = await client.rpc("tools/list")

    assert status == 200
    names = {tool["name"] for tool in body["result"]["tools"]}
    assert {"reminder_create", "calendar_list", "note_search", "memory_remember",
            "contact_create"} <= names
    # A generic shell escape hatch would defeat the whole capability model.
    assert not any("shell" in name or "exec" in name for name in names)

    reminder = next(t for t in body["result"]["tools"] if t["name"] == "reminder_create")
    assert "text" in reminder["inputSchema"]["properties"]


async def test_a_tool_call_without_an_active_turn_is_refused(mcp) -> None:
    """A stale token must not be able to act: there is no user to act for."""
    server, contexts, _ = mcp
    client = Client(server, contexts.issue_token("conv"))
    payload, is_error = await client.call_tool("task_list", {})
    assert is_error
    assert "no active conversation context" in payload["error"]


async def test_read_tool_runs_without_confirmation(mcp, repos) -> None:
    server, contexts, confirmations = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))

    await repos.tasks.create(user_id="tg:1", title="Заменить SSD", operation_id="op-1")

    payload, is_error = await Client(server, token).call_tool("task_list", {})
    assert not is_error
    assert [task["title"] for task in payload["tasks"]] == ["Заменить SSD"]
    assert confirmations.requests == []


async def test_explicit_safe_write_runs_immediately(mcp, repos) -> None:
    server, contexts, confirmations = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))

    payload, is_error = await Client(server, token).call_tool(
        "reminder_create",
        {"text": "Выключить духовку", "due_at": "2027-01-01T10:00:00", "operation_id": "op-a"},
    )

    assert not is_error and payload["created"] is True
    assert confirmations.requests == []
    assert len(await repos.reminders.list("tg:1")) == 1


async def test_write_inferred_from_a_recording_asks_first(mcp, repos) -> None:
    server, contexts, confirmations = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.UNTRUSTED_CONTENT))

    payload, _ = await Client(server, token).call_tool(
        "calendar_create",
        {"title": "Встреча из записи", "starts_at": "2027-01-05T15:00:00",
         "operation_id": "op-b"},
    )

    assert payload["created"] is True
    assert len(confirmations.requests) == 1
    assert confirmations.requests[0]["tool_name"] == "calendar_create"
    assert "Встреча из записи" in confirmations.requests[0]["prompt_text"]


async def test_a_refused_confirmation_does_not_write(repos) -> None:
    registry = ToolRegistry()
    contexts = ContextRegistry()
    confirmations = RecordingConfirmations(approve=False)
    register_tools(registry, repos, calendar_provider=LocalCalendarProvider(repos.calendar))
    server = McpServer(
        registry, contexts, repos.operations, confirmations, port=0, token=TOKEN
    )
    await server.start()
    try:
        token = contexts.issue_token("conv")
        contexts.set_current("conv", _ctx(Provenance.UNTRUSTED_CONTENT))
        payload, is_error = await Client(server, token).call_tool(
            "task_create", {"title": "Из записи", "operation_id": "op-c"}
        )
        assert payload["status"] == "rejected"
        assert not is_error  # a refusal is a normal outcome, not a tool failure
        assert await repos.tasks.list("tg:1") == []
    finally:
        await server.stop()


async def test_dangerous_tool_asks_even_for_a_direct_command(mcp, repos) -> None:
    server, contexts, confirmations = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))
    client = Client(server, token)

    note, _ = await client.call_tool(
        "note_create", {"body": "мысль", "operation_id": "op-d"}
    )
    assert confirmations.requests == []

    await client.call_tool(
        "note_delete", {"note_id": note["note_id"], "operation_id": "op-e"}
    )
    assert [r["tool_name"] for r in confirmations.requests] == ["note_delete"]


async def test_replaying_an_operation_id_does_not_act_twice(mcp, repos) -> None:
    """The lost-response case: the Core acted, the answer never arrived, the caller retries."""
    server, contexts, _ = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))
    client = Client(server, token)

    args = {"title": "Планёрка", "starts_at": "2027-02-01T10:00:00", "operation_id": "op-same"}
    first, _ = await client.call_tool("calendar_create", args)
    second, _ = await client.call_tool("calendar_create", args)

    assert first["event_id"] == second["event_id"]
    events = await repos.calendar.list_range(
        "tg:1", _dt("2027-01-01"), _dt("2027-03-01")
    )
    assert len(events) == 1


async def test_invalid_arguments_are_reported_not_executed(mcp) -> None:
    server, contexts, _ = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))

    payload, is_error = await Client(server, token).call_tool("reminder_create", {})
    assert is_error
    assert payload["error"] == "invalid arguments"


async def test_a_tool_error_is_returned_as_a_tool_error(mcp) -> None:
    server, contexts, _ = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))

    payload, is_error = await Client(server, token).call_tool(
        "reminder_create", {"text": "что-то", "due_at": "не дата"}
    )
    assert is_error
    assert "TimeParseError" in payload["error"] or "could not parse" in payload["error"]


async def test_ambiguous_contacts_are_never_guessed(mcp, repos) -> None:
    server, contexts, _ = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))

    await repos.contacts.upsert(user_id="tg:1", display_name="Саша Иванов")
    await repos.contacts.upsert(user_id="tg:1", display_name="Саша Петров")

    payload, _ = await Client(server, token).call_tool("contact_search", {"query": "саша"})
    assert payload["count"] == 2
    assert payload["ambiguous"] is True
    assert "не гадай" in payload["guidance"].lower() or "do not guess" in payload["guidance"]


async def test_contact_create_runs_immediately_on_a_direct_command(mcp, repos) -> None:
    server, contexts, confirmations = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))

    payload, is_error = await Client(server, token).call_tool(
        "contact_create",
        {
            "display_name": "Саша Иванов",
            "aliases": ["@sasha"],
            "phones": ["+79990001122"],
            "operation_id": "op-contact",
        },
    )

    assert not is_error and payload["created"] is True
    assert payload["display_name"] == "Саша Иванов"
    assert payload["aliases"] == ["@sasha"]
    assert confirmations.requests == []
    stored = await repos.contacts.search("tg:1", "саша")
    assert len(stored) == 1


async def test_contact_update_changes_an_existing_contact(mcp, repos) -> None:
    server, contexts, _ = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))
    client = Client(server, token)

    created, _ = await client.call_tool(
        "contact_create", {"display_name": "Саша", "operation_id": "op-cu-1"}
    )
    updated, is_error = await client.call_tool(
        "contact_update",
        {"contact_id": created["contact_id"], "phones": ["+7999"], "operation_id": "op-cu-2"},
    )

    assert not is_error
    assert updated["display_name"] == "Саша"
    assert updated["phones"] == ["+7999"]


async def test_contact_create_from_a_recording_asks_first(mcp, repos) -> None:
    server, contexts, confirmations = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.UNTRUSTED_CONTENT))

    payload, _ = await Client(server, token).call_tool(
        "contact_create",
        {"display_name": "Маша из записи", "operation_id": "op-contact-untrusted"},
    )

    assert payload["created"] is True
    assert len(confirmations.requests) == 1
    assert confirmations.requests[0]["tool_name"] == "contact_create"
    assert "Маша из записи" in confirmations.requests[0]["prompt_text"]


async def test_replaying_a_contact_operation_id_does_not_create_twice(mcp, repos) -> None:
    server, contexts, _ = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))
    client = Client(server, token)
    args = {"display_name": "Петя", "operation_id": "op-contact-same"}

    first, _ = await client.call_tool("contact_create", args)
    second, _ = await client.call_tool("contact_create", args)

    assert first["contact_id"] == second["contact_id"]
    assert len(await repos.contacts.search("tg:1", "петя")) == 1


def _ctx(provenance: Provenance) -> ToolContext:
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    return ToolContext(
        user_id="tg:1",
        conversation_id="conv",
        provenance=provenance,
        chat_id=100,
        timezone=ZoneInfo("Europe/Moscow"),
        now=datetime.now(timezone.utc),
    )


def _dt(text: str):
    from datetime import datetime, timezone

    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
