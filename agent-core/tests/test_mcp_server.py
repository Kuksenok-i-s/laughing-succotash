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
            "contact_create", "file_send", "file_list", "file_read",
            "training_log_save", "training_athlete_list", "training_export",
            "training_profile_get", "training_profile_set", "training_progress"} <= names
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


async def test_training_log_save_persists_and_remembers_the_journal(mcp, repos) -> None:
    server, contexts, confirmations = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))
    client = Client(server, token)

    payload, is_error = await client.call_tool(
        "training_log_save",
        {
            "athlete_name": "Вася",
            "local_date": "2026-08-30",
            "raw_text": "присед 4 по 8 на 80",
            "exercises": [
                {"name": "Присед", "sets": [{"reps": 8, "weight_kg": 80} for _ in range(4)]}
            ],
            "operation_id": "op-train-1",
        },
    )

    assert not is_error and payload["created"] is True
    assert payload["athlete_name"] == "Вася"
    assert payload["exercises"][0]["sets"][0]["weight_kg"] == 80
    assert payload["progress"]["done"] == 1
    assert "проведено 1" in payload["progress"]["label"]
    assert confirmations.requests == []
    assert await repos.training.is_enabled("tg:1") is True
    profile = await repos.training.get_profile("tg:1")
    assert profile is not None and profile["mode"] == "trainer"
    memories = await repos.memory.search("tg:1", "журнал тренировок")
    assert len(memories) == 1
    assert memories[0]["category"] == "training"


async def test_training_progress_counts_done_and_remaining(mcp, repos) -> None:
    server, contexts, _ = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))
    client = Client(server, token)

    program, is_error = await client.call_tool(
        "training_program_upsert",
        {
            "athlete_name": "Вася",
            "title": "Сила 4 недели",
            "days_per_week": 3,
            "weeks": 4,
            "started_on": "2026-08-01",
            "operation_id": "prog-len",
        },
    )
    assert not is_error
    assert program["total_sessions"] == 12
    assert program["progress"]["remaining"] == 12

    log, is_error = await client.call_tool(
        "training_log_save",
        {
            "athlete_name": "Вася",
            "local_date": "2026-08-30",
            "exercises": [{"name": "Присед", "sets": [{"reps": 5, "weight_kg": 100}]}],
            "operation_id": "log-len",
        },
    )
    assert not is_error
    assert log["progress"]["done"] == 1
    assert log["progress"]["remaining"] == 11
    assert log["progress"]["label"] == "проведено 1 из 12, осталось 11"

    payload, is_error = await client.call_tool(
        "training_progress", {"athlete_name": "Вася"}
    )
    assert not is_error
    assert payload["progress"]["done"] == 1
    assert payload["progress"]["remaining"] == 11


async def test_training_log_from_a_recording_asks_first(mcp, repos) -> None:
    server, contexts, confirmations = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.UNTRUSTED_CONTENT))

    payload, _ = await Client(server, token).call_tool(
        "training_log_save",
        {
            "athlete_name": "Вася",
            "local_date": "2026-08-30",
            "exercises": [{"name": "Жим", "sets": [{"reps": 5, "weight_kg": 60}]}],
            "operation_id": "op-train-untrusted",
        },
    )

    assert payload["created"] is True
    assert [item["tool_name"] for item in confirmations.requests] == ["training_log_save"]


async def test_ambiguous_athletes_are_never_guessed(mcp, repos) -> None:
    server, contexts, _ = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))
    client = Client(server, token)

    await client.call_tool(
        "training_athlete_upsert",
        {"display_name": "Саша Иванов", "operation_id": "ta-1"},
    )
    await client.call_tool(
        "training_athlete_upsert",
        {"display_name": "Саша Петров", "operation_id": "ta-2"},
    )
    payload, _ = await client.call_tool(
        "training_log_save",
        {
            "athlete_name": "Саша",
            "local_date": "2026-08-30",
            "exercises": [{"name": "Присед", "sets": [{"reps": 5, "weight_kg": 100}]}],
            "operation_id": "ta-log",
        },
    )

    assert payload["error"] == "ambiguous"
    assert len(payload["athletes"]) == 2
    assert await repos.training.list_logs("tg:1") == []


async def test_self_mode_logs_without_a_name_to_the_user(mcp, repos) -> None:
    server, contexts, _ = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))
    client = Client(server, token)

    payload, is_error = await client.call_tool(
        "training_log_save",
        {
            "local_date": "2026-08-30",
            "exercises": [{"name": "Присед", "sets": [{"reps": 5, "weight_kg": 80}]}],
            "operation_id": "self-log",
        },
    )

    assert not is_error and payload["created"] is True
    assert payload["mode"] == "self"
    assert payload["athlete_name"] == "я"
    athletes = await repos.training.list_athletes("tg:1")
    assert len(athletes) == 1 and athletes[0]["is_self"] is True


async def test_trainer_mode_refuses_a_log_without_an_athlete(mcp, repos) -> None:
    server, contexts, _ = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))
    client = Client(server, token)

    await client.call_tool(
        "training_profile_set", {"mode": "trainer", "operation_id": "mode-t"}
    )
    payload, is_error = await client.call_tool(
        "training_log_save",
        {
            "local_date": "2026-08-30",
            "exercises": [{"name": "Жим", "sets": [{"reps": 5, "weight_kg": 60}]}],
            "operation_id": "trainer-anon",
        },
    )

    assert not is_error
    assert payload["error"] == "athlete_required"
    assert payload["mode"] == "trainer"
    assert await repos.training.list_logs("tg:1") == []


async def test_trainer_mode_keeps_a_separate_schedule_per_athlete(mcp, repos) -> None:
    server, contexts, _ = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))
    client = Client(server, token)

    await client.call_tool("training_profile_set", {"mode": "trainer", "operation_id": "m"})
    vasya, _ = await client.call_tool(
        "training_schedule_upsert",
        {
            "athlete_name": "Вася",
            "local_date": "2026-09-01",
            "title": "Ноги",
            "operation_id": "s-v",
        },
    )
    masha, _ = await client.call_tool(
        "training_schedule_upsert",
        {
            "athlete_name": "Маша",
            "local_date": "2026-09-01",
            "title": "Жим",
            "operation_id": "s-m",
        },
    )

    assert vasya["mode"] == "trainer" and masha["mode"] == "trainer"
    assert vasya["athlete_id"] != masha["athlete_id"]
    listed, _ = await client.call_tool(
        "training_schedule_list", {"date_from": "2026-09-01", "date_to": "2026-09-01"}
    )
    assert listed["count"] == 2
    names = {item["athlete_name"] for item in listed["sessions"]}
    assert names == {"Вася", "Маша"}


async def test_training_export_returns_csv_without_sending(mcp, repos) -> None:
    server, contexts, _ = mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(Provenance.DIRECT_COMMAND))
    client = Client(server, token)

    await client.call_tool(
        "training_log_save",
        {
            "athlete_name": "Вася",
            "local_date": "2026-08-30",
            "exercises": [{"name": "Жим", "sets": [{"reps": 5, "weight_kg": 70}]}],
            "operation_id": "exp-1",
        },
    )
    payload, is_error = await client.call_tool(
        "training_export",
        {"kind": "logs", "send": False, "operation_id": "exp-csv"},
    )

    assert not is_error
    assert payload["sent"] == []
    assert "Жим" in payload["files"][0]["csv"]
    assert "70" in payload["files"][0]["csv"]


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
