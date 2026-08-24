"""The confirmation flow: the Core asks, the Gateway renders, only the user decides."""

from __future__ import annotations

import asyncio

import pytest
from pa_protocol import methods

from agent_core.assistant.confirmations import ConfirmationService


@pytest.fixture
def service(repos, gateway) -> ConfirmationService:
    return ConfirmationService(repos.pending_actions, gateway, timeout_seconds=2)


async def _ask(service: ConfirmationService, gateway) -> tuple[asyncio.Task[bool], str]:
    """Start a confirmation request and wait until the prompt has reached the Gateway."""
    task = asyncio.ensure_future(
        service.request(
            user_id="tg:1",
            chat_id=500,
            tool_name="calendar_delete",
            arguments={"event_id": "E1"},
            operation_id="op-1",
            tier="dangerous",
            prompt_text="Удалить встречу?",
        )
    )
    for _ in range(200):
        confirms = [p for m, p in gateway.delivered if m == methods.TELEGRAM_CONFIRM]
        if confirms:
            return task, confirms[0]["action_id"]
        await asyncio.sleep(0.01)
    raise AssertionError("the confirmation prompt never reached the gateway")


async def test_the_gateway_is_asked_to_render_buttons(service, gateway, repos) -> None:
    task, action_id = await _ask(service, gateway)

    confirms = [p for m, p in gateway.delivered if m == methods.TELEGRAM_CONFIRM]
    assert len(confirms) == 1
    assert confirms[0]["text"] == "Удалить встречу?"
    assert [action["id"] for action in confirms[0]["actions"]] == ["approve", "reject"]

    pending = await repos.pending_actions.pending_for_user("tg:1")
    # The exact validated arguments are stored, so approval executes what was shown.
    assert pending[0].arguments == {"event_id": "E1"}

    await service.resolve(action_id, "tg:1", "approve")
    assert await task is True


async def test_a_rejection_returns_false(service, gateway) -> None:
    task, action_id = await _ask(service, gateway)

    assert await service.resolve(action_id, "tg:1", "reject") == "applied"
    assert await task is False


async def test_pressing_the_button_twice_changes_nothing(service, gateway) -> None:
    task, action_id = await _ask(service, gateway)

    assert await service.resolve(action_id, "tg:1", "approve") == "applied"
    assert await service.resolve(action_id, "tg:1", "reject") == "already_resolved"
    assert await task is True


async def test_another_user_cannot_resolve_it(service, gateway) -> None:
    task, action_id = await _ask(service, gateway)

    assert await service.resolve(action_id, "tg:2", "approve") == "unknown"
    assert not task.done()

    await service.resolve(action_id, "tg:1", "reject")
    assert await task is False


async def test_an_unanswered_prompt_expires_as_a_refusal(repos, gateway) -> None:
    """Silence must never be taken as consent."""
    service = ConfirmationService(repos.pending_actions, gateway, timeout_seconds=0.05)
    result = await service.request(
        user_id="tg:1", chat_id=500, tool_name="task_delete", arguments={},
        operation_id="op", tier="dangerous", prompt_text="?",
    )

    assert result is False
    pending = await repos.pending_actions.pending_for_user("tg:1")
    assert pending == []


async def test_shutdown_refuses_everything_outstanding(service, gateway) -> None:
    task, _action_id = await _ask(service, gateway)
    service.abandon_all()
    assert await task is False


async def test_a_resolution_after_a_restart_is_reported_not_lost(repos, gateway) -> None:
    """Nobody is waiting any more, but the user's decision still has to be acknowledged."""
    service = ConfirmationService(repos.pending_actions, gateway, timeout_seconds=60)
    action = await repos.pending_actions.create(
        user_id="tg:1", tool_name="note_delete", arguments={}, operation_id="op",
        tier="dangerous", prompt_text="?", ttl_seconds=60,
    )

    assert await service.resolve(action.action_id, "tg:1", "approve") == "applied"


async def test_unknown_actions_are_reported_as_unknown(service) -> None:
    assert await service.resolve("does-not-exist", "tg:1", "approve") == "unknown"
