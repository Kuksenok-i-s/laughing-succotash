"""Buttoned reminder follow-up: done, snooze, propose, and survival across a restart."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent_core.assistant.confirmations import ConfirmationService
from agent_core.reminders import FollowupService

NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def followup(repos, gateway, settings):
    confirmations = ConfirmationService(repos.pending_actions, gateway, timeout_seconds=60)
    service = FollowupService(
        repos, confirmations, gateway, default_timezone=settings.default_timezone,
    )
    confirmations.register_handler(FollowupService.TOOL, service.handle)
    return service, confirmations


async def _reminder(repos, *, text="выключить духовку", rrule=None):
    reminder, _ = await repos.reminders.create(
        user_id="tg:1",
        text=text,
        due_at=NOW - timedelta(minutes=1),
        timezone_name="Europe/Moscow",
        rrule=rrule,
        operation_id="op-followup",
    )
    return reminder


def _press_target(gateway) -> str:
    return gateway.confirms()[-1]["action_id"]


async def test_a_fired_reminder_asks_whether_it_was_done(followup, repos, gateway) -> None:
    service, _confirmations = followup
    reminder = await _reminder(repos)

    await service.offer(reminder, 500)

    prompt = gateway.confirms()[0]
    assert "выключить духовку" in prompt["text"]
    assert [action["id"] for action in prompt["actions"]] == ["done", "not_done", "drop"]


async def test_done_completes_a_one_shot(followup, repos, gateway) -> None:
    service, confirmations = followup
    reminder = await _reminder(repos)
    await service.offer(reminder, 500)
    await repos.reminders.mark_fired(reminder.reminder_id, None)

    await confirmations.resolve(_press_target(gateway), "tg:1", "done")

    stored = await repos.reminders.get(reminder.reminder_id, "tg:1")
    assert stored is not None
    assert stored.status == "completed"
    assert any("Готово" in text for text in gateway.texts())


async def test_not_done_then_snooze_puts_it_back_on_the_clock(followup, repos, gateway) -> None:
    service, confirmations = followup
    reminder = await _reminder(repos)
    await service.offer(reminder, 500)
    await repos.reminders.mark_fired(reminder.reminder_id, None)

    await confirmations.resolve(_press_target(gateway), "tg:1", "not_done")
    missed = gateway.confirms()[-1]
    assert [action["id"] for action in missed["actions"]] == ["snooze", "reschedule", "drop"]

    await confirmations.resolve(missed["action_id"], "tg:1", "snooze")

    stored = await repos.reminders.get(reminder.reminder_id, "tg:1")
    assert stored is not None
    assert stored.status == "scheduled"
    assert stored.due_at is not None
    assert stored.due_at > datetime.now(timezone.utc)


async def test_reschedule_proposes_a_time_and_accept_applies_it(
    followup, repos, gateway
) -> None:
    service, confirmations = followup
    reminder = await _reminder(repos, text="выпить лекарство")
    await service.offer(reminder, 500)
    await repos.reminders.mark_fired(reminder.reminder_id, None)

    await confirmations.resolve(_press_target(gateway), "tg:1", "not_done")
    await confirmations.resolve(_press_target(gateway), "tg:1", "reschedule")

    proposal = gateway.confirms()[-1]
    assert [action["id"] for action in proposal["actions"]] == ["accept", "reject", "drop"]
    assert "высокая" in proposal["text"]

    await confirmations.resolve(proposal["action_id"], "tg:1", "accept")

    stored = await repos.reminders.get(reminder.reminder_id, "tg:1")
    assert stored is not None
    assert stored.status == "scheduled"
    assert stored.due_at is not None
    assert any("Перенёс" in text for text in gateway.texts())


async def test_refusing_a_proposal_returns_to_snooze_or_reschedule(
    followup, repos, gateway
) -> None:
    service, confirmations = followup
    reminder = await _reminder(repos)
    await service.offer(reminder, 500)
    await repos.reminders.mark_fired(reminder.reminder_id, None)

    await confirmations.resolve(_press_target(gateway), "tg:1", "not_done")
    await confirmations.resolve(_press_target(gateway), "tg:1", "reschedule")
    await confirmations.resolve(_press_target(gateway), "tg:1", "reject")

    again = gateway.confirms()[-1]
    assert [action["id"] for action in again["actions"]] == ["snooze", "reschedule", "drop"]


async def test_a_press_after_restart_still_completes(repos, gateway, settings) -> None:
    """Follow-up must not depend on an in-memory waiter."""
    first = ConfirmationService(repos.pending_actions, gateway, timeout_seconds=60)
    offering = FollowupService(
        repos, first, gateway, default_timezone=settings.default_timezone,
    )
    reminder = await _reminder(repos)
    await offering.offer(reminder, 500)
    await repos.reminders.mark_fired(reminder.reminder_id, None)
    action_id = gateway.confirms()[-1]["action_id"]

    second = ConfirmationService(repos.pending_actions, gateway, timeout_seconds=60)
    surviving = FollowupService(
        repos, second, gateway, default_timezone=settings.default_timezone,
    )
    second.register_handler(FollowupService.TOOL, surviving.handle)

    assert await second.resolve(action_id, "tg:1", "done") == "applied"
    stored = await repos.reminders.get(reminder.reminder_id, "tg:1")
    assert stored is not None
    assert stored.status == "completed"


async def test_snoozing_a_series_creates_an_extra_one_shot(
    followup, repos, gateway
) -> None:
    service, confirmations = followup
    reminder = await _reminder(
        repos, text="проверить отчёты",
        rrule="FREQ=WEEKLY;BYDAY=MO;BYHOUR=10;BYMINUTE=0",
    )
    next_due = NOW + timedelta(days=7)
    await service.offer(reminder, 500)
    await repos.reminders.mark_fired(reminder.reminder_id, next_due)

    await confirmations.resolve(_press_target(gateway), "tg:1", "not_done")
    await confirmations.resolve(_press_target(gateway), "tg:1", "snooze")

    original = await repos.reminders.get(reminder.reminder_id, "tg:1")
    assert original is not None
    assert original.status == "scheduled"
    assert original.rrule
    extras = [
        item for item in await repos.reminders.list("tg:1", status="scheduled")
        if item.reminder_id != reminder.reminder_id
    ]
    assert len(extras) == 1
    assert extras[0].rrule is None


async def test_offering_the_same_occurrence_twice_sends_one_prompt(
    followup, repos, gateway
) -> None:
    service, _confirmations = followup
    reminder = await _reminder(repos)
    await service.offer(reminder, 500)
    await service.offer(reminder, 500)
    assert len(gateway.confirms()) == 1


async def test_done_on_a_series_does_not_cancel_it(followup, repos, gateway) -> None:
    service, confirmations = followup
    reminder = await _reminder(
        repos, rrule="FREQ=DAILY;BYHOUR=10;BYMINUTE=0",
    )
    await service.offer(reminder, 500)
    await repos.reminders.mark_fired(reminder.reminder_id, NOW + timedelta(days=1))

    await confirmations.resolve(_press_target(gateway), "tg:1", "done")

    stored = await repos.reminders.get(reminder.reminder_id, "tg:1")
    assert stored is not None
    assert stored.status == "scheduled"
    assert stored.rrule


async def test_drop_cancels_a_fired_one_shot(followup, repos, gateway) -> None:
    service, confirmations = followup
    reminder = await _reminder(repos)
    await service.offer(reminder, 500)
    await repos.reminders.mark_fired(reminder.reminder_id, None)

    await confirmations.resolve(_press_target(gateway), "tg:1", "drop")

    stored = await repos.reminders.get(reminder.reminder_id, "tg:1")
    assert stored is not None
    assert stored.status == "cancelled"
    assert any("Отменил" in text for text in gateway.texts())


async def test_drop_stops_a_series(followup, repos, gateway) -> None:
    service, confirmations = followup
    reminder = await _reminder(
        repos, rrule="FREQ=DAILY;BYHOUR=10;BYMINUTE=0",
    )
    await service.offer(reminder, 500)
    await repos.reminders.mark_fired(reminder.reminder_id, NOW + timedelta(days=1))

    await confirmations.resolve(_press_target(gateway), "tg:1", "not_done")
    await confirmations.resolve(_press_target(gateway), "tg:1", "drop")

    stored = await repos.reminders.get(reminder.reminder_id, "tg:1")
    assert stored is not None
    assert stored.status == "cancelled"
