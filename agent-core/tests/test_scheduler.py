"""Scheduler behaviour, including the cases that only show up when something is broken."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from agent_core.assistant.service import AssistantService
from agent_core.scheduler.service import Scheduler

MOSCOW = ZoneInfo("Europe/Moscow")


@pytest.fixture
def notifier(settings, repos, gateway, backend):
    """A real AssistantService: only its notification path is exercised here."""
    return AssistantService(settings, repos, gateway, None, None, backend)


@pytest.fixture
def scheduler(settings, repos, notifier):
    return Scheduler(repos, notifier, default_timezone=settings.default_timezone)


async def _user(repos, chat_id: int = 500) -> None:
    await repos.conversations.ensure_user("tg:1")
    await repos.conversations.remember_chat("tg:1", chat_id)


async def test_one_shot_reminder_fires_exactly_once(repos, gateway, scheduler) -> None:
    await _user(repos)
    await repos.reminders.create(
        user_id="tg:1",
        text="Выключить духовку",
        due_at=_ago(minutes=1),
        timezone_name="Europe/Moscow",
        operation_id="op-1",
    )

    await scheduler.tick()
    await scheduler.tick()

    assert gateway.texts() == ["⏰ Выключить духовку"]
    assert await repos.reminders.list("tg:1", status="scheduled") == []
    assert (await repos.reminders.list("tg:1", status="fired"))[0].fire_count == 1


async def test_recurring_reminder_is_rescheduled(repos, gateway, scheduler) -> None:
    await _user(repos)
    await repos.reminders.create(
        user_id="tg:1",
        text="Проверить отчёты",
        due_at=_ago(minutes=1),
        timezone_name="Europe/Moscow",
        rrule="FREQ=WEEKLY;BYDAY=MO;BYHOUR=10;BYMINUTE=0",
        operation_id="op-2",
    )

    await scheduler.tick()

    still_scheduled = await repos.reminders.list("tg:1", status="scheduled")
    assert len(still_scheduled) == 1
    assert still_scheduled[0].fire_count == 1
    assert still_scheduled[0].due_at > datetime.now(timezone.utc)
    assert len(gateway.texts()) == 1

    # The next tick must not fire it again: its new due time is in the future.
    await scheduler.tick()
    assert len(gateway.texts()) == 1


async def test_reminder_fired_while_gateway_is_down_arrives_once(
    repos, gateway, scheduler
) -> None:
    await _user(repos)
    gateway.online = False
    await repos.reminders.create(
        user_id="tg:1", text="Позвонить Ивану", due_at=_ago(minutes=1),
        timezone_name="Europe/Moscow", operation_id="op-3",
    )

    await scheduler.tick()
    assert gateway.delivered == []
    assert await repos.events.pending_count() == 1

    # Gateway comes back.
    gateway.online = True
    assert await gateway.drain() == 1
    assert gateway.texts() == ["⏰ Позвонить Ивану"]

    # A second drain (a reconnect replay) must not resend it.
    assert await gateway.drain() == 0
    assert len(gateway.texts()) == 1


async def test_a_replayed_fire_is_deduplicated_by_delivery_id(repos, gateway, scheduler) -> None:
    """Models a crash between queueing the notification and advancing the reminder."""
    await _user(repos)
    reminder, _ = await repos.reminders.create(
        user_id="tg:1", text="дубль", due_at=_ago(minutes=1),
        timezone_name="Europe/Moscow", operation_id="op-4",
    )

    await scheduler._fire_reminders(datetime.now(timezone.utc))  # noqa: SLF001
    # Reminder was advanced by the first call; force the same occurrence again.
    await repos.reminders.update(reminder.reminder_id, "tg:1", due_at=_ago(minutes=1))
    await repos.reminders._db.execute(  # noqa: SLF001
        "UPDATE reminders SET status = 'scheduled', fire_count = 0 WHERE reminder_id = ?",
        (reminder.reminder_id,),
    )
    await scheduler._fire_reminders(datetime.now(timezone.utc))  # noqa: SLF001

    assert len(gateway.texts()) == 1


async def test_reminder_without_a_known_chat_stays_scheduled(repos, gateway, scheduler) -> None:
    """Nowhere to deliver is not the same as delivered; the reminder must not be lost."""
    await repos.conversations.ensure_user("tg:1")
    await repos.reminders.create(
        user_id="tg:1", text="никуда", due_at=_ago(minutes=1),
        timezone_name="Europe/Moscow", operation_id="op-5",
    )

    await scheduler.tick()

    assert gateway.delivered == []
    assert len(await repos.reminders.list("tg:1", status="scheduled")) == 1


async def test_due_time_respects_the_users_timezone(repos, gateway, scheduler) -> None:
    """18:00 Moscow is 15:00 UTC; a reminder set for tonight must not fire this morning."""
    await _user(repos)
    tonight_local = datetime.now(MOSCOW).replace(hour=23, minute=59, second=0, microsecond=0)
    await repos.reminders.create(
        user_id="tg:1", text="вечером", due_at=tonight_local.astimezone(timezone.utc),
        timezone_name="Europe/Moscow", operation_id="op-6",
    )

    await scheduler.tick(now=tonight_local.astimezone(timezone.utc) - timedelta(hours=2))
    assert gateway.delivered == []

    await scheduler.tick(now=tonight_local.astimezone(timezone.utc) + timedelta(minutes=1))
    assert gateway.texts() == ["⏰ вечером"]


async def test_timers_fire_and_do_not_repeat(repos, gateway, scheduler) -> None:
    await _user(repos)
    await repos.timers.create(
        user_id="tg:1", label=None, duration_seconds=17 * 60,
        fires_at=_ago(seconds=5), operation_id="op-7",
    )

    await scheduler.tick()
    await scheduler.tick()

    assert gateway.texts() == ["⏱ Таймер на 17 мин — время вышло."]
    assert await repos.timers.list("tg:1") == []


async def test_pending_reminders_survive_a_restart(settings, db, repos, gateway) -> None:
    """The state a reboot must not lose."""
    await _user(repos)
    await repos.reminders.create(
        user_id="tg:1", text="после перезагрузки", due_at=_ago(minutes=1),
        timezone_name="Europe/Moscow", operation_id="op-8",
    )

    from agent_core.storage.repositories import Repositories

    reopened = Repositories.build(db, settings.default_timezone)
    assert await reopened.reminders.pending_count() == 1


def _ago(**delta) -> datetime:
    return datetime.now(timezone.utc) - timedelta(**delta)
