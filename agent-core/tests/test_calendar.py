"""Local calendar provider, in particular free-slot search in the user's own timezone."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from agent_core.calendar.local import LocalCalendarProvider

MOSCOW = ZoneInfo("Europe/Moscow")


@pytest.fixture
def calendar(repos) -> LocalCalendarProvider:
    return LocalCalendarProvider(repos.calendar)


def local(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=MOSCOW)


async def book(calendar, day: int, start: int, end: int, title: str = "занято") -> dict:
    event, _ = await calendar.create_event(
        user_id="tg:1",
        title=title,
        starts_at=local(day, start),
        ends_at=local(day, end),
        timezone_name="Europe/Moscow",
        operation_id=f"{title}-{day}-{start}",
    )
    return event


async def test_listing_includes_events_that_straddle_the_window(calendar) -> None:
    """A meeting that began before the window still counts as 'what I have this afternoon'."""
    await book(calendar, 1, 13, 16, "длинная встреча")

    events = await calendar.list_events("tg:1", local(1, 14), local(1, 15))
    assert [event["title"] for event in events] == ["длинная встреча"]


async def test_free_slots_respect_working_hours(calendar) -> None:
    slots = await calendar.find_free_slots(
        "tg:1", local(1, 0), local(1, 23, 59),
        duration=timedelta(hours=1), user_tz=MOSCOW, workday=(9, 18),
    )

    assert len(slots) == 1
    assert slots[0]["start"].startswith("2026-09-01T09:00")
    assert slots[0]["end"].startswith("2026-09-01T18:00")


async def test_free_slots_are_split_around_meetings(calendar) -> None:
    await book(calendar, 1, 11, 12)
    await book(calendar, 1, 15, 16)

    slots = await calendar.find_free_slots(
        "tg:1", local(1, 9), local(1, 18),
        duration=timedelta(hours=1), user_tz=MOSCOW, workday=(9, 18),
    )

    windows = [(slot["start"][11:16], slot["end"][11:16]) for slot in slots]
    assert windows == [("09:00", "11:00"), ("12:00", "15:00"), ("16:00", "18:00")]


async def test_gaps_shorter_than_the_requested_duration_are_not_offered(calendar) -> None:
    await book(calendar, 1, 10, 11)
    await book(calendar, 1, 11, 18, "весь остаток дня")  # leaves only 09:00-10:00 free

    slots = await calendar.find_free_slots(
        "tg:1", local(1, 9), local(1, 18),
        duration=timedelta(hours=2), user_tz=MOSCOW, workday=(9, 18),
    )
    assert slots == []


async def test_overlapping_meetings_do_not_produce_phantom_gaps(calendar) -> None:
    await book(calendar, 1, 10, 13, "первая")
    await book(calendar, 1, 11, 12, "вторая внутри первой")

    slots = await calendar.find_free_slots(
        "tg:1", local(1, 9), local(1, 18),
        duration=timedelta(minutes=30), user_tz=MOSCOW, workday=(9, 18),
    )
    windows = [(slot["start"][11:16], slot["end"][11:16]) for slot in slots]
    assert windows == [("09:00", "10:00"), ("13:00", "18:00")]


async def test_a_multi_day_search_returns_one_window_per_day(calendar) -> None:
    slots = await calendar.find_free_slots(
        "tg:1", local(1, 9), local(3, 18),
        duration=timedelta(hours=1), user_tz=MOSCOW, workday=(9, 18),
    )
    assert len(slots) == 3


async def test_a_deleted_event_frees_its_slot(calendar) -> None:
    event = await book(calendar, 1, 10, 11)
    assert await calendar.delete_event("tg:1", event["event_id"]) is True

    slots = await calendar.find_free_slots(
        "tg:1", local(1, 9), local(1, 18),
        duration=timedelta(hours=8), user_tz=MOSCOW, workday=(9, 18),
    )
    assert len(slots) == 1


async def test_moving_an_event_updates_it_in_place(calendar) -> None:
    """'Перенеси встречу на час позже' is an update, not a delete plus a create."""
    event = await book(calendar, 1, 10, 11, "созвон")

    moved = await calendar.update_event(
        "tg:1", event["event_id"], starts_at=local(1, 11), ends_at=local(1, 12)
    )

    assert moved["event_id"] == event["event_id"]
    assert moved["starts_at"].startswith("2026-09-01T08:00")  # 11:00 Moscow in UTC


async def test_another_users_event_is_invisible(calendar) -> None:
    event = await book(calendar, 1, 10, 11)

    assert await calendar.get_event("tg:2", event["event_id"]) is None
    assert await calendar.delete_event("tg:2", event["event_id"]) is False
