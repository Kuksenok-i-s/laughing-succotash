"""Local SQLite calendar.

Free-slot search is done here rather than in the tool layer because a real backend can usually
answer it far more cheaply than by listing every event, and the tool should not care which.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any

from dateutil import parser as date_parser

from ..storage.repositories import CalendarRepository


class LocalCalendarProvider:
    def __init__(self, repository: CalendarRepository) -> None:
        self._repo = repository

    async def list_events(
        self, user_id: str, start: datetime, end: datetime, limit: int = 100
    ) -> list[dict[str, Any]]:
        return await self._repo.list_range(user_id, start, end, limit)

    async def get_event(self, user_id: str, event_id: str) -> dict[str, Any] | None:
        return await self._repo.get(event_id, user_id)

    async def create_event(
        self,
        *,
        user_id: str,
        title: str,
        starts_at: datetime,
        ends_at: datetime,
        timezone_name: str,
        location: str | None = None,
        description: str | None = None,
        attendees: list[str] | None = None,
        operation_id: str,
    ) -> tuple[dict[str, Any], bool]:
        return await self._repo.create(
            user_id=user_id,
            title=title,
            starts_at=starts_at,
            ends_at=ends_at,
            timezone_name=timezone_name,
            location=location,
            description=description,
            attendees=attendees,
            operation_id=operation_id,
        )

    async def update_event(
        self,
        user_id: str,
        event_id: str,
        *,
        title: str | None = None,
        starts_at: datetime | None = None,
        ends_at: datetime | None = None,
        location: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any] | None:
        return await self._repo.update(
            event_id,
            user_id,
            title=title,
            starts_at=starts_at,
            ends_at=ends_at,
            location=location,
            description=description,
        )

    async def delete_event(self, user_id: str, event_id: str) -> bool:
        return await self._repo.delete(event_id, user_id)

    async def find_free_slots(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
        *,
        duration: timedelta,
        user_tz: tzinfo,
        workday: tuple[int, int] = (9, 21),
    ) -> list[dict[str, Any]]:
        if end <= start or duration <= timedelta(0):
            return []

        events = await self._repo.list_range(user_id, start, end, limit=500)
        busy = _merge(
            [
                (_parse(event["starts_at"]), _parse(event["ends_at"]))
                for event in events
                if event.get("starts_at") and event.get("ends_at")
            ]
        )

        slots: list[dict[str, Any]] = []
        for window_start, window_end in _workday_windows(start, end, user_tz, workday):
            cursor = window_start
            for busy_start, busy_end in busy:
                if busy_end <= cursor or busy_start >= window_end:
                    continue
                if busy_start - cursor >= duration:
                    slots.append(_slot(cursor, busy_start, user_tz))
                cursor = max(cursor, busy_end)
            if window_end - cursor >= duration:
                slots.append(_slot(cursor, window_end, user_tz))

        return slots


def _slot(start: datetime, end: datetime, user_tz: tzinfo) -> dict[str, Any]:
    local_start = start.astimezone(user_tz)
    local_end = end.astimezone(user_tz)
    return {
        "start": local_start.isoformat(),
        "end": local_end.isoformat(),
        "minutes": int((end - start).total_seconds() // 60),
    }


def _parse(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = date_parser.isoparse(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _merge(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    """Collapse overlapping busy periods so double-booked time is not counted twice."""
    merged: list[tuple[datetime, datetime]] = []
    for begin, finish in sorted(intervals):
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], finish))
        else:
            merged.append((begin, finish))
    return merged


def _workday_windows(
    start: datetime, end: datetime, user_tz: tzinfo, workday: tuple[int, int]
) -> list[tuple[datetime, datetime]]:
    """Split a range into per-day working windows in the user's own timezone.

    "Найди свободный час завтра после 14" must not offer 03:00, and the day boundaries have to be
    local ones — a UTC day would be wrong for every user east or west of it.
    """
    day_start_hour, day_end_hour = workday
    windows: list[tuple[datetime, datetime]] = []

    local_day = start.astimezone(user_tz).replace(hour=0, minute=0, second=0, microsecond=0)
    local_end = end.astimezone(user_tz)

    while local_day <= local_end:
        opens = local_day.replace(hour=day_start_hour)
        closes = (
            local_day + timedelta(days=1)
            if day_end_hour >= 24
            else local_day.replace(hour=day_end_hour)
        )
        window_start = max(opens.astimezone(timezone.utc), start)
        window_end = min(closes.astimezone(timezone.utc), end)
        if window_end > window_start:
            windows.append((window_start, window_end))
        local_day += timedelta(days=1)

    return windows
