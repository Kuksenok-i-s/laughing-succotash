"""The calendar contract the MCP tools are written against.

The tool schemas must not depend on where events actually live. Today the only implementation is
SQLite; swapping in EventKit or CalDAV on the Mac mini should require no change to the tools, the
permission tiers or the confirmation prompts.
"""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo
from typing import Any, Protocol


class CalendarProvider(Protocol):
    async def list_events(
        self, user_id: str, start: datetime, end: datetime, limit: int = 100
    ) -> list[dict[str, Any]]: ...

    async def get_event(self, user_id: str, event_id: str) -> dict[str, Any] | None: ...

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
        """Create an event. The boolean reports that ``operation_id`` had already been applied."""
        ...

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
    ) -> dict[str, Any] | None: ...

    async def delete_event(self, user_id: str, event_id: str) -> bool: ...

    async def find_free_slots(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
        *,
        duration: timedelta,
        user_tz: tzinfo,
        workday: tuple[int, int] = (9, 21),
    ) -> list[dict[str, Any]]: ...
