"""Route one Telegram user to an external calendar without changing other users' storage."""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo
from typing import Any

from .base import CalendarProvider


class RoutedCalendarProvider:
    def __init__(
        self,
        *,
        routed_user_id: str,
        routed: CalendarProvider,
        fallback: CalendarProvider,
    ) -> None:
        self._routed_user_id = routed_user_id
        self._routed = routed
        self._fallback = fallback

    def _provider(self, user_id: str) -> CalendarProvider:
        return self._routed if user_id == self._routed_user_id else self._fallback

    async def list_events(
        self, user_id: str, start: datetime, end: datetime, limit: int = 100
    ) -> list[dict[str, Any]]:
        return await self._provider(user_id).list_events(user_id, start, end, limit)

    async def get_event(self, user_id: str, event_id: str) -> dict[str, Any] | None:
        return await self._provider(user_id).get_event(user_id, event_id)

    async def create_event(self, **kwargs) -> tuple[dict[str, Any], bool]:
        return await self._provider(kwargs["user_id"]).create_event(**kwargs)

    async def update_event(
        self, user_id: str, event_id: str, **kwargs
    ) -> dict[str, Any] | None:
        return await self._provider(user_id).update_event(user_id, event_id, **kwargs)

    async def delete_event(self, user_id: str, event_id: str) -> bool:
        return await self._provider(user_id).delete_event(user_id, event_id)

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
        return await self._provider(user_id).find_free_slots(
            user_id,
            start,
            end,
            duration=duration,
            user_tz=user_tz,
            workday=workday,
        )
