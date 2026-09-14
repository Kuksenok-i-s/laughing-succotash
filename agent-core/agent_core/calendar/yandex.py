"""Yandex Calendar provider over its supported CalDAV endpoint."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Any, Callable
from uuid import NAMESPACE_URL, uuid5

from caldav import error as caldav_error
from caldav.aio import get_async_davclient
from icalendar import Calendar, Event

from .local import _merge, _parse, _slot, _workday_windows


class YandexCalendarProvider:
    """A CalDAV calendar available to exactly one namespaced Telegram user."""

    def __init__(
        self,
        *,
        user_id: str,
        username: str,
        app_password: str,
        calendar_name: str | None = None,
        url: str = "https://caldav.yandex.ru/",
        client_factory: Callable[..., Any] = get_async_davclient,
    ) -> None:
        self._user_id = user_id
        self._username = username
        self._app_password = app_password
        self._calendar_name = calendar_name
        self._url = url
        self._client_factory = client_factory

    def _check_user(self, user_id: str) -> None:
        if user_id != self._user_id:
            raise PermissionError("Yandex Calendar is not configured for this Telegram user")

    async def _client(self):
        return await self._client_factory(
            url=self._url,
            username=self._username,
            password=self._app_password,
        )

    @staticmethod
    async def _calendars(client) -> list[Any]:
        principal = await client.get_principal()
        calendars = list(await principal.get_calendars())
        if not calendars:
            raise RuntimeError("Yandex account has no CalDAV calendars")
        return calendars

    async def _write_calendar(self, client):
        calendars = await self._calendars(client)
        if self._calendar_name is None:
            return calendars[0]
        for calendar in calendars:
            if await self._calendar_name_of(calendar) == self._calendar_name:
                return calendar
        raise RuntimeError(f"Yandex calendar {self._calendar_name!r} was not found")

    @staticmethod
    async def _calendar_name_of(calendar) -> str:
        return str(await calendar.get_display_name() or "default")

    async def list_events(
        self, user_id: str, start: datetime, end: datetime, limit: int = 100
    ) -> list[dict[str, Any]]:
        self._check_user(user_id)
        result: list[dict[str, Any]] = []
        async with await self._client() as client:
            for calendar in await self._calendars(client):
                calendar_name = await self._calendar_name_of(calendar)
                resources = await calendar.search(start=start, end=end, event=True, expand=True)
                for resource in resources:
                    item = self._public(resource, calendar_name)
                    if item is not None:
                        result.append(item)
        result.sort(key=lambda event: event["starts_at"])
        return result[:limit]

    async def get_event(self, user_id: str, event_id: str) -> dict[str, Any] | None:
        self._check_user(user_id)
        async with await self._client() as client:
            resource, calendar = await self._find_resource(client, event_id)
            if resource is None:
                return None
            return self._public(resource, await self._calendar_name_of(calendar))

    async def _find_resource(self, client, event_id: str):
        uid = event_id.split("#", 1)[0]
        for calendar in await self._calendars(client):
            try:
                return await calendar.get_event_by_uid(uid), calendar
            except caldav_error.NotFoundError:
                continue
        return None, None

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
        self._check_user(user_id)
        return await self._create_event(
            title,
            starts_at,
            ends_at,
            timezone_name,
            location,
            description,
            attendees or [],
            operation_id,
        )

    async def _create_event(
        self,
        title: str,
        starts_at: datetime,
        ends_at: datetime,
        timezone_name: str,
        location: str | None,
        description: str | None,
        attendees: list[str],
        operation_id: str,
    ) -> tuple[dict[str, Any], bool]:
        uid = f"{uuid5(NAMESPACE_URL, operation_id)}@telegram-assistant"
        document = Calendar()
        document.add("prodid", "-//Telegram Personal Assistant//EN")
        document.add("version", "2.0")
        event = Event()
        event.add("uid", uid)
        event.add("dtstamp", datetime.now(timezone.utc))
        event.add("summary", title)
        event.add("dtstart", starts_at)
        event.add("dtend", ends_at)
        event.add("x-pa-timezone", timezone_name)
        if location:
            event.add("location", location)
        if description:
            event.add("description", description)
        for attendee in attendees:
            address = attendee if attendee.lower().startswith("mailto:") else f"mailto:{attendee}"
            event.add("attendee", address)
        document.add_component(event)

        async with await self._client() as client:
            calendar = await self._write_calendar(client)
            try:
                existing = await calendar.get_event_by_uid(uid)
            except caldav_error.NotFoundError:
                existing = None
            if existing is not None:
                public = self._public(existing, await self._calendar_name_of(calendar))
                if public is None:
                    raise RuntimeError("Yandex returned an invalid existing calendar event")
                return public, True
            resource = await calendar.add_event(document.to_ical().decode("utf-8"))
            public = self._public(resource, await self._calendar_name_of(calendar))
            if public is None:
                raise RuntimeError("Yandex returned an invalid created calendar event")
            return public, False

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
        self._check_user(user_id)
        return await self._update_event(
            event_id,
            title,
            starts_at,
            ends_at,
            location,
            description,
        )

    async def _update_event(self, event_id, title, starts_at, ends_at, location, description):
        async with await self._client() as client:
            resource, calendar = await self._find_resource(client, event_id)
            if resource is None:
                return None
            with resource.edit_icalendar_component() as component:
                for key, value in (
                    ("SUMMARY", title),
                    ("DTSTART", starts_at),
                    ("DTEND", ends_at),
                    ("LOCATION", location),
                    ("DESCRIPTION", description),
                ):
                    if value is not None:
                        if key in component:
                            del component[key]
                        component.add(key, value)
            await resource.save(all_recurrences=True)
            return self._public(resource, await self._calendar_name_of(calendar))

    async def delete_event(self, user_id: str, event_id: str) -> bool:
        self._check_user(user_id)
        async with await self._client() as client:
            resource, _calendar = await self._find_resource(client, event_id)
            if resource is None:
                return False
            await resource.delete()
            return True

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
        events = await self.list_events(user_id, start, end, limit=500)
        busy = _merge([(_parse(item["starts_at"]), _parse(item["ends_at"])) for item in events])
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

    @staticmethod
    def _public(resource, calendar_name: str) -> dict[str, Any] | None:
        component = resource.get_icalendar_component()
        if component is None or component.name != "VEVENT":
            return None
        starts_at = _decoded_datetime(component, "DTSTART")
        ends_at = _decoded_datetime(component, "DTEND")
        if starts_at is None:
            return None
        if ends_at is None:
            ends_at = starts_at + timedelta(hours=1)
        uid = str(component.get("UID", ""))
        recurrence_id = _decoded_datetime(component, "RECURRENCE-ID")
        event_id = uid if recurrence_id is None else f"{uid}#{recurrence_id.isoformat()}"
        attendees = component.get("ATTENDEE", [])
        if not isinstance(attendees, list):
            attendees = [attendees]
        return {
            "event_id": event_id,
            "calendar": calendar_name,
            "title": str(component.get("SUMMARY", "(без названия)")),
            "starts_at": starts_at.astimezone(timezone.utc).isoformat(),
            "ends_at": ends_at.astimezone(timezone.utc).isoformat(),
            "timezone": str(component.get("X-PA-TIMEZONE", getattr(starts_at.tzinfo, "key", "UTC"))),
            "location": _optional_text(component, "LOCATION"),
            "description": _optional_text(component, "DESCRIPTION"),
            "attendees": [str(value).removeprefix("mailto:") for value in attendees],
            "status": "confirmed",
        }


def _decoded_datetime(component, key: str) -> datetime | None:
    value = component.get(key)
    if value is None:
        return None
    decoded = value.dt if hasattr(value, "dt") else component.decoded(key)
    if isinstance(decoded, datetime):
        return decoded if decoded.tzinfo else decoded.replace(tzinfo=timezone.utc)
    if isinstance(decoded, date):
        return datetime.combine(decoded, time.min, tzinfo=timezone.utc)
    return None


def _optional_text(component, key: str) -> str | None:
    value = component.get(key)
    return str(value) if value is not None else None
