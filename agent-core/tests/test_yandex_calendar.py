"""Yandex CalDAV adapter and exact Telegram-user routing."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

from caldav import error as caldav_error
from icalendar import Calendar

from agent_core.calendar.routed import RoutedCalendarProvider
from agent_core.calendar.yandex import YandexCalendarProvider
from agent_core.config import Settings

MOSCOW = ZoneInfo("Europe/Moscow")


class FakeResource:
    def __init__(self, data: str, calendar=None) -> None:
        self.document = Calendar.from_ical(data)
        self.calendar = calendar
        self.saved = 0
        self.deleted = False

    def get_icalendar_component(self):
        return next(component for component in self.document.walk() if component.name == "VEVENT")

    @contextmanager
    def edit_icalendar_component(self):
        yield self.get_icalendar_component()

    async def save(self, **_kwargs):
        # Exercise icalendar serialization too: invalid property replacement fails here.
        self.document.to_ical()
        self.saved += 1

    async def delete(self):
        self.deleted = True


class FakeCalendar:
    def __init__(self, name="Личный") -> None:
        self.name = name
        self.resources: dict[str, FakeResource] = {}

    async def get_display_name(self):
        return self.name

    async def search(self, **_kwargs):
        return list(self.resources.values())

    async def get_event_by_uid(self, uid):
        try:
            return self.resources[uid]
        except KeyError as exc:
            raise caldav_error.NotFoundError() from exc

    async def add_event(self, data):
        resource = FakeResource(data, self)
        uid = str(resource.get_icalendar_component()["UID"])
        self.resources[uid] = resource
        return resource


class FakePrincipal:
    def __init__(self, calendars):
        self._calendars = calendars

    async def get_calendars(self):
        return self._calendars


class FakeClient:
    def __init__(self, calendars):
        self.principal = FakePrincipal(calendars)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get_principal(self):
        return self.principal


def provider(calendar: FakeCalendar, *, user_id="tg:42") -> YandexCalendarProvider:
    async def factory(**kwargs):
        assert kwargs == {
            "url": "https://caldav.yandex.ru/",
            "username": "user@yandex.ru",
            "password": "app-secret",
        }
        return FakeClient([calendar])

    return YandexCalendarProvider(
        user_id=user_id,
        username="user@yandex.ru",
        app_password="app-secret",
        client_factory=factory,
    )


def moment(hour: int) -> datetime:
    return datetime(2026, 9, 5, hour, tzinfo=MOSCOW)


async def test_create_list_update_and_delete_round_trip() -> None:
    remote = FakeCalendar()
    yandex = provider(remote)

    created, duplicate = await yandex.create_event(
        user_id="tg:42",
        title="Созвон",
        starts_at=moment(10),
        ends_at=moment(11),
        timezone_name="Europe/Moscow",
        location="онлайн",
        description="план",
        attendees=["friend@example.com"],
        operation_id="stable-operation",
    )

    assert duplicate is False
    assert created["title"] == "Созвон"
    assert created["calendar"] == "Личный"
    assert created["attendees"] == ["friend@example.com"]

    listed = await yandex.list_events("tg:42", moment(0), moment(23))
    assert [item["event_id"] for item in listed] == [created["event_id"]]

    updated = await yandex.update_event(
        "tg:42", created["event_id"], title="Новый созвон", starts_at=moment(12)
    )
    assert updated["title"] == "Новый созвон"
    assert updated["starts_at"].startswith("2026-09-05T09:00:00+00:00")
    assert remote.resources[created["event_id"]].saved == 1

    assert await yandex.delete_event("tg:42", created["event_id"]) is True
    assert remote.resources[created["event_id"]].deleted is True


async def test_operation_id_makes_create_idempotent_at_provider_boundary() -> None:
    remote = FakeCalendar()
    yandex = provider(remote)
    arguments = dict(
        user_id="tg:42",
        title="Встреча",
        starts_at=moment(15),
        ends_at=moment(16),
        timezone_name="Europe/Moscow",
        operation_id="same-operation",
    )

    first, duplicate1 = await yandex.create_event(**arguments)
    second, duplicate2 = await yandex.create_event(**arguments)

    assert first["event_id"] == second["event_id"]
    assert duplicate1 is False
    assert duplicate2 is True
    assert len(remote.resources) == 1


async def test_yandex_provider_rejects_the_wrong_user_even_without_router() -> None:
    yandex = provider(FakeCalendar())

    try:
        await yandex.list_events("tg:99", moment(0), moment(23))
    except PermissionError as exc:
        assert "not configured" in str(exc)
    else:
        raise AssertionError("wrong Telegram user reached Yandex Calendar")


class RecordingProvider:
    def __init__(self, label):
        self.label = label
        self.users = []

    async def list_events(self, user_id, *_args):
        self.users.append(user_id)
        return [{"provider": self.label}]


async def test_router_uses_exact_namespaced_telegram_id() -> None:
    yandex = RecordingProvider("yandex")
    local = RecordingProvider("local")
    routed = RoutedCalendarProvider(routed_user_id="tg:42", routed=yandex, fallback=local)

    assert (await routed.list_events("tg:42", moment(0), moment(1)))[0]["provider"] == "yandex"
    assert (await routed.list_events("tg:420", moment(0), moment(1)))[0]["provider"] == "local"
    assert yandex.users == ["tg:42"]
    assert local.users == ["tg:420"]


def test_settings_normalize_bare_telegram_id_and_require_complete_credentials(tmp_path) -> None:
    configured = Settings(
        core_token="x" * 40,
        mcp_token="y" * 40,
        allowed_users=["tg:42"],
        data_dir=tmp_path,
        yandex_calendar_user_id="42",
        yandex_calendar_username="user@yandex.ru",
        yandex_calendar_app_password="secret",
    )
    assert configured.yandex_calendar_user_id == "tg:42"
    assert configured.validate_runtime() == []

    incomplete = Settings(
        core_token="x" * 40,
        mcp_token="y" * 40,
        allowed_users=["tg:42"],
        data_dir=tmp_path,
        yandex_calendar_user_id="42",
    )
    assert any("must be set together" in problem for problem in incomplete.validate_runtime())

    wrong_user = configured.model_copy(update={"yandex_calendar_user_id": "tg:99"})
    assert any("must also be present" in problem for problem in wrong_user.validate_runtime())
