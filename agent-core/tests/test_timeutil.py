"""Time parsing. Getting this wrong means reminders fire at the wrong hour."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from agent_core.mcp.timeutil import (
    TimeParseError,
    day_bounds,
    format_local,
    next_occurrence,
    parse_datetime,
    parse_duration_seconds,
    validate_rrule,
)

MOSCOW = ZoneInfo("Europe/Moscow")


def test_a_naive_time_is_read_in_the_users_timezone() -> None:
    """18:00 typed by a Moscow user is 15:00 UTC, not 18:00 UTC."""
    parsed = parse_datetime("2026-08-24T18:00:00", MOSCOW)
    assert parsed == datetime(2026, 8, 24, 15, 0, tzinfo=timezone.utc)


def test_an_explicit_offset_is_respected() -> None:
    parsed = parse_datetime("2026-08-24T18:00:00+00:00", MOSCOW)
    assert parsed == datetime(2026, 8, 24, 18, 0, tzinfo=timezone.utc)


def test_unparseable_input_raises_rather_than_guessing() -> None:
    with pytest.raises(TimeParseError):
        parse_datetime("когда-нибудь потом", MOSCOW)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("17m", 17 * 60),
        ("17 мин", 17 * 60),
        ("1h30m", 5400),
        ("1 час 30 минут", 5400),
        ("90", 90),
        ("45s", 45),
    ],
)
def test_durations_are_parsed_in_both_languages(text: str, seconds: int) -> None:
    assert parse_duration_seconds(text) == seconds


@pytest.mark.parametrize("bad", ["", "скоро", "-5", "0"])
def test_a_nonsense_duration_is_refused(bad: str) -> None:
    with pytest.raises(TimeParseError):
        parse_duration_seconds(bad)


def test_a_week_long_timer_is_refused() -> None:
    """Timers are ephemeral; anything longer belongs in a reminder that survives a restart."""
    with pytest.raises(TimeParseError):
        parse_duration_seconds(str(86400 * 8))


def test_recurrence_produces_the_next_occurrence() -> None:
    after = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)  # Monday
    following = next_occurrence("FREQ=WEEKLY;BYDAY=MO;BYHOUR=10;BYMINUTE=0", after, MOSCOW)

    assert following is not None
    assert following > after
    assert following.astimezone(MOSCOW).weekday() == 0
    assert following.astimezone(MOSCOW).hour == 10


def test_an_invalid_rrule_is_rejected_at_creation_time() -> None:
    with pytest.raises(TimeParseError):
        validate_rrule("EVERY MONDAY PLEASE")


def test_day_bounds_are_local_days() -> None:
    moment = datetime(2026, 8, 24, 22, 0, tzinfo=timezone.utc)  # already the 25th in Moscow
    start, end = day_bounds(moment, MOSCOW)

    assert start.astimezone(MOSCOW).hour == 0
    assert start.astimezone(MOSCOW).day == 25
    assert (end - start).total_seconds() == 86400


def test_formatting_uses_the_users_clock() -> None:
    assert format_local("2026-08-24T15:00:00+00:00", MOSCOW) == "2026-08-24 18:00"
    assert format_local(None, MOSCOW) == "—"
