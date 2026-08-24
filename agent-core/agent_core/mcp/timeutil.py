"""Timezone-aware parsing for tool arguments.

Every stored instant is timezone-aware UTC. A naive datetime from the agent is interpreted in the
*user's* timezone, never the server's — "завтра в 10" means ten o'clock where the user is.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone, tzinfo

from dateutil import parser as date_parser
from dateutil.rrule import rrulestr

_DURATION = re.compile(
    r"^\s*(?:(?P<hours>\d+)\s*(?:h|hr|hours?|ч|час(?:а|ов)?))?\s*"
    r"(?:(?P<minutes>\d+)\s*(?:m|min|minutes?|м|мин(?:ут[ыа]?)?))?\s*"
    r"(?:(?P<seconds>\d+)\s*(?:s|sec|seconds?|с|сек(?:унд[ыа]?)?))?\s*$",
    re.IGNORECASE,
)


class TimeParseError(ValueError):
    pass


def parse_datetime(value: str | datetime, user_tz: tzinfo) -> datetime:
    """Parse an instant and return it as timezone-aware UTC."""
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = date_parser.isoparse(value)
        except (ValueError, OverflowError):
            try:
                parsed = date_parser.parse(value)
            except (ValueError, OverflowError) as exc:
                raise TimeParseError(f"could not parse datetime: {value!r}") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=user_tz)
    return parsed.astimezone(timezone.utc)


def parse_duration_seconds(value: str | int | float) -> int:
    """Parse ``"17m"``, ``"1h30m"``, ``"90"`` or a number into seconds."""
    if isinstance(value, (int, float)):
        seconds = int(value)
    else:
        text = value.strip()
        if text.isdigit():
            seconds = int(text)
        else:
            match = _DURATION.match(text)
            if match is None or not any(match.groupdict().values()):
                raise TimeParseError(f"could not parse duration: {value!r}")
            seconds = (
                int(match.group("hours") or 0) * 3600
                + int(match.group("minutes") or 0) * 60
                + int(match.group("seconds") or 0)
            )
    if seconds <= 0:
        raise TimeParseError("duration must be positive")
    if seconds > 86400 * 7:
        raise TimeParseError("duration exceeds one week; use a reminder instead")
    return seconds


def next_occurrence(rrule_text: str, after: datetime, user_tz: tzinfo) -> datetime | None:
    """Next fire time for an RRULE, strictly after ``after``.

    Recurrence is evaluated in the user's local timezone so that "every Monday at 10:00" stays at
    10:00 across a DST change rather than drifting by an hour.
    """
    local_after = after.astimezone(user_tz)
    try:
        rule = rrulestr(rrule_text, dtstart=local_after.replace(tzinfo=user_tz))
    except (ValueError, TypeError) as exc:
        raise TimeParseError(f"invalid RRULE: {rrule_text!r}") from exc

    upcoming = rule.after(local_after, inc=False)
    if upcoming is None:
        return None
    if upcoming.tzinfo is None:
        upcoming = upcoming.replace(tzinfo=user_tz)
    return upcoming.astimezone(timezone.utc)


def validate_rrule(rrule_text: str) -> None:
    try:
        rrulestr(rrule_text, dtstart=datetime.now(timezone.utc))
    except (ValueError, TypeError) as exc:
        raise TimeParseError(f"invalid RRULE: {rrule_text!r}") from exc


def day_bounds(day: datetime, user_tz: tzinfo) -> tuple[datetime, datetime]:
    """Start and end of ``day`` in the user's timezone, returned as UTC."""
    local = day.astimezone(user_tz)
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(timezone.utc), (start + timedelta(days=1)).astimezone(timezone.utc)


def format_local(value: datetime | str | None, user_tz: tzinfo) -> str:
    """Render an instant in the user's timezone for display back to the model."""
    if value is None:
        return "—"
    if isinstance(value, str):
        try:
            value = date_parser.isoparse(value)
        except ValueError:
            return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(user_tz).strftime("%Y-%m-%d %H:%M")
