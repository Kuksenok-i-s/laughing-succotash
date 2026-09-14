"""How soon to offer a new time when the user did not do the thing.

A reminder must still be re-scheduled when Cursor is down, so this is a small local rule rather
than a model call. The user always confirms the proposed instant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

HIGH = "high"
NORMAL = "normal"
LOW = "low"

HIGH_MARKERS = (
    "лекарств",
    "таблет",
    "врач",
    "встреч",
    "звон",
    "созвон",
    "дедлайн",
    "срочн",
    "важн",
    "самолёт",
    "самолет",
    "поезд",
    "оплат",
    "паспорт",
    "билет",
)
LOW_MARKERS = ("почитать", "подумать", "когда-нибудь", "идея", "интересн")

LABELS = {HIGH: "высокая", NORMAL: "обычная", LOW: "низкая"}
_DEMOTE = {HIGH: NORMAL, NORMAL: LOW, LOW: LOW}

_MONTHS = (
    "янв", "фев", "мар", "апр", "мая", "июн",
    "июл", "авг", "сен", "окт", "ноя", "дек",
)


@dataclass(frozen=True, slots=True)
class Proposal:
    importance: str
    due_at: datetime
    reason: str


def zone(name: str | None, fallback: str = "UTC") -> tzinfo:
    for candidate in (name, fallback):
        if not candidate:
            continue
        try:
            return ZoneInfo(candidate)
        except ZoneInfoNotFoundError:
            continue
    return timezone.utc


def classify(text: str) -> str:
    lowered = text.casefold()
    if any(marker in lowered for marker in HIGH_MARKERS):
        return HIGH
    if any(marker in lowered for marker in LOW_MARKERS):
        return LOW
    return NORMAL


def propose(
    text: str,
    now: datetime,
    user_tz: tzinfo,
    *,
    later: bool = False,
) -> Proposal:
    """Pick the nearest sensible slot. ``later`` is used after the user refused the first offer."""
    importance = classify(text)
    if later:
        importance = _DEMOTE[importance]
    local_now = now.astimezone(user_tz)

    if importance == HIGH:
        due_local = _ceil_minutes(local_now + timedelta(minutes=15), 5)
        reason = "важное — ближайшие минуты"
    elif importance == NORMAL:
        if local_now.hour < 20:
            due_local = _ceil_minutes(local_now + timedelta(hours=2), 5)
        else:
            due_local = (local_now + timedelta(days=1)).replace(
                hour=9, minute=0, second=0, microsecond=0
            )
        due_local = _skip_quiet_hours(due_local)
        reason = "обычное — ближайшее удобное время"
    else:
        due_local = (local_now + timedelta(days=1)).replace(second=0, microsecond=0)
        due_local = _skip_quiet_hours(due_local)
        reason = "можно подождать до завтра"

    return Proposal(importance, due_local.astimezone(timezone.utc), reason)


def snooze_at(now: datetime, user_tz: tzinfo) -> datetime:
    local = now.astimezone(user_tz) + timedelta(minutes=20)
    return _ceil_minutes(local, 5).astimezone(timezone.utc)


def format_when(value: datetime, user_tz: tzinfo) -> str:
    local = value.astimezone(user_tz)
    return f"{local.day} {_MONTHS[local.month - 1]} {local.strftime('%H:%M')}"


def _ceil_minutes(moment: datetime, step: int) -> datetime:
    moment = moment.replace(second=0, microsecond=0)
    remainder = moment.minute % step
    if remainder:
        moment += timedelta(minutes=step - remainder)
    return moment


def _skip_quiet_hours(local: datetime) -> datetime:
    """Keep ordinary and low items out of 23:00–07:00. High importance is not shifted here."""
    if 7 <= local.hour < 23:
        return local
    if local.hour >= 23:
        local = local + timedelta(days=1)
    return local.replace(hour=8, minute=0, second=0, microsecond=0)
