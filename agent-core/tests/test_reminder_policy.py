"""Local reschedule policy: importance from the text, a nearby slot, no model."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from agent_core.reminders.policy import HIGH, LOW, NORMAL, classify, propose, snooze_at

MOSCOW = ZoneInfo("Europe/Moscow")


def test_medicine_and_meetings_are_high() -> None:
    assert classify("выпить лекарство") == HIGH
    assert classify("созвон с Иваном") == HIGH
    assert classify("купить молоко") == NORMAL
    assert classify("почитать статью") == LOW


def test_high_importance_is_a_few_minutes_away() -> None:
    now = datetime(2026, 8, 26, 12, 3, tzinfo=timezone.utc)  # 15:03 Moscow
    proposal = propose("выпить лекарство", now, MOSCOW)

    assert proposal.importance == HIGH
    local = proposal.due_at.astimezone(MOSCOW)
    assert local == datetime(2026, 8, 26, 15, 20, tzinfo=MOSCOW)


def test_ordinary_evening_moves_to_tomorrow_morning() -> None:
    now = datetime(2026, 8, 26, 18, 30, tzinfo=timezone.utc)  # 21:30 Moscow
    proposal = propose("купить молоко", now, MOSCOW)

    assert proposal.importance == NORMAL
    local = proposal.due_at.astimezone(MOSCOW)
    assert local == datetime(2026, 8, 27, 9, 0, tzinfo=MOSCOW)


def test_low_importance_is_tomorrow_same_clock() -> None:
    now = datetime(2026, 8, 26, 10, 0, tzinfo=timezone.utc)  # 13:00 Moscow
    proposal = propose("почитать статью", now, MOSCOW)

    assert proposal.importance == LOW
    local = proposal.due_at.astimezone(MOSCOW)
    assert local.day == 27
    assert local.hour == 13


def test_high_at_night_stays_close() -> None:
    now = datetime(2026, 8, 26, 22, 0, tzinfo=timezone.utc)  # 01:00 Moscow
    proposal = propose("таблетка", now, MOSCOW)
    delta = proposal.due_at - now
    assert timedelta(minutes=14) <= delta <= timedelta(minutes=20)


def test_refusing_the_first_offer_demotes_importance() -> None:
    now = datetime(2026, 8, 26, 10, 0, tzinfo=timezone.utc)
    first = propose("срочно позвонить врачу", now, MOSCOW)
    second = propose("срочно позвонить врачу", now, MOSCOW, later=True)
    assert first.importance == HIGH
    assert second.importance == NORMAL


def test_snooze_is_about_twenty_minutes() -> None:
    now = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
    due = snooze_at(now, MOSCOW)
    assert timedelta(minutes=20) <= due - now <= timedelta(minutes=24)
