"""Storage guarantees: idempotency, isolation between users, and search."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent_core.storage.database import to_iso

NOW = datetime.now(timezone.utc)


# ---- idempotency ---------------------------------------------------------


async def test_the_same_reminder_operation_twice_creates_one_reminder(repos) -> None:
    first, dup1 = await repos.reminders.create(
        user_id="tg:1", text="один раз", due_at=NOW, timezone_name="UTC", operation_id="op"
    )
    second, dup2 = await repos.reminders.create(
        user_id="tg:1", text="один раз", due_at=NOW, timezone_name="UTC", operation_id="op"
    )

    assert dup1 is False and dup2 is True
    assert first.reminder_id == second.reminder_id
    assert len(await repos.reminders.list("tg:1")) == 1


async def test_the_same_calendar_operation_twice_creates_one_event(repos) -> None:
    args = {
        "user_id": "tg:1", "title": "Планёрка", "starts_at": NOW,
        "ends_at": NOW + timedelta(hours=1), "timezone_name": "UTC", "operation_id": "cal-op",
    }
    first, _ = await repos.calendar.create(**args)
    second, duplicate = await repos.calendar.create(**args)

    assert duplicate is True
    assert first["event_id"] == second["event_id"]


async def test_the_same_contact_operation_twice_creates_one_contact(repos) -> None:
    args = {
        "user_id": "tg:1", "display_name": "Саша Иванов", "aliases": ["@sasha"],
        "operation_id": "contact-op",
    }
    first, _ = await repos.contacts.create(**args)
    second, duplicate = await repos.contacts.create(**args)

    assert duplicate is True
    assert first["contact_id"] == second["contact_id"]
    assert len(await repos.contacts.search("tg:1", "саша")) == 1


async def test_the_same_request_id_maps_to_one_job(repos) -> None:
    first, dup1 = await repos.jobs.create_or_get(
        request_id="req-1", user_id="tg:1", kind="text"
    )
    second, dup2 = await repos.jobs.create_or_get(
        request_id="req-1", user_id="tg:1", kind="text"
    )

    assert (dup1, dup2) == (False, True)
    assert first.job_id == second.job_id


async def test_the_operation_ledger_replays_the_original_result(repos) -> None:
    await repos.operations.record("op-1", "calendar_create", "tg:1", {"event_id": "E1"})
    await repos.operations.record("op-1", "calendar_create", "tg:1", {"event_id": "E2"})

    # The second write is ignored: a retry must see what actually happened the first time.
    assert await repos.operations.lookup("op-1") == {"event_id": "E1"}
    assert await repos.operations.lookup("never-happened") is None


async def test_a_duplicate_delivery_id_is_never_queued_twice(repos) -> None:
    first = await repos.events.enqueue("telegram.send", {"text": "hi"}, delivery_id="d-1")
    second = await repos.events.enqueue("telegram.send", {"text": "hi"}, delivery_id="d-1")

    assert first is not None
    assert second is None
    assert await repos.events.pending_count() == 1


# ---- isolation -------------------------------------------------------------


async def test_users_cannot_read_each_others_objects(repos) -> None:
    note, _ = await repos.notes.create(user_id="tg:1", body="личное", operation_id="n-1")

    assert await repos.notes.get(note["note_id"], "tg:2") is None
    assert await repos.notes.search("tg:2", "личное") == []
    assert await repos.notes.delete(note["note_id"], "tg:2") is False
    assert await repos.notes.get(note["note_id"], "tg:1") is not None


async def test_users_cannot_read_each_others_contacts(repos) -> None:
    contact, _ = await repos.contacts.create(
        user_id="tg:1", display_name="Саша", operation_id="c-1"
    )

    assert await repos.contacts.get(contact["contact_id"], "tg:2") is None
    assert await repos.contacts.search("tg:2", "саша") == []
    assert await repos.contacts.update(contact["contact_id"], "tg:2", note="чужой") is None
    stored = await repos.contacts.get(contact["contact_id"], "tg:1")
    assert stored is not None and stored["note"] is None


async def test_users_cannot_read_each_others_journal(repos) -> None:
    entry, _ = await repos.journal.ensure(user_id="tg:1", local_date="2026-08-27")
    await repos.journal.update(
        entry.entry_id, "tg:1", answers={"work": "секрет"}, complete=True,
    )

    assert await repos.journal.get(entry.entry_id, "tg:2") is None
    assert await repos.journal.search("tg:2", "секрет") == []
    assert await repos.journal.get_by_date("tg:2", "2026-08-27") is None


async def test_one_journal_row_per_user_per_day(repos) -> None:
    first, dup1 = await repos.journal.ensure(user_id="tg:1", local_date="2026-08-27")
    second, dup2 = await repos.journal.ensure(user_id="tg:1", local_date="2026-08-27")

    assert dup1 is False and dup2 is True
    assert first.entry_id == second.entry_id


async def test_a_new_conversation_archives_the_previous_one(repos) -> None:
    first = await repos.conversations.create_conversation("tg:1")
    second = await repos.conversations.create_conversation("tg:1")

    active = await repos.conversations.active_conversation("tg:1")
    assert active.conversation_id == second.conversation_id
    assert first.conversation_id != second.conversation_id


# ---- search ------------------------------------------------------------------


async def test_notes_are_searchable_in_russian(repos) -> None:
    await repos.notes.create(
        user_id="tg:1", body="scheduler можно вынести в отдельный сервис", operation_id="n-1"
    )
    await repos.notes.create(user_id="tg:1", body="купить молоко", operation_id="n-2")

    found = await repos.notes.search("tg:1", "сервис")
    assert len(found) == 1
    assert "scheduler" in found[0]["body"]


async def test_a_search_query_with_fts_syntax_does_not_error(repos) -> None:
    """User text is not a query language; an unbalanced quote must degrade, not raise."""
    await repos.notes.create(user_id="tg:1", body="важная мысль", operation_id="n-1")
    assert await repos.notes.search("tg:1", 'что-то "') is not None


async def test_an_empty_query_returns_recent_notes(repos) -> None:
    await repos.notes.create(user_id="tg:1", body="первая", operation_id="n-1")
    await repos.notes.create(user_id="tg:1", body="вторая", operation_id="n-2")

    assert len(await repos.notes.search("tg:1", "")) == 2


async def test_deleting_a_note_removes_it_from_the_index(repos) -> None:
    note, _ = await repos.notes.create(user_id="tg:1", body="временная", operation_id="n-1")
    await repos.notes.delete(note["note_id"], "tg:1")

    assert await repos.notes.search("tg:1", "временная") == []


async def test_memory_is_separate_from_notes(repos) -> None:
    await repos.notes.create(user_id="tg:1", body="я люблю кофе", operation_id="n-1")
    await repos.memory.remember(user_id="tg:1", content="пьёт кофе без сахара", operation_id="m-1")

    assert len(await repos.memory.search("tg:1", "кофе")) == 1
    assert len(await repos.notes.search("tg:1", "кофе")) == 1


# ---- lifecycle ------------------------------------------------------------------


async def test_a_reminder_can_only_be_cancelled_once(repos) -> None:
    reminder, _ = await repos.reminders.create(
        user_id="tg:1", text="x", due_at=NOW, timezone_name="UTC", operation_id="op"
    )

    assert await repos.reminders.cancel(reminder.reminder_id, "tg:1") is True
    assert await repos.reminders.cancel(reminder.reminder_id, "tg:1") is False


async def test_a_fired_reminder_can_be_completed_or_put_back(repos) -> None:
    reminder, _ = await repos.reminders.create(
        user_id="tg:1", text="x", due_at=NOW, timezone_name="UTC", operation_id="op-fire"
    )
    await repos.reminders.mark_fired(reminder.reminder_id, None)

    later = NOW + timedelta(hours=2)
    restored = await repos.reminders.reschedule(reminder.reminder_id, "tg:1", later)
    assert restored is not None
    assert restored.status == "scheduled"
    assert restored.due_at == later

    await repos.reminders.mark_fired(reminder.reminder_id, None)
    assert await repos.reminders.complete(reminder.reminder_id, "tg:1") is True
    assert await repos.reminders.complete(reminder.reminder_id, "tg:1") is False
    assert await repos.reminders.reschedule(
        reminder.reminder_id, "tg:1", later
    ) is None


async def test_a_finished_job_cannot_be_finished_again(repos) -> None:
    job, _ = await repos.jobs.create_or_get(request_id="r", user_id="tg:1", kind="text")

    assert await repos.jobs.finish(job.job_id, "completed") is True
    assert await repos.jobs.finish(job.job_id, "failed") is False


async def test_interrupted_jobs_are_failed_on_startup(repos) -> None:
    job, _ = await repos.jobs.create_or_get(request_id="r", user_id="tg:1", kind="text")
    await repos.jobs.mark_running(job.job_id)

    assert await repos.jobs.recover_orphans() == 1
    assert (await repos.jobs.get(job.job_id)).error_code == "interrupted"


async def test_a_pending_action_resolves_once_and_only_for_its_owner(repos) -> None:
    action = await repos.pending_actions.create(
        user_id="tg:1", tool_name="calendar_delete", arguments={"event_id": "E"},
        operation_id="op", tier="dangerous", prompt_text="Удалить?", ttl_seconds=60,
    )

    assert await repos.pending_actions.resolve(action.action_id, "tg:2", "approved") == "unknown"
    assert await repos.pending_actions.resolve(action.action_id, "tg:1", "approved") == "applied"
    assert (
        await repos.pending_actions.resolve(action.action_id, "tg:1", "approved")
        == "already_resolved"
    )


async def test_an_expired_confirmation_cannot_be_approved(repos, db) -> None:
    action = await repos.pending_actions.create(
        user_id="tg:1", tool_name="task_delete", arguments={}, operation_id="op",
        tier="dangerous", prompt_text="?", ttl_seconds=60,
    )
    await db.execute(
        "UPDATE pending_actions SET expires_at = ? WHERE action_id = ?",
        (to_iso(NOW - timedelta(minutes=5)), action.action_id),
    )

    assert await repos.pending_actions.resolve(action.action_id, "tg:1", "approved") == "expired"


async def test_naive_datetimes_are_refused_at_the_storage_boundary() -> None:
    """A naive datetime here would silently mean UTC and fire a reminder at the wrong hour."""
    with pytest.raises(ValueError, match="naive datetime"):
        to_iso(datetime(2026, 8, 24, 18, 0))
