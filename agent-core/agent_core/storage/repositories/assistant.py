"""Repositories for the assistant's own objects.

Every creating method takes an ``operation_id`` and is idempotent on it: replaying the same tool
call after a lost response returns the original row instead of creating a second reminder, task or
calendar event.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pa_protocol import new_ulid

from ..database import Database, from_iso, to_iso, utcnow


def _json_list(value: Any) -> str:
    return json.dumps(list(value or []), ensure_ascii=False)


def _load_list(value: str | None) -> list[Any]:
    return json.loads(value) if value else []


@dataclass(slots=True)
class Reminder:
    reminder_id: str
    user_id: str
    text: str
    due_at: datetime | None
    timezone: str
    rrule: str | None
    status: str
    fire_count: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Reminder:
        return cls(
            reminder_id=row["reminder_id"],
            user_id=row["user_id"],
            text=row["text"],
            due_at=from_iso(row["due_at"]),
            timezone=row["timezone"],
            rrule=row["rrule"],
            status=row["status"],
            fire_count=row["fire_count"],
        )

    def to_public(self) -> dict[str, Any]:
        return {
            "reminder_id": self.reminder_id,
            "text": self.text,
            "due_at": to_iso(self.due_at),
            "timezone": self.timezone,
            "rrule": self.rrule,
            "status": self.status,
            "fire_count": self.fire_count,
        }


class ReminderRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(
        self,
        *,
        user_id: str,
        text: str,
        due_at: datetime | None,
        timezone_name: str,
        rrule: str | None = None,
        operation_id: str,
    ) -> tuple[Reminder, bool]:
        reminder_id = new_ulid()
        now = utcnow().isoformat()
        due_iso = to_iso(due_at)

        def run(connection: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
            existing = connection.execute(
                "SELECT * FROM reminders WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if existing is not None:
                return existing, True
            connection.execute(
                "INSERT INTO reminders(reminder_id, user_id, text, due_at, timezone, rrule, "
                "status, operation_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'scheduled', ?, ?, ?)",
                (reminder_id, user_id, text, due_iso, timezone_name, rrule, operation_id, now, now),
            )
            created = connection.execute(
                "SELECT * FROM reminders WHERE reminder_id = ?", (reminder_id,)
            ).fetchone()
            return created, False

        row, duplicate = await self._db.transaction(run)
        return Reminder.from_row(row), duplicate

    async def get(self, reminder_id: str, user_id: str) -> Reminder | None:
        row = await self._db.fetch_one(
            "SELECT * FROM reminders WHERE reminder_id = ? AND user_id = ?",
            (reminder_id, user_id),
        )
        return Reminder.from_row(row) if row else None

    async def list(self, user_id: str, *, status: str | None = "scheduled", limit: int = 50):
        if status:
            rows = await self._db.fetch_all(
                "SELECT * FROM reminders WHERE user_id = ? AND status = ? "
                "ORDER BY due_at IS NULL, due_at LIMIT ?",
                (user_id, status, limit),
            )
        else:
            rows = await self._db.fetch_all(
                "SELECT * FROM reminders WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
        return [Reminder.from_row(row) for row in rows]

    async def due_before(self, moment: datetime, limit: int = 100) -> list[Reminder]:
        rows = await self._db.fetch_all(
            "SELECT * FROM reminders WHERE status = 'scheduled' AND due_at IS NOT NULL "
            "AND due_at <= ? ORDER BY due_at LIMIT ?",
            (to_iso(moment), limit),
        )
        return [Reminder.from_row(row) for row in rows]

    async def update(
        self,
        reminder_id: str,
        user_id: str,
        *,
        text: str | None = None,
        due_at: datetime | None = None,
        rrule: str | None = None,
    ) -> Reminder | None:
        sets, params = [], []
        if text is not None:
            sets.append("text = ?")
            params.append(text)
        if due_at is not None:
            sets.append("due_at = ?")
            params.append(to_iso(due_at))
        if rrule is not None:
            sets.append("rrule = ?")
            params.append(rrule or None)
        if not sets:
            return await self.get(reminder_id, user_id)
        sets.append("updated_at = ?")
        params.extend([utcnow().isoformat(), reminder_id, user_id])
        await self._db.execute(
            f"UPDATE reminders SET {', '.join(sets)} WHERE reminder_id = ? AND user_id = ?",
            params,
        )
        return await self.get(reminder_id, user_id)

    async def cancel(self, reminder_id: str, user_id: str) -> bool:
        changed = await self._db.execute(
            "UPDATE reminders SET status = 'cancelled', updated_at = ? "
            "WHERE reminder_id = ? AND user_id = ? AND status IN ('scheduled', 'fired')",
            (utcnow().isoformat(), reminder_id, user_id),
        )
        return changed > 0

    async def complete(self, reminder_id: str, user_id: str) -> bool:
        """Mark a fired one-shot as done. Recurring rows stay scheduled and are not touched."""
        changed = await self._db.execute(
            "UPDATE reminders SET status = 'completed', updated_at = ? "
            "WHERE reminder_id = ? AND user_id = ? AND status = 'fired'",
            (utcnow().isoformat(), reminder_id, user_id),
        )
        return changed > 0

    async def reschedule(self, reminder_id: str, user_id: str, due_at: datetime) -> Reminder | None:
        """Put a fired one-shot back on the clock. Cancelled rows stay cancelled."""
        changed = await self._db.execute(
            "UPDATE reminders SET status = 'scheduled', due_at = ?, updated_at = ? "
            "WHERE reminder_id = ? AND user_id = ? AND status IN ('fired', 'scheduled')",
            (to_iso(due_at), utcnow().isoformat(), reminder_id, user_id),
        )
        if changed <= 0:
            return None
        return await self.get(reminder_id, user_id)

    async def mark_fired(self, reminder_id: str, next_due: datetime | None) -> None:
        """Advance a reminder after it fires.

        A recurring reminder gets its next occurrence and stays scheduled; a one-shot becomes
        terminal so it can never fire twice.
        """
        now = utcnow().isoformat()
        if next_due is not None:
            await self._db.execute(
                "UPDATE reminders SET due_at = ?, last_fired_at = ?, "
                "fire_count = fire_count + 1, updated_at = ? WHERE reminder_id = ?",
                (to_iso(next_due), now, now, reminder_id),
            )
        else:
            await self._db.execute(
                "UPDATE reminders SET status = 'fired', last_fired_at = ?, "
                "fire_count = fire_count + 1, updated_at = ? WHERE reminder_id = ?",
                (now, now, reminder_id),
            )

    async def pending_count(self) -> int:
        return int(
            await self._db.fetch_value(
                "SELECT COUNT(*) FROM reminders WHERE status = 'scheduled'"
            ) or 0
        )

    async def next_trigger(self) -> datetime | None:
        value = await self._db.fetch_value(
            "SELECT MIN(due_at) FROM reminders WHERE status = 'scheduled' AND due_at IS NOT NULL"
        )
        return from_iso(value)


class TimerRepository:
    """Timers are ephemeral: a countdown that fires once and is not a durable reminder."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(
        self, *, user_id: str, label: str | None, duration_seconds: int,
        fires_at: datetime, operation_id: str,
    ) -> tuple[dict[str, Any], bool]:
        timer_id = new_ulid()
        now = utcnow().isoformat()

        def run(connection: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
            existing = connection.execute(
                "SELECT * FROM timers WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if existing is not None:
                return existing, True
            connection.execute(
                "INSERT INTO timers(timer_id, user_id, label, duration_seconds, fires_at, "
                "status, operation_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)",
                (timer_id, user_id, label, duration_seconds, to_iso(fires_at), operation_id, now, now),
            )
            return connection.execute(
                "SELECT * FROM timers WHERE timer_id = ?", (timer_id,)
            ).fetchone(), False

        row, duplicate = await self._db.transaction(run)
        return self._public(row), duplicate

    async def list(self, user_id: str) -> list[dict[str, Any]]:
        rows = await self._db.fetch_all(
            "SELECT * FROM timers WHERE user_id = ? AND status = 'running' ORDER BY fires_at",
            (user_id,),
        )
        return [self._public(row) for row in rows]

    async def due_before(self, moment: datetime) -> list[dict[str, Any]]:
        rows = await self._db.fetch_all(
            "SELECT * FROM timers WHERE status = 'running' AND fires_at <= ? ORDER BY fires_at",
            (to_iso(moment),),
        )
        return [self._public(row) for row in rows]

    async def mark_fired(self, timer_id: str) -> None:
        await self._db.execute(
            "UPDATE timers SET status = 'fired', updated_at = ? WHERE timer_id = ?",
            (utcnow().isoformat(), timer_id),
        )

    async def cancel(self, timer_id: str, user_id: str) -> bool:
        changed = await self._db.execute(
            "UPDATE timers SET status = 'cancelled', updated_at = ? "
            "WHERE timer_id = ? AND user_id = ? AND status = 'running'",
            (utcnow().isoformat(), timer_id, user_id),
        )
        return changed > 0

    @staticmethod
    def _public(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "timer_id": row["timer_id"],
            "label": row["label"],
            "duration_seconds": row["duration_seconds"],
            "fires_at": row["fires_at"],
            "status": row["status"],
            "user_id": row["user_id"],
        }


class TaskRepository:
    """Tasks are things to do; reminders are things to be told about. Kept separate on purpose."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(
        self, *, user_id: str, title: str, details: str | None = None,
        due_at: datetime | None = None, priority: str = "normal", owner: str | None = None,
        tags: list[str] | None = None, source: str = "user", operation_id: str,
    ) -> tuple[dict[str, Any], bool]:
        task_id = new_ulid()
        now = utcnow().isoformat()

        def run(connection: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
            existing = connection.execute(
                "SELECT * FROM tasks WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if existing is not None:
                return existing, True
            connection.execute(
                "INSERT INTO tasks(task_id, user_id, title, details, status, priority, due_at, "
                "owner, tags, source, operation_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?)",
                (task_id, user_id, title, details, priority, to_iso(due_at), owner,
                 _json_list(tags), source, operation_id, now, now),
            )
            return connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone(), False

        row, duplicate = await self._db.transaction(run)
        return self._public(row), duplicate

    async def get(self, task_id: str, user_id: str) -> dict[str, Any] | None:
        row = await self._db.fetch_one(
            "SELECT * FROM tasks WHERE task_id = ? AND user_id = ?", (task_id, user_id)
        )
        return self._public(row) if row else None

    async def list(
        self, user_id: str, *, status: str = "open", limit: int = 50
    ) -> list[dict[str, Any]]:
        if status == "all":
            rows = await self._db.fetch_all(
                "SELECT * FROM tasks WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            )
        else:
            rows = await self._db.fetch_all(
                "SELECT * FROM tasks WHERE user_id = ? AND status = ? "
                "ORDER BY due_at IS NULL, due_at, created_at LIMIT ?",
                (user_id, status, limit),
            )
        return [self._public(row) for row in rows]

    async def update(self, task_id: str, user_id: str, **fields: Any) -> dict[str, Any] | None:
        allowed = {"title", "details", "priority", "owner", "status"}
        sets, params = [], []
        for key, value in fields.items():
            if key in allowed and value is not None:
                sets.append(f"{key} = ?")
                params.append(value)
        if "due_at" in fields and fields["due_at"] is not None:
            sets.append("due_at = ?")
            params.append(to_iso(fields["due_at"]))
        if "tags" in fields and fields["tags"] is not None:
            sets.append("tags = ?")
            params.append(_json_list(fields["tags"]))
        if not sets:
            return await self.get(task_id, user_id)
        sets.append("updated_at = ?")
        params.extend([utcnow().isoformat(), task_id, user_id])
        await self._db.execute(
            f"UPDATE tasks SET {', '.join(sets)} WHERE task_id = ? AND user_id = ?", params
        )
        return await self.get(task_id, user_id)

    async def complete(self, task_id: str, user_id: str) -> bool:
        now = utcnow().isoformat()
        changed = await self._db.execute(
            "UPDATE tasks SET status = 'done', completed_at = ?, updated_at = ? "
            "WHERE task_id = ? AND user_id = ? AND status = 'open'",
            (now, now, task_id, user_id),
        )
        return changed > 0

    async def delete(self, task_id: str, user_id: str) -> bool:
        changed = await self._db.execute(
            "DELETE FROM tasks WHERE task_id = ? AND user_id = ?", (task_id, user_id)
        )
        return changed > 0

    @staticmethod
    def _public(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "task_id": row["task_id"],
            "title": row["title"],
            "details": row["details"],
            "status": row["status"],
            "priority": row["priority"],
            "due_at": row["due_at"],
            "owner": row["owner"],
            "tags": _load_list(row["tags"]),
            "source": row["source"],
        }


class NoteRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(
        self, *, user_id: str, body: str, title: str | None = None,
        tags: list[str] | None = None, source: str = "user", operation_id: str,
    ) -> tuple[dict[str, Any], bool]:
        note_id = new_ulid()
        now = utcnow().isoformat()

        def run(connection: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
            existing = connection.execute(
                "SELECT * FROM notes WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if existing is not None:
                return existing, True
            connection.execute(
                "INSERT INTO notes(note_id, user_id, title, body, tags, source, operation_id, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (note_id, user_id, title, body, _json_list(tags), source, operation_id, now, now),
            )
            connection.execute(
                "INSERT INTO notes_fts(title, body, note_id) VALUES (?, ?, ?)",
                (title or "", body, note_id),
            )
            return connection.execute(
                "SELECT * FROM notes WHERE note_id = ?", (note_id,)
            ).fetchone(), False

        row, duplicate = await self._db.transaction(run)
        return self._public(row), duplicate

    async def get(self, note_id: str, user_id: str) -> dict[str, Any] | None:
        row = await self._db.fetch_one(
            "SELECT * FROM notes WHERE note_id = ? AND user_id = ?", (note_id, user_id)
        )
        return self._public(row) if row else None

    async def search(self, user_id: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
        if not query.strip():
            rows = await self._db.fetch_all(
                "SELECT * FROM notes WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            )
            return [self._public(row) for row in rows]
        try:
            rows = await self._db.fetch_all(
                "SELECT n.* FROM notes_fts f JOIN notes n ON n.note_id = f.note_id "
                "WHERE notes_fts MATCH ? AND n.user_id = ? ORDER BY rank LIMIT ?",
                (_fts_query(query), user_id, limit),
            )
        except sqlite3.OperationalError:
            # A query FTS5 cannot parse should degrade to substring matching rather than
            # surfacing a database error to the agent.
            rows = await self._db.fetch_all(
                "SELECT * FROM notes WHERE user_id = ? AND (body LIKE ? OR title LIKE ?) "
                "ORDER BY updated_at DESC LIMIT ?",
                (user_id, f"%{query}%", f"%{query}%", limit),
            )
        return [self._public(row) for row in rows]

    async def update(
        self, note_id: str, user_id: str, *, body: str | None = None,
        title: str | None = None, tags: list[str] | None = None,
    ) -> dict[str, Any] | None:
        current = await self.get(note_id, user_id)
        if current is None:
            return None
        new_body = body if body is not None else current["body"]
        new_title = title if title is not None else current["title"]
        new_tags = tags if tags is not None else current["tags"]

        def run(connection: sqlite3.Connection) -> None:
            connection.execute(
                "UPDATE notes SET title = ?, body = ?, tags = ?, updated_at = ? "
                "WHERE note_id = ? AND user_id = ?",
                (new_title, new_body, _json_list(new_tags), utcnow().isoformat(), note_id, user_id),
            )
            connection.execute("DELETE FROM notes_fts WHERE note_id = ?", (note_id,))
            connection.execute(
                "INSERT INTO notes_fts(title, body, note_id) VALUES (?, ?, ?)",
                (new_title or "", new_body, note_id),
            )

        await self._db.transaction(run)
        return await self.get(note_id, user_id)

    async def delete(self, note_id: str, user_id: str) -> bool:
        def run(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(
                "DELETE FROM notes WHERE note_id = ? AND user_id = ?", (note_id, user_id)
            )
            connection.execute("DELETE FROM notes_fts WHERE note_id = ?", (note_id,))
            return cursor.rowcount

        return await self._db.transaction(run) > 0

    @staticmethod
    def _public(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "note_id": row["note_id"],
            "title": row["title"],
            "body": row["body"],
            "tags": _load_list(row["tags"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


class MemoryRepository:
    """Long-term facts. Written only on explicit instruction or confirmed proposal (ADR 7)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def remember(
        self, *, user_id: str, content: str, category: str | None = None,
        source: str = "explicit", operation_id: str,
    ) -> tuple[dict[str, Any], bool]:
        memory_id = new_ulid()
        now = utcnow().isoformat()

        def run(connection: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
            existing = connection.execute(
                "SELECT * FROM memory WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if existing is not None:
                return existing, True
            connection.execute(
                "INSERT INTO memory(memory_id, user_id, content, category, source, "
                "operation_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (memory_id, user_id, content, category, source, operation_id, now, now),
            )
            connection.execute(
                "INSERT INTO memory_fts(content, category, memory_id) VALUES (?, ?, ?)",
                (content, category or "", memory_id),
            )
            return connection.execute(
                "SELECT * FROM memory WHERE memory_id = ?", (memory_id,)
            ).fetchone(), False

        row, duplicate = await self._db.transaction(run)
        return self._public(row), duplicate

    async def search(self, user_id: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
        if not query.strip():
            rows = await self._db.fetch_all(
                "SELECT * FROM memory WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            )
            return [self._public(row) for row in rows]
        try:
            rows = await self._db.fetch_all(
                "SELECT m.* FROM memory_fts f JOIN memory m ON m.memory_id = f.memory_id "
                "WHERE memory_fts MATCH ? AND m.user_id = ? ORDER BY rank LIMIT ?",
                (_fts_query(query), user_id, limit),
            )
        except sqlite3.OperationalError:
            rows = await self._db.fetch_all(
                "SELECT * FROM memory WHERE user_id = ? AND content LIKE ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (user_id, f"%{query}%", limit),
            )
        return [self._public(row) for row in rows]

    async def get(self, memory_id: str, user_id: str) -> dict[str, Any] | None:
        row = await self._db.fetch_one(
            "SELECT * FROM memory WHERE memory_id = ? AND user_id = ?", (memory_id, user_id)
        )
        return self._public(row) if row else None

    async def forget(self, memory_id: str, user_id: str) -> bool:
        def run(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(
                "DELETE FROM memory WHERE memory_id = ? AND user_id = ?", (memory_id, user_id)
            )
            connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
            return cursor.rowcount

        return await self._db.transaction(run) > 0

    @staticmethod
    def _public(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "memory_id": row["memory_id"],
            "content": row["content"],
            "category": row["category"],
            "source": row["source"],
            "created_at": row["created_at"],
        }


class ContactRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def upsert(
        self, *, user_id: str, display_name: str, aliases: list[str] | None = None,
        emails: list[str] | None = None, phones: list[str] | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        contact_id = new_ulid()
        now = utcnow().isoformat()
        await self._db.execute(
            "INSERT INTO contacts(contact_id, user_id, display_name, aliases, emails, phones, "
            "note, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (contact_id, user_id, display_name, _json_list(aliases), _json_list(emails),
             _json_list(phones), note, now, now),
        )
        row = await self._db.fetch_one(
            "SELECT * FROM contacts WHERE contact_id = ?", (contact_id,)
        )
        return self._public(row)

    async def search(self, user_id: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Substring match across name and aliases.

        Matching is done in Python: SQLite's LOWER() and LIKE only fold ASCII, so "Саша" would
        never match "саша" in SQL. Contact lists are small enough that scanning is fine.

        Returns every match rather than a best guess — when several people match, the agent is
        required to ask which one rather than pick.
        """
        needle = query.strip().casefold()
        rows = await self._db.fetch_all(
            "SELECT * FROM contacts WHERE user_id = ? ORDER BY display_name", (user_id,)
        )
        matches = [
            row
            for row in rows
            if not needle
            or needle in row["display_name"].casefold()
            or needle in (row["aliases"] or "").casefold()
        ]
        return [self._public(row) for row in matches[:limit]]

    async def get(self, contact_id: str, user_id: str) -> dict[str, Any] | None:
        row = await self._db.fetch_one(
            "SELECT * FROM contacts WHERE contact_id = ? AND user_id = ?", (contact_id, user_id)
        )
        return self._public(row) if row else None

    @staticmethod
    def _public(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "contact_id": row["contact_id"],
            "display_name": row["display_name"],
            "aliases": _load_list(row["aliases"]),
            "emails": _load_list(row["emails"]),
            "phones": _load_list(row["phones"]),
            "note": row["note"],
        }


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 prefix query.

    User text can contain FTS operators that would either error or mean something unintended, so
    each token is quoted and turned into a prefix match.
    """
    tokens = [t for t in "".join(c if c.isalnum() or c.isspace() else " " for c in query).split() if t]
    if not tokens:
        return '""'
    return " ".join(f'"{token}"*' for token in tokens)
