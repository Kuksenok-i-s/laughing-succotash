"""Daily diary entries and monthly summaries.

One row per user per local calendar day. The evening check-in writes here; the month-end pass
reads the completed rows and stores the resulting summary. Neither depends on Cursor being up
for collection — only the narrative monthly write uses the agent, with a local fallback.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pa_protocol import new_ulid

from ..database import Database, from_iso, utcnow


def _load_obj(value: str | None) -> dict[str, Any]:
    return json.loads(value) if value else {}


def _dump_obj(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


@dataclass(slots=True)
class JournalEntry:
    entry_id: str
    user_id: str
    local_date: str
    status: str
    step: str
    answers: dict[str, Any]
    prompted_at: datetime | None
    completed_at: datetime | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> JournalEntry:
        return cls(
            entry_id=row["entry_id"],
            user_id=row["user_id"],
            local_date=row["local_date"],
            status=row["status"],
            step=row["step"],
            answers=_load_obj(row["answers"]),
            prompted_at=from_iso(row["prompted_at"]),
            completed_at=from_iso(row["completed_at"]),
        )

    def to_public(self) -> dict[str, Any]:
        answers = self.answers
        return {
            "entry_id": self.entry_id,
            "date": self.local_date,
            "status": self.status,
            "step": self.step,
            "work": answers.get("work") or None,
            "personal": answers.get("personal") or None,
            "mood": answers.get("mood"),
            "mood_label": answers.get("mood_label"),
            "progress": answers.get("progress"),
            "progress_label": answers.get("progress_label"),
            "tomorrow": answers.get("tomorrow") or None,
        }


@dataclass(slots=True)
class JournalSummary:
    summary_id: str
    user_id: str
    period: str
    body: str
    entry_count: int
    skipped_count: int
    status: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> JournalSummary:
        return cls(
            summary_id=row["summary_id"],
            user_id=row["user_id"],
            period=row["period"],
            body=row["body"] or "",
            entry_count=int(row["entry_count"] or 0),
            skipped_count=int(row["skipped_count"] or 0),
            status=row["status"],
        )

    def to_public(self) -> dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "period": self.period,
            "body": self.body,
            "entry_count": self.entry_count,
            "skipped_count": self.skipped_count,
            "status": self.status,
        }


class JournalRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def ensure(
        self, *, user_id: str, local_date: str, step: str = "offer"
    ) -> tuple[JournalEntry, bool]:
        """Create the day's row if it does not exist. Idempotent on (user_id, local_date)."""
        entry_id = new_ulid()
        now = utcnow().isoformat()

        def run(connection: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
            existing = connection.execute(
                "SELECT * FROM journal_entries WHERE user_id = ? AND local_date = ?",
                (user_id, local_date),
            ).fetchone()
            if existing is not None:
                return existing, True
            connection.execute(
                "INSERT INTO journal_entries(entry_id, user_id, local_date, status, step, "
                "answers, prompted_at, created_at, updated_at) "
                "VALUES (?, ?, ?, 'open', ?, '{}', ?, ?, ?)",
                (entry_id, user_id, local_date, step, now, now, now),
            )
            created = connection.execute(
                "SELECT * FROM journal_entries WHERE entry_id = ?", (entry_id,)
            ).fetchone()
            return created, False

        row, duplicate = await self._db.transaction(run)
        return JournalEntry.from_row(row), duplicate

    async def get(self, entry_id: str, user_id: str) -> JournalEntry | None:
        row = await self._db.fetch_one(
            "SELECT * FROM journal_entries WHERE entry_id = ? AND user_id = ?",
            (entry_id, user_id),
        )
        return JournalEntry.from_row(row) if row else None

    async def get_by_date(self, user_id: str, local_date: str) -> JournalEntry | None:
        row = await self._db.fetch_one(
            "SELECT * FROM journal_entries WHERE user_id = ? AND local_date = ?",
            (user_id, local_date),
        )
        return JournalEntry.from_row(row) if row else None

    async def open_for(self, user_id: str) -> JournalEntry | None:
        row = await self._db.fetch_one(
            "SELECT * FROM journal_entries WHERE user_id = ? AND status = 'open' "
            "ORDER BY local_date DESC LIMIT 1",
            (user_id,),
        )
        return JournalEntry.from_row(row) if row else None

    async def list_range(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        *,
        statuses: tuple[str, ...] | None = None,
    ) -> list[JournalEntry]:
        if statuses:
            placeholders = ",".join("?" * len(statuses))
            rows = await self._db.fetch_all(
                f"SELECT * FROM journal_entries WHERE user_id = ? AND local_date >= ? "
                f"AND local_date <= ? AND status IN ({placeholders}) "
                "ORDER BY local_date",
                (user_id, start_date, end_date, *statuses),
            )
        else:
            rows = await self._db.fetch_all(
                "SELECT * FROM journal_entries WHERE user_id = ? AND local_date >= ? "
                "AND local_date <= ? ORDER BY local_date",
                (user_id, start_date, end_date),
            )
        return [JournalEntry.from_row(row) for row in rows]

    async def search(
        self, user_id: str, query: str, *, limit: int = 40
    ) -> list[JournalEntry]:
        if not query.strip():
            rows = await self._db.fetch_all(
                "SELECT * FROM journal_entries WHERE user_id = ? AND status = 'complete' "
                "ORDER BY local_date DESC LIMIT ?",
                (user_id, limit),
            )
            return [JournalEntry.from_row(row) for row in rows]
        needle = f"%{query.strip()}%"
        rows = await self._db.fetch_all(
            "SELECT * FROM journal_entries WHERE user_id = ? AND status = 'complete' "
            "AND answers LIKE ? ORDER BY local_date DESC LIMIT ?",
            (user_id, needle, limit),
        )
        return [JournalEntry.from_row(row) for row in rows]

    async def update(
        self,
        entry_id: str,
        user_id: str,
        *,
        step: str | None = None,
        status: str | None = None,
        answers: dict[str, Any] | None = None,
        complete: bool = False,
    ) -> JournalEntry | None:
        current = await self.get(entry_id, user_id)
        if current is None:
            return None
        merged = dict(current.answers)
        if answers:
            merged.update(answers)
        new_step = step if step is not None else current.step
        new_status = status if status is not None else current.status
        now = utcnow().isoformat()
        completed_at = now if complete else (
            current.completed_at.isoformat() if current.completed_at else None
        )
        if complete:
            new_status = "complete"
            new_step = "done"

        await self._db.execute(
            "UPDATE journal_entries SET step = ?, status = ?, answers = ?, "
            "completed_at = ?, updated_at = ? WHERE entry_id = ? AND user_id = ?",
            (new_step, new_status, _dump_obj(merged), completed_at, now, entry_id, user_id),
        )
        return await self.get(entry_id, user_id)

    async def close_stale(self, user_id: str, before_date: str) -> int:
        """Yesterday's unfinished check-ins: keep what was said, skip empty offers."""
        rows = await self._db.fetch_all(
            "SELECT * FROM journal_entries WHERE user_id = ? AND status = 'open' "
            "AND local_date < ?",
            (user_id, before_date),
        )
        closed = 0
        now = utcnow().isoformat()
        for row in rows:
            entry = JournalEntry.from_row(row)
            if entry.answers:
                await self._db.execute(
                    "UPDATE journal_entries SET status = 'complete', step = 'done', "
                    "completed_at = ?, updated_at = ? WHERE entry_id = ?",
                    (now, now, entry.entry_id),
                )
            else:
                await self._db.execute(
                    "UPDATE journal_entries SET status = 'skipped', updated_at = ? "
                    "WHERE entry_id = ?",
                    (now, entry.entry_id),
                )
            closed += 1
        return closed

    async def reopen(self, entry_id: str, user_id: str, *, step: str) -> JournalEntry | None:
        now = utcnow().isoformat()
        await self._db.execute(
            "UPDATE journal_entries SET status = 'open', step = ?, completed_at = NULL, "
            "updated_at = ? WHERE entry_id = ? AND user_id = ?",
            (step, now, entry_id, user_id),
        )
        return await self.get(entry_id, user_id)

    # ---- summaries ----------------------------------------------------

    async def get_summary(self, user_id: str, period: str) -> JournalSummary | None:
        row = await self._db.fetch_one(
            "SELECT * FROM journal_summaries WHERE user_id = ? AND period = ?",
            (user_id, period),
        )
        return JournalSummary.from_row(row) if row else None

    async def latest_summary(self, user_id: str) -> JournalSummary | None:
        row = await self._db.fetch_one(
            "SELECT * FROM journal_summaries WHERE user_id = ? AND status = 'ready' "
            "ORDER BY period DESC LIMIT 1",
            (user_id,),
        )
        return JournalSummary.from_row(row) if row else None

    async def begin_summary(self, *, user_id: str, period: str) -> JournalSummary | None:
        """Claim the month. ``None`` means a ready summary already exists."""
        summary_id = new_ulid()
        now = utcnow().isoformat()

        def run(connection: sqlite3.Connection) -> sqlite3.Row | None:
            existing = connection.execute(
                "SELECT * FROM journal_summaries WHERE user_id = ? AND period = ?",
                (user_id, period),
            ).fetchone()
            if existing is not None:
                if existing["status"] == "ready":
                    return None
                return existing
            connection.execute(
                "INSERT INTO journal_summaries(summary_id, user_id, period, body, "
                "status, created_at, updated_at) VALUES (?, ?, ?, '', 'pending', ?, ?)",
                (summary_id, user_id, period, now, now),
            )
            return connection.execute(
                "SELECT * FROM journal_summaries WHERE summary_id = ?", (summary_id,)
            ).fetchone()

        row = await self._db.transaction(run)
        return JournalSummary.from_row(row) if row else None

    async def finish_summary(
        self,
        summary_id: str,
        *,
        body: str,
        entry_count: int,
        skipped_count: int,
        status: str = "ready",
    ) -> JournalSummary | None:
        now = utcnow().isoformat()
        await self._db.execute(
            "UPDATE journal_summaries SET body = ?, entry_count = ?, skipped_count = ?, "
            "status = ?, updated_at = ? WHERE summary_id = ?",
            (body, entry_count, skipped_count, status, now, summary_id),
        )
        row = await self._db.fetch_one(
            "SELECT * FROM journal_summaries WHERE summary_id = ?", (summary_id,)
        )
        return JournalSummary.from_row(row) if row else None
