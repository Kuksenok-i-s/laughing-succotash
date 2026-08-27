"""Pending confirmations, calendar storage and the operation idempotency ledger."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from pa_protocol import new_ulid

from ..database import Database, from_iso, to_iso, utcnow


@dataclass(slots=True)
class PendingAction:
    action_id: str
    user_id: str
    chat_id: int | None
    job_id: str | None
    tool_name: str
    arguments: dict[str, Any]
    operation_id: str
    tier: str
    prompt_text: str
    status: str
    expires_at: datetime | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> PendingAction:
        return cls(
            action_id=row["action_id"],
            user_id=row["user_id"],
            chat_id=row["chat_id"],
            job_id=row["job_id"],
            tool_name=row["tool_name"],
            arguments=json.loads(row["arguments"]),
            operation_id=row["operation_id"],
            tier=row["tier"],
            prompt_text=row["prompt_text"],
            status=row["status"],
            expires_at=from_iso(row["expires_at"]),
        )


class PendingActionRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(
        self, *, user_id: str, tool_name: str, arguments: dict[str, Any],
        operation_id: str, tier: str, prompt_text: str, ttl_seconds: int,
        chat_id: int | None = None, job_id: str | None = None,
    ) -> PendingAction:
        action_id = new_ulid()
        now = utcnow()
        await self._db.execute(
            "INSERT INTO pending_actions(action_id, user_id, chat_id, job_id, tool_name, "
            "arguments, operation_id, tier, prompt_text, status, expires_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
            (action_id, user_id, chat_id, job_id, tool_name,
             json.dumps(arguments, ensure_ascii=False), operation_id, tier, prompt_text,
             to_iso(now + timedelta(seconds=ttl_seconds)), now.isoformat()),
        )
        row = await self._db.fetch_one(
            "SELECT * FROM pending_actions WHERE action_id = ?", (action_id,)
        )
        return PendingAction.from_row(row)

    async def get(self, action_id: str) -> PendingAction | None:
        row = await self._db.fetch_one(
            "SELECT * FROM pending_actions WHERE action_id = ?", (action_id,)
        )
        return PendingAction.from_row(row) if row else None

    async def get_by_operation_id(self, operation_id: str) -> PendingAction | None:
        row = await self._db.fetch_one(
            "SELECT * FROM pending_actions WHERE operation_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (operation_id,),
        )
        return PendingAction.from_row(row) if row else None

    async def resolve(self, action_id: str, user_id: str, status: str) -> str:
        """Atomically move a pending action to a terminal state.

        Returns what happened, so a double button press reports ``already_resolved`` rather than
        executing the action a second time.
        """

        def run(connection: sqlite3.Connection) -> str:
            row = connection.execute(
                "SELECT * FROM pending_actions WHERE action_id = ?", (action_id,)
            ).fetchone()
            if row is None:
                return "unknown"
            # A confirmation is bound to the user it was shown to; another user pressing the
            # button (or a Gateway replaying someone else's callback) must not resolve it.
            if row["user_id"] != user_id:
                return "unknown"
            if row["status"] != "pending":
                return "already_resolved"
            expires_at = from_iso(row["expires_at"])
            if expires_at is not None and expires_at < utcnow():
                connection.execute(
                    "UPDATE pending_actions SET status = 'expired', resolved_at = ? "
                    "WHERE action_id = ?",
                    (utcnow().isoformat(), action_id),
                )
                return "expired"
            connection.execute(
                "UPDATE pending_actions SET status = ?, resolved_at = ? WHERE action_id = ?",
                (status, utcnow().isoformat(), action_id),
            )
            return "applied"

        return await self._db.transaction(run)

    async def expire_overdue(self) -> list[PendingAction]:
        rows = await self._db.fetch_all(
            "SELECT * FROM pending_actions WHERE status = 'pending' AND expires_at < ?",
            (to_iso(utcnow()),),
        )
        actions = [PendingAction.from_row(row) for row in rows]
        if actions:
            await self._db.execute(
                "UPDATE pending_actions SET status = 'expired', resolved_at = ? "
                "WHERE status = 'pending' AND expires_at < ?",
                (utcnow().isoformat(), to_iso(utcnow())),
            )
        return actions

    async def pending_for_user(self, user_id: str) -> list[PendingAction]:
        rows = await self._db.fetch_all(
            "SELECT * FROM pending_actions WHERE user_id = ? AND status = 'pending' "
            "ORDER BY created_at",
            (user_id,),
        )
        return [PendingAction.from_row(row) for row in rows]


class OperationLedger:
    """Records completed side-effecting tool calls so a replay returns the original result."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def lookup(self, operation_id: str) -> dict[str, Any] | None:
        value = await self._db.fetch_value(
            "SELECT result FROM operations WHERE operation_id = ?", (operation_id,)
        )
        return json.loads(value) if value else None

    async def record(
        self, operation_id: str, tool_name: str, user_id: str, result: dict[str, Any]
    ) -> None:
        await self._db.execute(
            "INSERT INTO operations(operation_id, tool_name, user_id, result, created_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(operation_id) DO NOTHING",
            (operation_id, tool_name, user_id, json.dumps(result, ensure_ascii=False),
             utcnow().isoformat()),
        )


class CalendarRepository:
    """Storage behind the built-in local calendar provider.

    The MCP contract is defined against ``CalendarProvider``, not against this table, so a macOS
    EventKit or CalDAV backend can replace it without touching the tool schemas.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(
        self, *, user_id: str, title: str, starts_at: datetime, ends_at: datetime,
        timezone_name: str, calendar: str = "default", location: str | None = None,
        description: str | None = None, attendees: list[str] | None = None,
        operation_id: str,
    ) -> tuple[dict[str, Any], bool]:
        event_id = new_ulid()
        now = utcnow().isoformat()

        def run(connection: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
            existing = connection.execute(
                "SELECT * FROM calendar_events WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if existing is not None:
                return existing, True
            connection.execute(
                "INSERT INTO calendar_events(event_id, user_id, calendar, title, starts_at, "
                "ends_at, timezone, location, description, attendees, operation_id, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event_id, user_id, calendar, title, to_iso(starts_at), to_iso(ends_at),
                 timezone_name, location, description,
                 json.dumps(list(attendees or []), ensure_ascii=False), operation_id, now, now),
            )
            return connection.execute(
                "SELECT * FROM calendar_events WHERE event_id = ?", (event_id,)
            ).fetchone(), False

        row, duplicate = await self._db.transaction(run)
        return self._public(row), duplicate

    async def list_range(
        self, user_id: str, start: datetime, end: datetime, limit: int = 100
    ) -> list[dict[str, Any]]:
        # Overlap, not containment: a meeting that starts before the window and ends inside it is
        # still "what I have tomorrow".
        rows = await self._db.fetch_all(
            "SELECT * FROM calendar_events WHERE user_id = ? AND status = 'confirmed' "
            "AND starts_at < ? AND ends_at > ? ORDER BY starts_at LIMIT ?",
            (user_id, to_iso(end), to_iso(start), limit),
        )
        return [self._public(row) for row in rows]

    async def get(self, event_id: str, user_id: str) -> dict[str, Any] | None:
        row = await self._db.fetch_one(
            "SELECT * FROM calendar_events WHERE event_id = ? AND user_id = ?",
            (event_id, user_id),
        )
        return self._public(row) if row else None

    async def update(self, event_id: str, user_id: str, **fields: Any) -> dict[str, Any] | None:
        sets, params = [], []
        for key in ("title", "location", "description"):
            if fields.get(key) is not None:
                sets.append(f"{key} = ?")
                params.append(fields[key])
        for key in ("starts_at", "ends_at"):
            if fields.get(key) is not None:
                sets.append(f"{key} = ?")
                params.append(to_iso(fields[key]))
        if not sets:
            return await self.get(event_id, user_id)
        sets.append("updated_at = ?")
        params.extend([utcnow().isoformat(), event_id, user_id])
        await self._db.execute(
            f"UPDATE calendar_events SET {', '.join(sets)} WHERE event_id = ? AND user_id = ?",
            params,
        )
        return await self.get(event_id, user_id)

    async def delete(self, event_id: str, user_id: str) -> bool:
        changed = await self._db.execute(
            "UPDATE calendar_events SET status = 'cancelled', updated_at = ? "
            "WHERE event_id = ? AND user_id = ? AND status = 'confirmed'",
            (utcnow().isoformat(), event_id, user_id),
        )
        return changed > 0

    @staticmethod
    def _public(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "calendar": row["calendar"],
            "title": row["title"],
            "starts_at": row["starts_at"],
            "ends_at": row["ends_at"],
            "timezone": row["timezone"],
            "location": row["location"],
            "description": row["description"],
            "attendees": json.loads(row["attendees"] or "[]"),
            "status": row["status"],
        }
