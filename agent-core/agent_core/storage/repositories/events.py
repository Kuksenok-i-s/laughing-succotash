"""Durable Core → Gateway event log.

This is what makes "the reminder fires while the Gateway is offline and is delivered exactly once
afterwards" true. An event is committed to SQLite *before* it is written to the socket, and it is
only marked sent once the Gateway has acknowledged it.

Delivery is at-least-once; the Gateway deduplicates on ``delivery_id``. We additionally refuse to
enqueue a second event with the same ``delivery_id`` so a retry inside the Core cannot even create
the duplicate in the first place.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from pa_protocol import new_ulid

from ..database import Database, utcnow

log = logging.getLogger(__name__)

STATE_LAST_ACK_SEQ = "last_ack_seq"


@dataclass(slots=True)
class OutboundEvent:
    seq: int
    event_id: str
    delivery_id: str | None
    method: str
    params: dict[str, Any]
    user_id: str | None
    attempts: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> OutboundEvent:
        return cls(
            seq=row["seq"],
            event_id=row["event_id"],
            delivery_id=row["delivery_id"],
            method=row["method"],
            params=json.loads(row["params"]),
            user_id=row["user_id"],
            attempts=row["attempts"],
        )


class OutboundEventRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def enqueue(
        self,
        method: str,
        params: dict[str, Any],
        *,
        delivery_id: str | None = None,
        user_id: str | None = None,
        event_id: str | None = None,
    ) -> OutboundEvent | None:
        """Durably record an outbound event.

        Returns ``None`` when an event with the same ``delivery_id`` already exists, which means
        the caller is retrying something already recorded and must not send it again.
        """
        event_id = event_id or new_ulid()
        payload = json.dumps(params, ensure_ascii=False)
        now = utcnow().isoformat()

        def run(connection: sqlite3.Connection) -> OutboundEvent | None:
            if delivery_id is not None:
                existing = connection.execute(
                    "SELECT seq FROM outbound_events WHERE delivery_id = ?", (delivery_id,)
                ).fetchone()
                if existing is not None:
                    return None
            cursor = connection.execute(
                "INSERT INTO outbound_events"
                "(event_id, delivery_id, method, params, user_id, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (event_id, delivery_id, method, payload, user_id, now),
            )
            return OutboundEvent(
                seq=int(cursor.lastrowid or 0),
                event_id=event_id,
                delivery_id=delivery_id,
                method=method,
                params=params,
                user_id=user_id,
                attempts=0,
            )

        return await self._db.transaction(run)

    async def pending(self, *, after_seq: int = 0, limit: int = 200) -> list[OutboundEvent]:
        """Unsent events in creation order — the replay set after a reconnect."""
        rows = await self._db.fetch_all(
            "SELECT * FROM outbound_events WHERE status = 'pending' AND seq > ? "
            "ORDER BY seq LIMIT ?",
            (after_seq, limit),
        )
        return [OutboundEvent.from_row(row) for row in rows]

    async def mark_sent(self, seq: int) -> None:
        await self._db.execute(
            "UPDATE outbound_events SET status = 'sent', sent_at = ? WHERE seq = ?",
            (utcnow().isoformat(), seq),
        )

    async def mark_attempt_failed(self, seq: int, error: str) -> None:
        await self._db.execute(
            "UPDATE outbound_events SET attempts = attempts + 1, last_error = ? WHERE seq = ?",
            (error[:500], seq),
        )

    async def drop(self, seq: int, error: str) -> None:
        """Give up on an event permanently — e.g. the user blocked the bot."""
        await self._db.execute(
            "UPDATE outbound_events SET status = 'dropped', last_error = ? WHERE seq = ?",
            (error[:500], seq),
        )

    async def max_seq(self) -> int:
        return int(await self._db.fetch_value("SELECT COALESCE(MAX(seq), 0) FROM outbound_events") or 0)

    async def pending_count(self) -> int:
        return int(
            await self._db.fetch_value(
                "SELECT COUNT(*) FROM outbound_events WHERE status = 'pending'"
            )
            or 0
        )

    async def acknowledge_through(self, seq: int) -> None:
        """Mark everything up to ``seq`` as sent.

        Used after the handshake when the Gateway reports a ``last_received_seq`` ahead of what we
        recorded — the events did arrive, we just lost the acknowledgement.
        """
        await self._db.execute(
            "UPDATE outbound_events SET status = 'sent', sent_at = ? "
            "WHERE seq <= ? AND status = 'pending'",
            (utcnow().isoformat(), seq),
        )

    # ---- peer sequence bookkeeping ------------------------------------

    async def get_state(self, key: str, default: int = 0) -> int:
        value = await self._db.fetch_value("SELECT value FROM delivery_state WHERE key = ?", (key,))
        return int(value) if value is not None else default

    async def set_state(self, key: str, value: int) -> None:
        await self._db.execute(
            "INSERT INTO delivery_state(key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, str(value), utcnow().isoformat()),
        )

    async def prune(self, keep_last: int = 5000) -> int:
        """Drop old delivered events so the log does not grow forever."""
        cutoff = await self._db.fetch_value(
            "SELECT seq FROM outbound_events WHERE status IN ('sent','dropped') "
            "ORDER BY seq DESC LIMIT 1 OFFSET ?",
            (keep_last,),
        )
        if cutoff is None:
            return 0
        return await self._db.execute(
            "DELETE FROM outbound_events WHERE seq <= ? AND status IN ('sent','dropped')",
            (cutoff,),
        )
