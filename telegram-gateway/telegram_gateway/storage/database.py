"""Gateway-local SQLite: transport state only.

Nothing here is agent state. Losing this database loses in-flight requests and delivery
bookkeeping; it cannot lose a task, note, memory, reminder or calendar entry, because there is
nowhere in this schema to put one.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

SCHEMA = """
-- Inbound Telegram interactions handed to the Core. Survives a Core outage so the request can be
-- submitted after reconnect.
CREATE TABLE IF NOT EXISTS pending_requests (
    request_id   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    chat_id      INTEGER NOT NULL,
    message_id   INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    payload      TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending|submitted|failed
    job_id       TEXT,
    attempts     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pending_requests_status
    ON pending_requests(status, created_at);

CREATE TABLE IF NOT EXISTS pending_uploads (
    request_id   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    chat_id      INTEGER NOT NULL,
    message_id   INTEGER NOT NULL,
    file_path    TEXT NOT NULL,
    filename     TEXT NOT NULL,
    content_type TEXT,
    size         INTEGER NOT NULL,
    duration_seconds REAL,
    purpose      TEXT NOT NULL DEFAULT 'assistant',
    sha256       TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending|uploading|done|failed
    attempts     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pending_uploads_status ON pending_uploads(status, created_at);

-- Records every Core-originated delivery we have executed, keyed by delivery_id. This is the
-- dedup table that makes at-least-once delivery safe: a replayed event finds its row and returns
-- the original Telegram message_id instead of sending twice.
CREATE TABLE IF NOT EXISTS outbound_delivery (
    delivery_id  TEXT PRIMARY KEY,
    method       TEXT NOT NULL,
    chat_id      INTEGER,
    message_id   INTEGER,
    status       TEXT NOT NULL,               -- done|failed
    error        TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processed_event_ids (
    event_id   TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rpc_sequence_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Maps opaque inline-button callback data to a Core action. The Core never constructs Telegram
-- callback data; it describes intent and the Gateway owns the encoding.
CREATE TABLE IF NOT EXISTS confirmations (
    token       TEXT PRIMARY KEY,
    action_id   TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    chat_id     INTEGER NOT NULL,
    message_id  INTEGER,
    choice      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_confirmations_action ON confirmations(action_id);

-- One status message per job, edited in place instead of posting a line per progress event.
CREATE TABLE IF NOT EXISTS job_status_messages (
    job_id      TEXT PRIMARY KEY,
    chat_id     INTEGER NOT NULL,
    message_id  INTEGER NOT NULL,
    last_text   TEXT,
    last_edit_at TEXT,
    created_at  TEXT NOT NULL
);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Database:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._connection: sqlite3.Connection | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gw-db")
        self._closed = False

    async def connect(self) -> None:
        await self._run(self._connect_sync)

    def _connect_sync(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self._path, check_same_thread=False, isolation_level=None, timeout=30.0
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.executescript(SCHEMA)
        self._connection = connection

    def _require(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("database is not connected")
        return self._connection

    async def _run(self, fn: Callable[..., T], *args: Any) -> T:
        if self._closed:
            raise RuntimeError("database is closed")
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        def run() -> int:
            return self._require().execute(sql, params).rowcount

        return await self._run(run)

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        def run() -> sqlite3.Row | None:
            return self._require().execute(sql, params).fetchone()

        return await self._run(run)

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        def run() -> list[sqlite3.Row]:
            return self._require().execute(sql, params).fetchall()

        return await self._run(run)

    async def fetch_value(self, sql: str, params: Sequence[Any] = (), default: Any = None) -> Any:
        row = await self.fetch_one(sql, params)
        return row[0] if row is not None else default

    async def transaction(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        def run() -> T:
            connection = self._require()
            connection.execute("BEGIN IMMEDIATE")
            try:
                result = fn(connection)
            except Exception:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
            return result

        return await self._run(run)

    async def close(self) -> None:
        if self._closed:
            return

        def run() -> None:
            if self._connection is not None:
                try:
                    self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    log.debug("wal checkpoint failed on shutdown", exc_info=True)
                self._connection.close()
                self._connection = None

        try:
            await self._run(run)
        finally:
            self._closed = True
            self._executor.shutdown(wait=True)
