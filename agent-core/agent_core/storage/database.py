"""SQLite access for the Agent Core.

Every statement runs in a dedicated worker thread. SQLite calls are blocking, and a slow write
must never stall the WebSocket that carries reminders and replies. A single connection on a single
thread also sidesteps SQLite's threading rules entirely and gives writes a natural serialisation
point.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from .schema import MIGRATIONS

log = logging.getLogger(__name__)

T = TypeVar("T")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(value: datetime | None) -> str | None:
    """Serialise to UTC ISO-8601. Naive datetimes are rejected rather than guessed at."""
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("naive datetime reached storage; all times must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _split_statements(script: str) -> list[str]:
    """Split a migration script into individual statements.

    Uses SQLite's own completeness check rather than splitting on semicolons, so a semicolon
    inside a string literal or trigger body does not produce a truncated statement.
    """
    statements: list[str] = []
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            candidate = buffer.strip()
            if candidate:
                statements.append(candidate)
            buffer = ""
    if buffer.strip():
        raise ValueError(f"migration ends with an incomplete statement: {buffer.strip()[:80]!r}")
    return statements


class Database:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._connection: sqlite3.Connection | None = None
        # One thread: serialises access and keeps the connection thread-affine.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="core-db")
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    async def connect(self) -> None:
        await self._run(self._connect_sync)
        await self._run(self._migrate_sync)

    def _connect_sync(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self._path,
            check_same_thread=False,
            isolation_level=None,  # explicit transactions
            timeout=30.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        self._connection = connection

    def _migrate_sync(self) -> None:
        connection = self._require()
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        applied = {row["name"] for row in connection.execute("SELECT name FROM schema_migrations")}
        for name, statements in MIGRATIONS:
            if name in applied:
                continue
            log.info("applying migration %s", name)
            # Statements are executed one at a time rather than via executescript, which would
            # implicitly commit and leave the migration half-applied on failure.
            connection.execute("BEGIN")
            try:
                for statement in _split_statements(statements):
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(name, applied_at) VALUES (?, ?)",
                    (name, utcnow().isoformat()),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def _require(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("database is not connected")
        return self._connection

    async def _run(self, fn: Callable[..., T], *args: Any) -> T:
        if self._closed:
            raise RuntimeError("database is closed")
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    # ---- query helpers ------------------------------------------------

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        def run() -> int:
            cursor = self._require().execute(sql, params)
            return cursor.rowcount

        return await self._run(run)

    async def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        materialised = list(rows)

        def run() -> None:
            self._require().executemany(sql, materialised)

        await self._run(run)

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
        """Run ``fn`` inside a transaction on the database thread.

        Multi-statement work that must be atomic — such as inserting a durable event and
        allocating its sequence number — goes through here rather than issuing separate calls.
        """

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
                # A checkpoint on the way out keeps the WAL from growing without bound across
                # restarts.
                try:
                    self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    log.debug("wal checkpoint on shutdown failed", exc_info=True)
                self._connection.close()
                self._connection = None

        try:
            await self._run(run)
        finally:
            self._closed = True
            self._executor.shutdown(wait=True)


@contextmanager
def sync_connection(path: Path):
    """Open a short-lived synchronous connection. For CLI utilities and tests only."""
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()
