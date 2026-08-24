"""Job and upload records.

``jobs.request_id`` carries a UNIQUE constraint, which is where request idempotency is actually
enforced: a Gateway that retries ``assistant.submit`` after a lost response gets the original
``job_id`` back rather than starting a second job.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pa_protocol import new_ulid

from ..database import Database, from_iso, utcnow

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


@dataclass(slots=True)
class Job:
    job_id: str
    request_id: str
    user_id: str
    chat_id: int | None
    message_id: int | None
    kind: str
    status: str
    stage: str | None
    payload: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    created_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Job:
        return cls(
            job_id=row["job_id"],
            request_id=row["request_id"],
            user_id=row["user_id"],
            chat_id=row["chat_id"],
            message_id=row["message_id"],
            kind=row["kind"],
            status=row["status"],
            stage=row["stage"],
            payload=json.loads(row["payload"] or "{}"),
            error_code=row["error_code"],
            created_at=from_iso(row["created_at"]),
        )


class JobRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def create_or_get(
        self,
        *,
        request_id: str,
        user_id: str,
        kind: str,
        chat_id: int | None = None,
        message_id: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> tuple[Job, bool]:
        """Create a job for ``request_id``, or return the existing one.

        The boolean is ``True`` when this was a duplicate submission.
        """
        job_id = new_ulid()
        now = utcnow().isoformat()
        encoded = json.dumps(payload or {}, ensure_ascii=False)

        def run(connection: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
            existing = connection.execute(
                "SELECT * FROM jobs WHERE request_id = ?", (request_id,)
            ).fetchone()
            if existing is not None:
                return existing, True
            connection.execute(
                "INSERT INTO jobs(job_id, request_id, user_id, chat_id, message_id, kind, "
                "status, stage, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'queued', 'queued', ?, ?)",
                (job_id, request_id, user_id, chat_id, message_id, kind, encoded, now),
            )
            created = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return created, False

        row, duplicate = await self._db.transaction(run)
        return Job.from_row(row), duplicate

    async def get(self, job_id: str) -> Job | None:
        row = await self._db.fetch_one("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        return Job.from_row(row) if row else None

    async def mark_running(self, job_id: str) -> None:
        await self._db.execute(
            "UPDATE jobs SET status = 'running', stage = 'agent', started_at = ? "
            "WHERE job_id = ? AND status = 'queued'",
            (utcnow().isoformat(), job_id),
        )

    async def set_stage(self, job_id: str, stage: str) -> None:
        await self._db.execute(
            "UPDATE jobs SET stage = ? WHERE job_id = ? AND status NOT IN "
            "('completed','failed','cancelled')",
            (stage, job_id),
        )

    async def finish(
        self,
        job_id: str,
        status: str,
        *,
        result: str | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> bool:
        """Move a job to a terminal state. Returns ``False`` if it was already terminal."""
        changed = await self._db.execute(
            "UPDATE jobs SET status = ?, stage = ?, result = ?, error_code = ?, "
            "error_detail = ?, finished_at = ? "
            "WHERE job_id = ? AND status NOT IN ('completed','failed','cancelled')",
            (
                status,
                "completed" if status == "completed" else status,
                result,
                error_code,
                (error_detail or "")[:2000] or None,
                utcnow().isoformat(),
                job_id,
            ),
        )
        return changed > 0

    async def counts(self) -> dict[str, int]:
        rows = await self._db.fetch_all(
            "SELECT status, COUNT(*) AS n FROM jobs "
            "WHERE status IN ('queued','running') GROUP BY status"
        )
        counts = {row["status"]: row["n"] for row in rows}
        return {"queued": counts.get("queued", 0), "running": counts.get("running", 0)}

    async def active_for_user(self, user_id: str) -> list[Job]:
        rows = await self._db.fetch_all(
            "SELECT * FROM jobs WHERE user_id = ? AND status IN ('queued','running') "
            "ORDER BY created_at",
            (user_id,),
        )
        return [Job.from_row(row) for row in rows]

    async def recover_orphans(self) -> int:
        """Fail jobs left running by a crash.

        Nothing is executing them after a restart, so leaving them 'running' would make ``/status``
        lie and would strand the user's status message forever.
        """
        return await self._db.execute(
            "UPDATE jobs SET status = 'failed', error_code = 'interrupted', "
            "error_detail = 'core restarted while job was running', finished_at = ? "
            "WHERE status IN ('queued','running')",
            (utcnow().isoformat(),),
        )


@dataclass(slots=True)
class Upload:
    upload_id: str
    request_id: str
    user_id: str
    chat_id: int | None
    message_id: int | None
    filename: str
    content_type: str | None
    declared_size: int
    received_size: int
    duration_seconds: float | None
    purpose: str
    temp_path: Path
    status: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Upload:
        return cls(
            upload_id=row["upload_id"],
            request_id=row["request_id"],
            user_id=row["user_id"],
            chat_id=row["chat_id"],
            message_id=row["message_id"],
            filename=row["filename"],
            content_type=row["content_type"],
            declared_size=row["declared_size"],
            received_size=row["received_size"],
            duration_seconds=row["duration_seconds"],
            purpose=row["purpose"],
            temp_path=Path(row["temp_path"]),
            status=row["status"],
        )


class UploadRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(
        self,
        *,
        request_id: str,
        user_id: str,
        filename: str,
        content_type: str | None,
        declared_size: int,
        temp_path: Path,
        chat_id: int | None = None,
        message_id: int | None = None,
        duration_seconds: float | None = None,
        purpose: str = "assistant",
    ) -> Upload:
        upload_id = new_ulid()
        now = utcnow().isoformat()
        await self._db.execute(
            "INSERT INTO uploads(upload_id, request_id, user_id, chat_id, message_id, filename, "
            "content_type, declared_size, received_size, duration_seconds, purpose, temp_path, "
            "status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, 'open', ?, ?)",
            (
                upload_id, request_id, user_id, chat_id, message_id, filename, content_type,
                declared_size, duration_seconds, purpose, str(temp_path), now, now,
            ),
        )
        return Upload(
            upload_id, request_id, user_id, chat_id, message_id, filename, content_type,
            declared_size, 0, duration_seconds, purpose, temp_path, "open",
        )

    async def get(self, upload_id: str) -> Upload | None:
        row = await self._db.fetch_one("SELECT * FROM uploads WHERE upload_id = ?", (upload_id,))
        return Upload.from_row(row) if row else None

    async def find_open_by_request(self, request_id: str) -> Upload | None:
        row = await self._db.fetch_one(
            "SELECT * FROM uploads WHERE request_id = ? AND status = 'open' "
            "ORDER BY created_at DESC LIMIT 1",
            (request_id,),
        )
        return Upload.from_row(row) if row else None

    async def record_progress(self, upload_id: str, received_size: int) -> None:
        await self._db.execute(
            "UPDATE uploads SET received_size = ?, updated_at = ? WHERE upload_id = ?",
            (received_size, utcnow().isoformat(), upload_id),
        )

    async def set_status(self, upload_id: str, status: str) -> None:
        await self._db.execute(
            "UPDATE uploads SET status = ?, updated_at = ? WHERE upload_id = ?",
            (status, utcnow().isoformat(), upload_id),
        )

    async def stale_open(self, older_than_iso: str) -> list[Upload]:
        rows = await self._db.fetch_all(
            "SELECT * FROM uploads WHERE status = 'open' AND updated_at < ?", (older_than_iso,)
        )
        return [Upload.from_row(row) for row in rows]


class TranscriptionMetadataRepository:
    """Metadata only.

    Transcript text is never persisted here — logging policy forbids storing full transcripts, and
    the recording itself is deleted once processed.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def record(
        self,
        *,
        user_id: str,
        job_id: str | None,
        filename: str | None,
        language: str | None,
        duration: float | None,
        segment_count: int,
        char_count: int,
        model: str,
        elapsed_seconds: float,
    ) -> str:
        transcription_id = new_ulid()
        await self._db.execute(
            "INSERT INTO transcription_metadata(transcription_id, job_id, user_id, filename, "
            "language, duration, segment_count, char_count, model, elapsed_seconds, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                transcription_id, job_id, user_id, filename, language, duration,
                segment_count, char_count, model, elapsed_seconds, utcnow().isoformat(),
            ),
        )
        return transcription_id
