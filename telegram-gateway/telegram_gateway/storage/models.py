"""Gateway transport-state repositories."""

from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .database import Database, utcnow


@dataclass(slots=True)
class PendingRequest:
    request_id: str
    user_id: str
    chat_id: int
    message_id: int
    kind: str
    payload: dict[str, Any]
    attempts: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> PendingRequest:
        return cls(
            request_id=row["request_id"],
            user_id=row["user_id"],
            chat_id=row["chat_id"],
            message_id=row["message_id"],
            kind=row["kind"],
            payload=json.loads(row["payload"]),
            attempts=row["attempts"],
        )


@dataclass(slots=True)
class PendingUpload:
    request_id: str
    user_id: str
    chat_id: int
    message_id: int
    file_path: Path
    filename: str
    content_type: str | None
    size: int
    duration_seconds: float | None
    purpose: str
    sha256: str | None
    attempts: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> PendingUpload:
        return cls(
            request_id=row["request_id"],
            user_id=row["user_id"],
            chat_id=row["chat_id"],
            message_id=row["message_id"],
            file_path=Path(row["file_path"]),
            filename=row["filename"],
            content_type=row["content_type"],
            size=row["size"],
            duration_seconds=row["duration_seconds"],
            purpose=row["purpose"],
            sha256=row["sha256"],
            attempts=row["attempts"],
        )


class GatewayStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ---- inbound requests ---------------------------------------------

    async def save_request(
        self, *, request_id: str, user_id: str, chat_id: int, message_id: int,
        kind: str, payload: dict[str, Any],
    ) -> None:
        now = utcnow().isoformat()
        await self._db.execute(
            "INSERT INTO pending_requests(request_id, user_id, chat_id, message_id, kind, "
            "payload, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(request_id) DO NOTHING",
            (request_id, user_id, chat_id, message_id, kind,
             json.dumps(payload, ensure_ascii=False), now, now),
        )

    async def mark_request_submitted(self, request_id: str, job_id: str | None) -> None:
        await self._db.execute(
            "UPDATE pending_requests SET status = 'submitted', job_id = ?, updated_at = ? "
            "WHERE request_id = ?",
            (job_id, utcnow().isoformat(), request_id),
        )

    async def mark_request_attempt_failed(self, request_id: str, error: str) -> None:
        await self._db.execute(
            "UPDATE pending_requests SET attempts = attempts + 1, last_error = ?, updated_at = ? "
            "WHERE request_id = ?",
            (error[:400], utcnow().isoformat(), request_id),
        )

    async def pending_requests(self, limit: int = 100) -> list[PendingRequest]:
        rows = await self._db.fetch_all(
            "SELECT * FROM pending_requests WHERE status = 'pending' ORDER BY created_at LIMIT ?",
            (limit,),
        )
        return [PendingRequest.from_row(row) for row in rows]

    async def pending_request_count(self) -> int:
        return int(
            await self._db.fetch_value(
                "SELECT COUNT(*) FROM pending_requests WHERE status = 'pending'"
            ) or 0
        )

    # ---- uploads -------------------------------------------------------

    async def save_upload(
        self, *, request_id: str, user_id: str, chat_id: int, message_id: int,
        file_path: Path, filename: str, content_type: str | None, size: int,
        sha256: str, duration_seconds: float | None = None, purpose: str = "assistant",
    ) -> None:
        now = utcnow().isoformat()
        await self._db.execute(
            "INSERT INTO pending_uploads(request_id, user_id, chat_id, message_id, file_path, "
            "filename, content_type, size, duration_seconds, purpose, sha256, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(request_id) DO NOTHING",
            (request_id, user_id, chat_id, message_id, str(file_path), filename, content_type,
             size, duration_seconds, purpose, sha256, now, now),
        )

    async def pending_uploads(self, limit: int = 20) -> list[PendingUpload]:
        rows = await self._db.fetch_all(
            "SELECT * FROM pending_uploads WHERE status = 'pending' ORDER BY created_at LIMIT ?",
            (limit,),
        )
        return [PendingUpload.from_row(row) for row in rows]

    async def set_upload_status(self, request_id: str, status: str) -> None:
        await self._db.execute(
            "UPDATE pending_uploads SET status = ?, updated_at = ? WHERE request_id = ?",
            (status, utcnow().isoformat(), request_id),
        )

    async def mark_upload_attempt(self, request_id: str) -> int:
        await self._db.execute(
            "UPDATE pending_uploads SET attempts = attempts + 1, updated_at = ? "
            "WHERE request_id = ?",
            (utcnow().isoformat(), request_id),
        )
        return int(
            await self._db.fetch_value(
                "SELECT attempts FROM pending_uploads WHERE request_id = ?", (request_id,)
            ) or 0
        )

    async def get_upload(self, request_id: str) -> PendingUpload | None:
        row = await self._db.fetch_one(
            "SELECT * FROM pending_uploads WHERE request_id = ?", (request_id,)
        )
        return PendingUpload.from_row(row) if row else None

    # ---- outbound delivery dedup ---------------------------------------

    async def claim_delivery(self, delivery_id: str, method: str) -> tuple[bool, int | None]:
        """Reserve a delivery id.

        Returns ``(is_new, existing_message_id)``. A replayed event finds its row and is answered
        from it rather than producing a second Telegram message.
        """

        def run(connection: sqlite3.Connection) -> tuple[bool, int | None]:
            row = connection.execute(
                "SELECT message_id, status FROM outbound_delivery WHERE delivery_id = ?",
                (delivery_id,),
            ).fetchone()
            if row is not None:
                return False, row["message_id"]
            connection.execute(
                "INSERT INTO outbound_delivery(delivery_id, method, status, created_at) "
                "VALUES (?, ?, 'in_progress', ?)",
                (delivery_id, method, utcnow().isoformat()),
            )
            return True, None

        return await self._db.transaction(run)

    async def complete_delivery(
        self, delivery_id: str, *, chat_id: int | None, message_id: int | None
    ) -> None:
        await self._db.execute(
            "UPDATE outbound_delivery SET status = 'done', chat_id = ?, message_id = ? "
            "WHERE delivery_id = ?",
            (chat_id, message_id, delivery_id),
        )

    async def fail_delivery(self, delivery_id: str, error: str) -> None:
        """Release a failed delivery so a Core retry is allowed to try again."""
        await self._db.execute(
            "DELETE FROM outbound_delivery WHERE delivery_id = ? AND status = 'in_progress'",
            (delivery_id,),
        )
        await self._db.execute(
            "INSERT INTO outbound_delivery(delivery_id, method, status, error, created_at) "
            "VALUES (?, 'unknown', 'failed', ?, ?) ON CONFLICT(delivery_id) DO NOTHING",
            (f"{delivery_id}:failed", error[:400], utcnow().isoformat()),
        )

    async def release_delivery(self, delivery_id: str) -> None:
        await self._db.execute(
            "DELETE FROM outbound_delivery WHERE delivery_id = ? AND status = 'in_progress'",
            (delivery_id,),
        )

    # ---- confirmations --------------------------------------------------

    async def create_confirmation_tokens(
        self, *, action_id: str, user_id: str, chat_id: int, choices: list[str]
    ) -> dict[str, str]:
        """Mint one opaque token per button.

        Telegram callback data is capped at 64 bytes and is attacker-visible, so it carries a
        random token rather than the action id or any payload.
        """
        tokens = {choice: secrets.token_urlsafe(16) for choice in choices}
        now = utcnow().isoformat()

        def run(connection: sqlite3.Connection) -> None:
            for choice, token in tokens.items():
                connection.execute(
                    "INSERT INTO confirmations(token, action_id, user_id, chat_id, choice, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (token, action_id, user_id, chat_id, choice, now),
                )

        await self._db.transaction(run)
        return tokens

    async def resolve_confirmation_token(self, token: str) -> sqlite3.Row | None:
        return await self._db.fetch_one("SELECT * FROM confirmations WHERE token = ?", (token,))

    async def mark_confirmation_used(self, action_id: str) -> None:
        await self._db.execute(
            "UPDATE confirmations SET status = 'used' WHERE action_id = ?", (action_id,)
        )

    async def set_confirmation_message(self, action_id: str, message_id: int) -> None:
        await self._db.execute(
            "UPDATE confirmations SET message_id = ? WHERE action_id = ?", (message_id, action_id)
        )

    # ---- job status messages ---------------------------------------------

    async def get_status_message(self, job_id: str) -> sqlite3.Row | None:
        return await self._db.fetch_one(
            "SELECT * FROM job_status_messages WHERE job_id = ?", (job_id,)
        )

    async def set_status_message(
        self, job_id: str, chat_id: int, message_id: int, text: str
    ) -> None:
        now = utcnow().isoformat()
        await self._db.execute(
            "INSERT INTO job_status_messages(job_id, chat_id, message_id, last_text, "
            "last_edit_at, created_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(job_id) DO UPDATE SET message_id = excluded.message_id, "
            "last_text = excluded.last_text, last_edit_at = excluded.last_edit_at",
            (job_id, chat_id, message_id, text, now, now),
        )

    async def touch_status_message(self, job_id: str, text: str) -> None:
        await self._db.execute(
            "UPDATE job_status_messages SET last_text = ?, last_edit_at = ? WHERE job_id = ?",
            (text, utcnow().isoformat(), job_id),
        )

    async def clear_status_message(self, job_id: str) -> None:
        await self._db.execute("DELETE FROM job_status_messages WHERE job_id = ?", (job_id,))

    # ---- sequence state ---------------------------------------------------

    async def get_seq(self, key: str, default: int = 0) -> int:
        value = await self._db.fetch_value(
            "SELECT value FROM rpc_sequence_state WHERE key = ?", (key,)
        )
        return int(value) if value is not None else default

    async def set_seq(self, key: str, value: int) -> None:
        await self._db.execute(
            "INSERT INTO rpc_sequence_state(key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, str(value), utcnow().isoformat()),
        )
