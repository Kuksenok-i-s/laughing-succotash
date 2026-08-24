"""Users, conversations and their Cursor sessions.

An external identity (``tg:123456789``) maps to a conversation, which maps to a Cursor session.
The namespace prefix is mandatory so a future non-Telegram front end cannot collide with a
Telegram user ID.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pa_protocol import new_ulid

from ..database import Database, from_iso, utcnow


@dataclass(slots=True)
class User:
    user_id: str
    timezone: str | None
    display_name: str | None


@dataclass(slots=True)
class Conversation:
    conversation_id: str
    user_id: str
    status: str
    title: str | None
    created_at: datetime | None


@dataclass(slots=True)
class CursorSession:
    session_id: str
    conversation_id: str
    backend: str
    external_id: str | None
    workspace: str
    mode: str
    status: str


class ConversationRepository:
    def __init__(self, db: Database, default_timezone: str) -> None:
        self._db = db
        self._default_timezone = default_timezone

    # ---- users --------------------------------------------------------

    async def ensure_user(self, user_id: str, display_name: str | None = None) -> User:
        now = utcnow().isoformat()
        await self._db.execute(
            "INSERT INTO users(user_id, display_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET "
            "display_name = COALESCE(excluded.display_name, users.display_name), "
            "updated_at = excluded.updated_at",
            (user_id, display_name, now, now),
        )
        row = await self._db.fetch_one("SELECT * FROM users WHERE user_id = ?", (user_id,))
        assert row is not None
        return User(row["user_id"], row["timezone"], row["display_name"])

    async def remember_chat(self, user_id: str, chat_id: int | None) -> None:
        """Record where to reach this user for unsolicited messages."""
        if chat_id is None:
            return
        await self._db.execute(
            "UPDATE users SET last_chat_id = ?, updated_at = ? WHERE user_id = ?",
            (chat_id, utcnow().isoformat(), user_id),
        )

    async def chat_for(self, user_id: str) -> int | None:
        value = await self._db.fetch_value(
            "SELECT last_chat_id FROM users WHERE user_id = ?", (user_id,)
        )
        return int(value) if value is not None else None

    async def set_timezone(self, user_id: str, tz_name: str) -> None:
        ZoneInfo(tz_name)  # reject an invalid zone before persisting it
        await self._db.execute(
            "UPDATE users SET timezone = ?, updated_at = ? WHERE user_id = ?",
            (tz_name, utcnow().isoformat(), user_id),
        )

    async def timezone_for(self, user_id: str) -> tzinfo:
        """Resolve the user's timezone, falling back to the configured default.

        Relative expressions like "завтра" or "через два часа" are meaningless without this, so it
        is resolved once per turn and passed explicitly into the agent context.
        """
        name = await self._db.fetch_value(
            "SELECT timezone FROM users WHERE user_id = ?", (user_id,)
        )
        for candidate in (name, self._default_timezone):
            if not candidate:
                continue
            try:
                return ZoneInfo(candidate)
            except ZoneInfoNotFoundError:
                continue
        return ZoneInfo("UTC")

    # ---- conversations ------------------------------------------------

    async def active_conversation(self, user_id: str) -> Conversation | None:
        row = await self._db.fetch_one(
            "SELECT * FROM conversations WHERE user_id = ? AND status = 'active' "
            "ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        )
        return self._conversation(row) if row else None

    async def get_or_create_conversation(self, user_id: str) -> Conversation:
        existing = await self.active_conversation(user_id)
        if existing is not None:
            return existing
        return await self.create_conversation(user_id)

    async def create_conversation(self, user_id: str) -> Conversation:
        await self.ensure_user(user_id)
        conversation_id = new_ulid()
        now = utcnow().isoformat()

        def run(connection: sqlite3.Connection) -> None:
            # Archive prior conversations first: /new must not leave two active ones behind, or
            # the next message could land in either.
            connection.execute(
                "UPDATE conversations SET status = 'archived', updated_at = ? "
                "WHERE user_id = ? AND status = 'active'",
                (now, user_id),
            )
            connection.execute(
                "INSERT INTO conversations(conversation_id, user_id, status, created_at, updated_at) "
                "VALUES (?, ?, 'active', ?, ?)",
                (conversation_id, user_id, now, now),
            )

        await self._db.transaction(run)
        return Conversation(conversation_id, user_id, "active", None, utcnow())

    async def touch_conversation(self, conversation_id: str) -> None:
        await self._db.execute(
            "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
            (utcnow().isoformat(), conversation_id),
        )

    async def set_title(self, conversation_id: str, title: str) -> None:
        await self._db.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ?",
            (title[:200], utcnow().isoformat(), conversation_id),
        )

    # ---- cursor sessions ----------------------------------------------

    async def session_for_conversation(self, conversation_id: str) -> CursorSession | None:
        row = await self._db.fetch_one(
            "SELECT * FROM cursor_sessions WHERE conversation_id = ? AND status = 'active' "
            "ORDER BY updated_at DESC LIMIT 1",
            (conversation_id,),
        )
        return self._session(row) if row else None

    async def create_session(
        self,
        conversation_id: str,
        *,
        backend: str,
        workspace: str,
        external_id: str | None = None,
        mode: str = "agent",
    ) -> CursorSession:
        session_id = new_ulid()
        now = utcnow().isoformat()
        await self._db.execute(
            "INSERT INTO cursor_sessions"
            "(session_id, conversation_id, backend, external_id, workspace, mode, status, "
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)",
            (session_id, conversation_id, backend, external_id, workspace, mode, now, now),
        )
        return CursorSession(
            session_id, conversation_id, backend, external_id, workspace, mode, "active"
        )

    async def attach_external_id(self, session_id: str, external_id: str) -> None:
        await self._db.execute(
            "UPDATE cursor_sessions SET external_id = ?, updated_at = ? WHERE session_id = ?",
            (external_id, utcnow().isoformat(), session_id),
        )

    async def set_session_mode(self, session_id: str, mode: str) -> None:
        await self._db.execute(
            "UPDATE cursor_sessions SET mode = ?, updated_at = ? WHERE session_id = ?",
            (mode, utcnow().isoformat(), session_id),
        )

    async def close_session(self, session_id: str) -> None:
        await self._db.execute(
            "UPDATE cursor_sessions SET status = 'closed', updated_at = ? WHERE session_id = ?",
            (utcnow().isoformat(), session_id),
        )

    async def find_session_by_workspace(
        self, conversation_id: str, workspace: str
    ) -> CursorSession | None:
        """Coding work reuses one session per project so follow-ups continue the same context."""
        row = await self._db.fetch_one(
            "SELECT * FROM cursor_sessions WHERE conversation_id = ? AND workspace = ? "
            "AND status = 'active' ORDER BY updated_at DESC LIMIT 1",
            (conversation_id, workspace),
        )
        return self._session(row) if row else None

    # ---- mapping ------------------------------------------------------

    @staticmethod
    def _conversation(row: sqlite3.Row) -> Conversation:
        return Conversation(
            conversation_id=row["conversation_id"],
            user_id=row["user_id"],
            status=row["status"],
            title=row["title"],
            created_at=from_iso(row["created_at"]),
        )

    @staticmethod
    def _session(row: sqlite3.Row) -> CursorSession:
        return CursorSession(
            session_id=row["session_id"],
            conversation_id=row["conversation_id"],
            backend=row["backend"],
            external_id=row["external_id"],
            workspace=row["workspace"],
            mode=row["mode"],
            status=row["status"],
        )
