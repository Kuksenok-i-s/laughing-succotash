"""Maps conversations onto Cursor sessions.

One Telegram user gets one conversation, and a conversation gets one Cursor session, so contexts
cannot bleed between users. The mapping is persisted, so after a Mac reboot the same conversation
reattaches to the same Cursor session rather than starting the user's history over.

Each session is also issued its own MCP token. That token is what lets the MCP server work out
which conversation — and therefore which user — a tool call belongs to, instead of trusting a
``user_id`` argument the model could invent.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..agent.base import AgentBackend, AgentContext, AgentError
from ..storage.repositories import ConversationRepository, CursorSession

log = logging.getLogger(__name__)


class SessionManager:
    def __init__(
        self,
        conversations: ConversationRepository,
        backend: AgentBackend,
        contexts,
        mcp_server,
        *,
        default_workspace: Path,
    ) -> None:
        self._conversations = conversations
        self._backend = backend
        self._contexts = contexts
        self._mcp = mcp_server
        self._default_workspace = default_workspace
        self._locks: dict[str, asyncio.Lock] = {}
        # conversation_id -> MCP token, so a token survives for the life of the session.
        self._tokens: dict[str, str] = {}

    def _lock(self, conversation_id: str) -> asyncio.Lock:
        return self._locks.setdefault(conversation_id, asyncio.Lock())

    async def ensure_session(
        self, conversation_id: str, *, workspace: Path | None = None
    ) -> tuple[CursorSession, bool]:
        """Return a usable Cursor session and whether it was just created.

        The caller needs to know: a brand-new session has never seen the operating instructions,
        while a resumed one still has them in its history.
        """
        target = workspace or self._default_workspace

        async with self._lock(conversation_id):
            record = (
                await self._conversations.find_session_by_workspace(conversation_id, str(target))
                if workspace is not None
                else await self._conversations.session_for_conversation(conversation_id)
            )

            if record is not None and record.external_id:
                token = self._token_for(conversation_id)
                resumed = await self._resume(record, target, token)
                if resumed:
                    return record, False
                # The session is gone from Cursor's side (a reinstall, a cleared cache). Close the
                # record and open a fresh one rather than failing the user's message.
                log.info("cursor session %s could not be resumed; starting a new one",
                         record.external_id)
                await self._conversations.close_session(record.session_id)

            return await self._create(conversation_id, target), True

    async def _resume(self, record: CursorSession, workspace: Path, token: str) -> bool:
        resume = getattr(self._backend, "resume_session", None)
        if resume is None:
            return True
        try:
            return await resume(
                record.external_id, workspace, mcp_servers=self._mcp_entries(token)
            )
        except AgentError as exc:
            log.info("resume failed for %s: %s", record.external_id, exc)
            return False

    async def _create(self, conversation_id: str, workspace: Path) -> CursorSession:
        token = self._token_for(conversation_id)
        external_id = await self._backend.create_session(
            workspace=workspace, mcp_servers=self._mcp_entries(token)
        )
        record = await self._conversations.create_session(
            conversation_id,
            backend=self._backend.name,
            workspace=str(workspace),
            external_id=external_id,
        )
        log.info(
            "cursor session %s created for conversation %s (workspace=%s)",
            external_id, conversation_id, workspace,
        )
        return record

    def _token_for(self, conversation_id: str) -> str:
        token = self._tokens.get(conversation_id)
        if token is None:
            token = self._contexts.issue_token(conversation_id)
            self._tokens[conversation_id] = token
        else:
            self._contexts.bind_token(token, conversation_id)
        return token

    def _mcp_entries(self, token: str) -> list[dict]:
        return [self._mcp.session_entry(token)] if self._mcp is not None else []

    def begin_turn(self, conversation_id: str, context) -> None:
        self._contexts.set_current(conversation_id, context)

    def end_turn(self, conversation_id: str) -> None:
        self._contexts.clear_current(conversation_id)

    async def reset(self, user_id: str) -> str:
        """Start a fresh conversation. Old memory and reminders survive; the dialogue does not."""
        previous = await self._conversations.active_conversation(user_id)
        if previous is not None:
            session = await self._conversations.session_for_conversation(
                previous.conversation_id
            )
            if session is not None:
                await self._conversations.close_session(session.session_id)
            self._tokens.pop(previous.conversation_id, None)
            self._locks.pop(previous.conversation_id, None)

        conversation = await self._conversations.create_conversation(user_id)
        log.info("conversation reset for %s -> %s", user_id, conversation.conversation_id)
        return conversation.conversation_id

    async def context_for(
        self, user_id: str, conversation_id: str, **fields
    ) -> AgentContext:
        return AgentContext(
            user_id=user_id,
            conversation_id=conversation_id,
            timezone=await self._conversations.timezone_for(user_id),
            **fields,
        )
