"""Maps conversations onto Cursor sessions.

One Telegram user gets one conversation, and a conversation gets one Cursor session, so contexts
cannot bleed between users. The mapping is persisted, so after a Mac reboot the same conversation
reattaches to the same Cursor session rather than starting the user's history over.

Each session is also issued its own MCP token. That token is what lets the MCP server work out
which conversation — and therefore which user — a tool call belongs to, instead of trusting a
``user_id`` argument the model could invent.

The Cursor session ``cwd`` is the per-user directory ``DATA_DIR/user_{tg_id}``: the agent may
only touch files there (enforced again on ACP ``fs/*``). Chat sessions also run in Cursor
``plan`` mode so built-in shell/write stay blocked; MCP tools still work.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from ..agent.base import AgentBackend, AgentContext, AgentError
from ..storage.repositories import ConversationRepository, CursorSession
from ..training.skill import seed_into as seed_trainer_skill

log = logging.getLogger(__name__)


class SessionManager:
    def __init__(
        self,
        conversations: ConversationRepository,
        backend: AgentBackend,
        contexts,
        mcp_server,
        *,
        user_workspace: Callable[[str], Path],
    ) -> None:
        self._conversations = conversations
        self._backend = backend
        self._contexts = contexts
        self._mcp = mcp_server
        self._user_workspace = user_workspace
        self._locks: dict[str, asyncio.Lock] = {}
        # conversation_id -> MCP token, so a token survives for the life of the session.
        self._tokens: dict[str, str] = {}

    def _lock(self, conversation_id: str) -> asyncio.Lock:
        return self._locks.setdefault(conversation_id, asyncio.Lock())

    async def ensure_session(
        self, conversation_id: str, *, user_id: str
    ) -> tuple[CursorSession, bool]:
        """Return a usable Cursor session and whether it was just created.

        The caller needs to know: a brand-new session has never seen the operating instructions,
        while a resumed one still has them in its history.

        If an older session was rooted in the shared ``workspace/`` directory, it is closed and
        replaced — history in Cursor does not migrate; SQLite memory and reminders do.
        """
        target = self._user_workspace(user_id)
        seed_trainer_skill(target)

        async with self._lock(conversation_id):
            record = await self._conversations.find_session_by_workspace(
                conversation_id, str(target)
            )

            if record is None:
                stale = await self._conversations.session_for_conversation(conversation_id)
                if stale is not None and stale.workspace != str(target):
                    log.info(
                        "closing cursor session %s with obsolete workspace %s "
                        "(now %s)",
                        stale.external_id,
                        stale.workspace,
                        target,
                    )
                    await self._conversations.close_session(stale.session_id)

            if record is not None and record.external_id:
                token = self._token_for(conversation_id)
                resumed = await self._resume(record, target, token)
                if resumed:
                    await self._apply_chat_mode(record.external_id)
                    return record, False
                log.info(
                    "cursor session %s could not be resumed; starting a new one",
                    record.external_id,
                )
                await self._conversations.close_session(record.session_id)

            created = await self._create(conversation_id, target)
            await self._apply_chat_mode(created.external_id)
            return created, True

    async def _apply_chat_mode(self, session_id: str) -> None:
        """Telegram chat sessions run in plan mode so built-in shell/write stay blocked.

        MCP tools still work in plan (verified by tools.acp_probe plan-mcp). Coding workspaces
        with writable=true would use agent mode — those are not opened through SessionManager.
        """
        set_mode = getattr(self._backend, "set_mode", None)
        if set_mode is None or not session_id:
            return
        try:
            await set_mode(session_id, "plan")
        except AgentError as exc:
            log.warning("could not set plan mode on session %s: %s", session_id, exc)

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
