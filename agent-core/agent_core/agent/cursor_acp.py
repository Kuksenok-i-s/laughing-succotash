"""``AgentBackend`` implementation on top of Cursor's ACP server."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from .acp_client import AcpClient, AcpProcessError, option_of_kind
from .base import (
    AgentContext,
    AgentError,
    AgentResponse,
    AgentUnavailable,
    ProgressCallback,
    ToolInvocation,
)

log = logging.getLogger(__name__)

# tool_call.kind values seen from the real agent, mapped to the job stages the Gateway renders.
_KIND_TO_STAGE = {
    "read": "executing_tool",
    "edit": "executing_tool",
    "execute": "executing_tool",
    "search": "executing_tool",
    "other": "executing_tool",
}


class _Turn:
    """Accumulates the streamed updates for one ``session/prompt``."""

    def __init__(self, on_progress: ProgressCallback | None) -> None:
        self.chunks: list[str] = []
        self.tool_calls: dict[str, ToolInvocation] = {}
        self.on_progress = on_progress
        # Replayed history from session/load must not be mistaken for this turn's reply.
        self.accepting = True

    @property
    def text(self) -> str:
        return "".join(self.chunks).strip()


class CursorACPBackend:
    def __init__(
        self,
        binary: str = "cursor-agent",
        *,
        default_workspace: Path,
        mcp_servers: list[dict[str, Any]] | None = None,
        model: str | None = None,
        startup_timeout: float = 60.0,
        prompt_timeout: float = 1800.0,
        permission_resolver=None,
    ) -> None:
        self._binary = binary
        self._default_workspace = default_workspace
        self._mcp_servers = mcp_servers or []
        self._model = model
        self._startup_timeout = startup_timeout
        self._prompt_timeout = prompt_timeout
        self._permission_resolver = permission_resolver

        self._client: AcpClient | None = None
        self._state = "stopped"
        self._turns: dict[str, _Turn] = {}
        # Sessions this process has created or loaded. A session from a previous run must be
        # loaded once before it will accept a prompt.
        self._live_sessions: set[str] = set()
        self._start_lock = asyncio.Lock()
        # Cursor keys permission answers by session, and a turn is serialised per conversation
        # anyway, so a single map is enough to correlate a permission request with its turn.
        self._permission_context: dict[str, AgentContext] = {}

    @property
    def name(self) -> str:
        return "acp"

    @property
    def state(self) -> str:
        return self._state

    # ---- lifecycle -----------------------------------------------------

    async def start(self) -> None:
        async with self._start_lock:
            if self._client is not None and self._client.running:
                return
            self._state = "starting"
            client = AcpClient(
                self._binary,
                cwd=self._default_workspace,
                request_timeout=self._prompt_timeout,
            )
            client.on_update = self._on_update
            client.on_permission = self._on_permission
            try:
                await client.start()
                await asyncio.wait_for(client.initialize(), self._startup_timeout)
            except (AcpProcessError, asyncio.TimeoutError, OSError) as exc:
                self._state = "unavailable"
                await client.close()
                raise AgentUnavailable(f"cursor-agent acp did not start: {exc}") from exc

            self._client = client
            self._live_sessions.clear()
            self._state = "ready"
            log.info(
                "cursor acp ready (loadSession=%s, mcp=%s)",
                client.agent_capabilities.get("loadSession"),
                client.agent_capabilities.get("mcpCapabilities"),
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._state = "stopped"

    def _require(self) -> AcpClient:
        if self._client is None or not self._client.running:
            raise AgentUnavailable("cursor acp is not running")
        return self._client

    # ---- sessions --------------------------------------------------------

    async def create_session(
        self,
        *,
        workspace: Path,
        context: AgentContext | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> str:
        await self._ensure_started()
        client = self._require()
        try:
            result = await client.call(
                "session/new",
                {
                    "cwd": str(workspace),
                    # Per-session MCP entries: each carries the token that tells the MCP server
                    # which conversation a tool call came from.
                    "mcpServers": self._mcp_servers if mcp_servers is None else mcp_servers,
                },
                timeout=self._startup_timeout,
            )
        except AcpProcessError as exc:
            raise AgentError(f"could not create cursor session: {exc}") from exc

        session_id = result.get("sessionId")
        if not session_id:
            raise AgentError("cursor session/new returned no sessionId")
        self._live_sessions.add(session_id)
        client.bind_session_root(session_id, workspace)

        if self._model:
            await self._select_model(session_id)
        return session_id

    async def _select_model(self, session_id: str) -> None:
        """Best-effort model pin.

        A configured model that this account cannot use should degrade to Auto rather than break
        the assistant entirely.
        """
        try:
            await self._require().call(
                "session/set_model", {"sessionId": session_id, "modelId": self._model}, timeout=30
            )
        except AcpProcessError as exc:
            log.warning("could not select model %s: %s; using Auto", self._model, exc)

    async def resume_session(
        self,
        session_id: str,
        workspace: Path,
        *,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Reattach to a session created before this process started.

        ``session/load`` replays the whole conversation as notifications, so replay is suppressed
        while it runs to keep old assistant text out of the next reply.
        """
        await self._ensure_started()
        if session_id in self._live_sessions:
            client = self._require()
            client.bind_session_root(session_id, workspace)
            return True
        client = self._require()
        if not client.agent_capabilities.get("loadSession"):
            return False
        try:
            await client.call(
                "session/load",
                {
                    "sessionId": session_id,
                    "cwd": str(workspace),
                    "mcpServers": self._mcp_servers if mcp_servers is None else mcp_servers,
                },
                timeout=self._startup_timeout,
            )
        except AcpProcessError as exc:
            log.info("could not resume cursor session %s: %s", session_id, exc)
            return False
        self._live_sessions.add(session_id)
        client.bind_session_root(session_id, workspace)
        return True

    async def set_mode(self, session_id: str, mode: str) -> None:
        try:
            await self._require().call(
                "session/set_mode", {"sessionId": session_id, "modeId": mode}, timeout=30
            )
        except AcpProcessError as exc:
            raise AgentError(f"could not set mode {mode}: {exc}") from exc

    # ---- prompting --------------------------------------------------------

    async def send_message(
        self,
        session_id: str,
        message: str,
        context: AgentContext | None = None,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> AgentResponse:
        await self._ensure_started()
        client = self._require()

        turn = _Turn(on_progress)
        self._turns[session_id] = turn
        if context is not None:
            self._permission_context[session_id] = context

        try:
            result = await client.call(
                "session/prompt",
                {"sessionId": session_id, "prompt": [{"type": "text", "text": message}]},
                timeout=self._prompt_timeout,
            )
        except AcpProcessError as exc:
            self._state = "degraded" if client.running else "unavailable"
            raise AgentError(f"cursor prompt failed: {exc}", retryable=not client.running) from exc
        except asyncio.TimeoutError as exc:
            # Leave the session usable: cancel the turn rather than abandoning it mid-flight.
            await self.cancel(session_id)
            raise AgentError("cursor prompt timed out") from exc
        finally:
            self._turns.pop(session_id, None)
            self._permission_context.pop(session_id, None)

        return AgentResponse(
            text=turn.text,
            stop_reason=result.get("stopReason", "end_turn"),
            tool_calls=list(turn.tool_calls.values()),
            session_id=session_id,
        )

    async def cancel(self, session_id: str) -> None:
        client = self._client
        if client is None or not client.running:
            return
        try:
            await client.notify("session/cancel", {"sessionId": session_id})
        except Exception:
            log.debug("cancel notification failed", exc_info=True)

    # ---- streaming updates -------------------------------------------------

    async def _on_update(self, session_id: str, update: dict[str, Any]) -> None:
        turn = self._turns.get(session_id)
        if turn is None or not turn.accepting:
            return

        kind = update.get("sessionUpdate")

        if kind == "agent_message_chunk":
            text = (update.get("content") or {}).get("text")
            if text:
                turn.chunks.append(text)
            return

        # agent_thought_chunk is reasoning. It is deliberately dropped: it is not the answer, and
        # forwarding it to Telegram would leak half-formed conclusions.
        if kind == "agent_thought_chunk":
            return

        if kind == "tool_call":
            call = ToolInvocation(
                tool_call_id=update.get("toolCallId", ""),
                title=update.get("title", ""),
                kind=update.get("kind", "other"),
                status=update.get("status", "pending"),
            )
            turn.tool_calls[call.tool_call_id] = call
            if turn.on_progress is not None:
                stage = _KIND_TO_STAGE.get(call.kind, "executing_tool")
                await turn.on_progress(stage, call.title or None)
            return

        if kind == "tool_call_update":
            existing = turn.tool_calls.get(update.get("toolCallId", ""))
            if existing is not None:
                existing.status = update.get("status", existing.status)
            return

    async def _on_permission(self, params: dict[str, Any]) -> str | None:
        """Coarse first-layer permission gate.

        The authoritative decision happens inside the MCP server, which sees exact tool names and
        validated arguments. Here we only distinguish "our own MCP server" from anything else,
        because this payload identifies the tool by a display title (ADR 7).
        """
        options = params.get("options") or []
        tool_call = params.get("toolCall") or {}
        title = tool_call.get("title", "")

        allow = option_of_kind(options, "allow_once")
        reject = option_of_kind(options, "reject_once")

        if self._permission_resolver is not None:
            decision = await self._permission_resolver(params)
            if decision is False:
                return reject
            if decision is True:
                return allow

        log.debug("auto-allowing tool call: %s", title)
        return allow

    async def _ensure_started(self) -> None:
        if self._client is None or not self._client.running:
            await self.start()
