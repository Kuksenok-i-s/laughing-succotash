"""Codex ACP transport with protected chat settings and no Cursor CLI arguments."""
from __future__ import annotations

import json

from .acp_client import AcpClient, option_of_kind
from .base import AgentError
from .cursor_acp import CursorACPBackend


class CodexACPBackend(CursorACPBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assistant_tool_calls: set[tuple[str, str]] = set()

    async def _on_update(self, session_id: str, update: dict) -> None:
        if (update.get("_meta") or {}).get("is_mcp_tool_call") is True:
            raw = update.get("rawInput") or {}
            if raw.get("server") == "assistant" and update.get("toolCallId"):
                self._assistant_tool_calls.add((session_id, update["toolCallId"]))
            # ACP labels MCP calls 'execute'; they are not built-in shell execution.
            update = {**update, "kind": "other"}
        await super()._on_update(session_id, update)

    async def send_message(self, session_id, message, context=None, *, on_progress=None):
        try:
            return await super().send_message(session_id, message, context, on_progress=on_progress)
        finally:
            self._assistant_tool_calls = {
                key for key in self._assistant_tool_calls if key[0] != session_id
            }

    @property
    def name(self) -> str:
        return "codex-acp"

    def _make_client(self) -> AcpClient:
        config = {
            "features": {
                "shell_tool": False,
                "apply_patch_freeform": False,
                "js_repl": False,
                "multi_agent": False,
            },
            "sandbox_mode": "read-only",
            "approval_policy": "on-request",
        }
        if self._model:
            config["model"] = self._model
        return AcpClient(
            self._binary,
            argv=[self._binary],
            cwd=self._default_workspace,
            request_timeout=self._prompt_timeout,
            env={"INITIAL_AGENT_MODE": "read-only", "CODEX_CONFIG": json.dumps(config)},
        )

    async def set_mode(self, session_id: str, mode: str) -> None:
        if mode != "plan":
            raise AgentError("Codex chat backend only supports protected mode")
        await super().set_mode(session_id, "read-only")
        self._plan_sessions.add(session_id)

    async def _on_permission(self, params: dict) -> str | None:
        options = params.get("options") or []
        key = (params.get("sessionId"), (params.get("toolCall") or {}).get("toolCallId"))
        if ((params.get("_meta") or {}).get("is_mcp_tool_approval") is True
                and key in self._assistant_tool_calls):
            # The local assistant MCP server applies exact tool authorization and confirmation.
            return option_of_kind(options, "allow_once")
        return option_of_kind(options, "reject_once")
