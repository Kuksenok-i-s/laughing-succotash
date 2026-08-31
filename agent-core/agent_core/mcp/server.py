"""Loopback HTTP MCP server exposing the assistant's capabilities to Cursor.

HTTP rather than stdio so the handlers run in this process, with direct access to the SQLite
repositories, the scheduler, and the confirmation service that has to ask the user a question and
await the answer (ADR 5).

The wire shape Cursor expects was probed, not assumed — see ``docs/cursor-acp.md``. In particular
the ``session/new`` entry must carry both ``type`` and a present ``headers`` array.
"""

from __future__ import annotations

import hmac
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from aiohttp import web
from pa_protocol import new_ulid
from pydantic import BaseModel, ValidationError

from ..storage.repositories import OperationLedger
from .permissions import Decision, Tier, ToolContext, decide, describe_action, tier_of

log = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"

ToolHandler = Callable[[BaseModel, ToolContext], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    schema: type[BaseModel]
    handler: ToolHandler
    # Tools whose effects must survive a replay carry an operation_id and are recorded in the
    # ledger, so a retried call returns the original result instead of acting twice.
    idempotent_write: bool = False

    def to_mcp(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": _json_schema(self.schema),
        }


def _json_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    # Pydantic emits $defs/$ref, which some MCP clients handle poorly. Flattening keeps the tool
    # schemas simple enough to be universally readable.
    schema.pop("title", None)
    return schema


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(
        self,
        name: str,
        description: str,
        schema: type[BaseModel],
        handler: ToolHandler,
        *,
        idempotent_write: bool = False,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"duplicate tool: {name}")
        self._tools[name] = Tool(name, description, schema, handler, idempotent_write)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def names(self) -> list[str]:
        return sorted(self._tools)


class ContextRegistry:
    """Maps a session token to the context of the turn currently running for it.

    One Cursor session serves one conversation, and turns within a conversation are serialised, so
    a single current-context slot per token is sufficient and unambiguous.
    """

    def __init__(self) -> None:
        self._tokens: dict[str, str] = {}  # token -> conversation_id
        self._current: dict[str, ToolContext] = {}  # conversation_id -> context

    def issue_token(self, conversation_id: str) -> str:
        token = new_ulid()
        self._tokens[token] = conversation_id
        return token

    def bind_token(self, token: str, conversation_id: str) -> None:
        self._tokens[token] = conversation_id

    def resolve(self, token: str) -> ToolContext | None:
        conversation_id = self._tokens.get(token)
        if conversation_id is None:
            return None
        return self._current.get(conversation_id)

    def set_current(self, conversation_id: str, context: ToolContext) -> None:
        self._current[conversation_id] = context

    def clear_current(self, conversation_id: str) -> None:
        self._current.pop(conversation_id, None)


class McpServer:
    def __init__(
        self,
        registry: ToolRegistry,
        contexts: ContextRegistry,
        operations: OperationLedger,
        confirmations,
        *,
        host: str = "127.0.0.1",
        port: int = 8931,
        token: str = "",
    ) -> None:
        self._registry = registry
        self._contexts = contexts
        self._operations = operations
        self._confirmations = confirmations
        self._host = host
        self._port = port
        self._token = token
        self._runner: web.AppRunner | None = None
        self._actual_port = port

    @property
    def port(self) -> int:
        return self._actual_port

    def session_entry(self, session_token: str) -> dict[str, Any]:
        """The ``mcpServers`` entry to hand to ACP ``session/new``.

        ``type`` and ``headers`` are both mandatory in the installed Cursor build; omitting either
        yields an opaque ``-32603`` with a Zod union error.
        """
        return {
            "name": "assistant",
            "type": "http",
            "url": f"http://{self._host}:{self._actual_port}/mcp/{session_token}",
            "headers": [{"name": "Authorization", "value": f"Bearer {self._token}"}],
        }

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/mcp/{token}", self._handle)
        # Cursor may probe with GET before POSTing; answering 405 rather than 404 keeps it from
        # concluding the server does not exist.
        app.router.add_get("/mcp/{token}", self._handle_get)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

        if self._port == 0:
            sockets = getattr(site._server, "sockets", None)  # noqa: SLF001
            if sockets:
                self._actual_port = sockets[0].getsockname()[1]

        log.info("mcp server listening on http://%s:%d/mcp", self._host, self._actual_port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    # ---- http ----------------------------------------------------------

    async def _handle_get(self, _request: web.Request) -> web.Response:
        return web.Response(status=405, text="use POST")

    async def _handle(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            return web.json_response({"error": "unauthorized"}, status=401)

        token = request.match_info["token"]
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"jsonrpc": "2.0", "id": None,
                 "error": {"code": -32700, "message": "parse error"}}
            )

        method = body.get("method")
        message_id = body.get("id")

        if method == "initialize":
            return self._ok(message_id, {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "personal-assistant-mcp", "version": "1.0.0"},
            })

        if method in ("notifications/initialized", "notifications/cancelled"):
            return web.Response(status=202)

        if method == "ping":
            return self._ok(message_id, {})

        if method == "tools/list":
            return self._ok(message_id, {"tools": [t.to_mcp() for t in self._registry.list()]})

        if method == "tools/call":
            return await self._call_tool(token, message_id, body.get("params") or {})

        return web.json_response(
            {"jsonrpc": "2.0", "id": message_id,
             "error": {"code": -32601, "message": f"method not found: {method}"}}
        )

    def _authorized(self, request: web.Request) -> bool:
        if not self._token:
            return True
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[7:], self._token)

    @staticmethod
    def _ok(message_id: Any, result: dict[str, Any]) -> web.Response:
        response = web.json_response({"jsonrpc": "2.0", "id": message_id, "result": result})
        response.headers["Mcp-Session-Id"] = "personal-assistant"
        return response

    @staticmethod
    def _tool_result(message_id: Any, payload: Any, *, is_error: bool = False) -> web.Response:
        import json as _json

        text = payload if isinstance(payload, str) else _json.dumps(
            payload, ensure_ascii=False, default=str
        )
        return McpServer._ok(
            message_id, {"content": [{"type": "text", "text": text}], "isError": is_error}
        )

    # ---- dispatch --------------------------------------------------------

    async def _call_tool(
        self, token: str, message_id: Any, params: dict[str, Any]
    ) -> web.Response:
        name = params.get("name", "")
        arguments = params.get("arguments") or {}

        tool = self._registry.get(name)
        if tool is None:
            return self._tool_result(message_id, {"error": f"unknown tool: {name}"}, is_error=True)

        context = self._contexts.resolve(token)
        if context is None:
            # No turn is running for this session. Either the token is stale or Cursor is calling
            # outside a prompt; either way there is no authenticated user to act for.
            log.warning("tool %s called with no active turn context", name)
            return self._tool_result(
                message_id,
                {"error": "no active conversation context; the tool call was not executed"},
                is_error=True,
            )

        try:
            validated = tool.schema.model_validate(arguments)
        except ValidationError as exc:
            return self._tool_result(
                message_id,
                {"error": "invalid arguments", "detail": exc.errors(include_url=False)},
                is_error=True,
            )

        normalised = validated.model_dump(mode="json", exclude_none=True)
        decision = decide(name, context)
        log.info(
            "mcp %s user=%s job=%s decision=%s",
            name, context.user_id, context.job_id, decision.value,
        )

        if decision is Decision.DENY:
            return self._tool_result(
                message_id, {"error": "not permitted"}, is_error=True
            )

        operation_id = _operation_id(arguments)

        if tool.idempotent_write:
            previous = await self._operations.lookup(operation_id)
            if previous is not None:
                log.info("replaying recorded result for operation %s (%s)", operation_id, name)
                return self._tool_result(message_id, previous)

        if decision is Decision.CONFIRM:
            approved = await self._confirmations.request(
                user_id=context.user_id,
                chat_id=context.chat_id,
                tool_name=name,
                arguments=normalised,
                operation_id=operation_id,
                tier=tier_of(name).value,
                prompt_text=describe_action(name, normalised),
                job_id=context.job_id,
            )
            if not approved:
                return self._tool_result(
                    message_id,
                    {
                        "status": "rejected",
                        "reason": "the user did not confirm this action",
                        "hint": "Do not retry. Tell the user it was not performed.",
                    },
                )

        try:
            result = await tool.handler(validated, context)
        except PermissionError as exc:
            return self._tool_result(message_id, {"error": str(exc)}, is_error=True)
        except Exception as exc:
            log.exception("tool %s failed", name)
            return self._tool_result(
                message_id,
                {"error": f"{type(exc).__name__}: {exc}"[:300]},
                is_error=True,
            )

        if tool.idempotent_write:
            await self._operations.record(operation_id, name, context.user_id, result)

        return self._tool_result(message_id, result)


def _operation_id(arguments: dict[str, Any]) -> str:
    """Use a caller-supplied idempotency key when present, otherwise mint one.

    A minted key still protects against MCP-level retries of the same HTTP request, which is the
    case that actually occurs; it cannot deduplicate two genuinely separate model decisions, and
    it should not.
    """
    supplied = arguments.get("operation_id")
    if isinstance(supplied, str) and supplied:
        return supplied
    return new_ulid()
