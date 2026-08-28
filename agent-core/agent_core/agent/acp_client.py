"""Low-level client for ``cursor-agent acp``.

Written against the traffic recorded in ``docs/cursor-acp.md``, not against the generic ACP
specification. Where the installed build differs, this follows the build.

Transport is newline-delimited JSON-RPC 2.0 over the subprocess's stdin/stdout. There is no
``Content-Length`` framing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .fs_sandbox import resolve_under_root

log = logging.getLogger(__name__)

# Cursor streams replies in very small pieces; a single line can still be large when a tool result
# is echoed back, so the reader gets a generous limit.
_STDOUT_LIMIT = 32 * 1024 * 1024

UpdateHandler = Callable[[str, dict[str, Any]], Awaitable[None]]
PermissionHandler = Callable[[dict[str, Any]], Awaitable[str | None]]


class AcpProcessError(RuntimeError):
    pass


class AcpClient:
    """Owns the ``cursor-agent acp`` subprocess and the JSON-RPC conversation with it."""

    def __init__(
        self,
        binary: str = "cursor-agent",
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        request_timeout: float = 1800.0,
    ) -> None:
        self._binary = binary
        self._cwd = cwd
        self._env = env
        self._request_timeout = request_timeout

        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._next_id = 0
        self._reader: asyncio.Task[None] | None = None
        self._stderr_reader: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._closing = False
        # sessionId -> absolute root the agent may touch via fs/* (default deny elsewhere).
        self._session_roots: dict[str, Path] = {}

        self.agent_capabilities: dict[str, Any] = {}
        self.auth_methods: list[dict[str, Any]] = []

        self.on_update: UpdateHandler | None = None
        self.on_permission: PermissionHandler | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def bind_session_root(self, session_id: str, root: Path) -> None:
        """Restrict ACP filesystem tools for ``session_id`` to ``root`` (and below)."""
        self._session_roots[session_id] = root.expanduser().resolve()

    def unbind_session_root(self, session_id: str) -> None:
        self._session_roots.pop(session_id, None)

    # ---- lifecycle ----------------------------------------------------

    async def start(self) -> None:
        env = dict(os.environ)
        if self._env:
            env.update(self._env)

        try:
            self._process = await asyncio.create_subprocess_exec(
                self._binary,
                "acp",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._cwd) if self._cwd else None,
                env=env,
                limit=_STDOUT_LIMIT,
            )
        except FileNotFoundError as exc:
            raise AcpProcessError(f"{self._binary} not found on PATH") from exc

        self._reader = asyncio.ensure_future(self._read_stdout())
        self._stderr_reader = asyncio.ensure_future(self._read_stderr())

    async def initialize(self) -> dict[str, Any]:
        result = await self.call(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": True, "writeTextFile": True},
                    "terminal": False,
                },
            },
            timeout=60,
        )
        self.agent_capabilities = result.get("agentCapabilities", {}) or {}
        self.auth_methods = result.get("authMethods", []) or []
        return result

    async def close(self) -> None:
        self._closing = True
        self._session_roots.clear()
        process = self._process
        if process is None:
            return

        for future in self._pending.values():
            if not future.done():
                future.set_exception(AcpProcessError("acp client shutting down"))
        self._pending.clear()

        if process.returncode is None:
            try:
                if process.stdin is not None and not process.stdin.is_closing():
                    process.stdin.close()
            except Exception:
                log.debug("error closing acp stdin", exc_info=True)
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                log.warning("acp process did not exit; terminating")
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()

        for task in (self._reader, self._stderr_reader):
            if task is not None:
                task.cancel()
        tasks = [t for t in (self._reader, self._stderr_reader) if t is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._process = None

    # ---- rpc -----------------------------------------------------------

    async def call(
        self, method: str, params: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        if not self.running:
            raise AcpProcessError("acp process is not running")

        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[str(request_id)] = future

        await self._write(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        try:
            return await asyncio.wait_for(
                future, timeout if timeout is not None else self._request_timeout
            )
        finally:
            self._pending.pop(str(request_id), None)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        if not self.running:
            return
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def _write(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise AcpProcessError("acp stdin is unavailable")
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
        async with self._write_lock:
            process.stdin.write(data)
            await process.stdin.drain()

    # ---- reading --------------------------------------------------------

    async def _read_stdout(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None
        try:
            while True:
                try:
                    line = await process.stdout.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    log.error("acp emitted a line beyond the read limit; skipping")
                    continue
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    # Cursor occasionally writes non-JSON diagnostics to stdout. Dropping the line
                    # is preferable to tearing down a working session.
                    log.debug("non-JSON line from acp: %s", text[:200])
                    continue
                await self._dispatch(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("acp stdout reader failed")
        finally:
            if not self._closing:
                self._fail_pending("acp process exited unexpectedly")

    def _fail_pending(self, reason: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(AcpProcessError(reason))
        self._pending.clear()

    async def _read_stderr(self) -> None:
        process = self._process
        assert process is not None and process.stderr is not None
        try:
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                log.debug("acp stderr: %s", line.decode(errors="replace").rstrip()[:500])
        except asyncio.CancelledError:
            raise
        except Exception:
            log.debug("acp stderr reader stopped", exc_info=True)

    async def _dispatch(self, message: dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            future = self._pending.pop(str(message["id"]), None)
            if future is None or future.done():
                return
            if "error" in message:
                error = message["error"] or {}
                future.set_exception(
                    AcpProcessError(f"{error.get('code')}: {error.get('message')}")
                )
            else:
                future.set_result(message.get("result") or {})
            return

        method = message.get("method")
        if method is None:
            return

        if "id" in message:
            asyncio.ensure_future(self._handle_server_request(message))
        elif method == "session/update":
            await self._handle_update(message.get("params") or {})

    async def _handle_update(self, params: dict[str, Any]) -> None:
        if self.on_update is None:
            return
        session_id = params.get("sessionId", "")
        update = params.get("update") or {}
        try:
            await self.on_update(session_id, update)
        except Exception:
            log.exception("session update handler failed")

    async def _handle_server_request(self, message: dict[str, Any]) -> None:
        """Answer a request originated by Cursor.

        Filesystem calls are gated against the session root registered at ``session/new`` /
        ``session/load``. Paths outside that root are denied (ADR 7 / per-user sandbox).
        """
        method = message["method"]
        params = message.get("params") or {}
        result: dict[str, Any] = {}

        try:
            if method == "session/request_permission":
                result = await self._answer_permission(params)
            elif method == "fs/read_text_file":
                result = self._read_file(params)
            elif method == "fs/write_text_file":
                result = self._write_file(params)
            else:
                log.debug("unhandled acp client request: %s", method)
        except Exception:
            log.exception("failed to answer acp request %s", method)
            result = {}

        try:
            await self._write({"jsonrpc": "2.0", "id": message["id"], "result": result})
        except Exception:
            log.debug("could not reply to acp request", exc_info=True)

    async def _answer_permission(self, params: dict[str, Any]) -> dict[str, Any]:
        options = params.get("options") or []
        chosen: str | None = None

        if self.on_permission is not None:
            chosen = await self.on_permission(params)

        if chosen is None:
            chosen = _option_of_kind(options, "reject_once")
        if chosen is None and options:
            chosen = options[0].get("optionId")
        if chosen is None:
            return {"outcome": {"outcome": "cancelled"}}
        return {"outcome": {"outcome": "selected", "optionId": chosen}}

    def _root_for(self, params: dict[str, Any]) -> Path | None:
        session_id = params.get("sessionId") or ""
        if not session_id:
            return None
        return self._session_roots.get(session_id)

    def _read_file(self, params: dict[str, Any]) -> dict[str, Any]:
        root = self._root_for(params)
        if root is None:
            log.warning("fs/read denied: unknown or missing sessionId")
            return {"content": ""}
        allowed = resolve_under_root(root, params.get("path", ""))
        if allowed is None:
            log.warning("fs/read denied outside session root: %s", params.get("path"))
            return {"content": ""}
        try:
            return {"content": allowed.read_text(encoding="utf-8", errors="replace")}
        except OSError:
            return {"content": ""}

    def _write_file(self, params: dict[str, Any]) -> dict[str, Any]:
        root = self._root_for(params)
        if root is None:
            log.warning("fs/write denied: unknown or missing sessionId")
            return {}
        allowed = resolve_under_root(root, params.get("path", ""))
        if allowed is None:
            log.warning("fs/write denied outside session root: %s", params.get("path"))
            return {}
        try:
            allowed.parent.mkdir(parents=True, exist_ok=True)
            if not resolve_under_root(root, allowed.parent):
                log.warning("fs/write denied: parent escaped session root: %s", allowed)
                return {}
            allowed.write_text(params.get("content", ""), encoding="utf-8")
        except OSError:
            log.warning("fs/write failed for %s", allowed, exc_info=True)
        return {}


def option_of_kind(options: list[dict[str, Any]], kind: str) -> str | None:
    return _option_of_kind(options, kind)


def _option_of_kind(options: list[dict[str, Any]], kind: str) -> str | None:
    for option in options:
        if option.get("kind") == kind:
            return option.get("optionId")
    return None
