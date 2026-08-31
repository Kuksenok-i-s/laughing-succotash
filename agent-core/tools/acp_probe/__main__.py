"""Re-verify every claim in docs/cursor-acp.md against the installed Cursor Agent.

    python -m tools.acp_probe --all
    python -m tools.acp_probe --list
    python -m tools.acp_probe initialize permissions

Exits non-zero if a previously verified capability has disappeared, which is the point: the ACP
surface is undocumented and version-gated, so after a CLI upgrade the findings the Core is built on
have to be checked rather than assumed.

Deliberately standalone — it speaks the protocol directly instead of going through
``agent_core.agent``. A probe that used our own client could pass because both sides share a wrong
assumption.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

BINARY = os.environ.get("CURSOR_AGENT_BINARY", "cursor-agent")
STDOUT_LIMIT = 32 * 1024 * 1024


class Acp:
    """Minimal newline-delimited JSON-RPC client for the acp subcommand."""

    def __init__(self, cwd: Path) -> None:
        self._cwd = cwd
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._reader: asyncio.Task | None = None

        self.updates: list[dict[str, Any]] = []
        self.permission_requests: list[dict[str, Any]] = []
        # Which option to answer a permission request with, by kind.
        self.permission_answer = "reject_once"
        self.capabilities: dict[str, Any] = {}

    async def start(self) -> dict[str, Any]:
        self._process = await asyncio.create_subprocess_exec(
            BINARY, "acp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=str(self._cwd),
            limit=STDOUT_LIMIT,
        )
        self._reader = asyncio.ensure_future(self._read())
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
        self.capabilities = result.get("agentCapabilities", {})
        return result

    async def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.returncode is None and process.stdin is not None:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except asyncio.TimeoutError:
                process.kill()
        if self._reader is not None:
            self._reader.cancel()
        self._process = None

    async def call(self, method: str, params: dict, *, timeout: float = 600.0) -> dict:
        assert self._process is not None and self._process.stdin is not None
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _send(self, payload: dict) -> None:
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
        await self._process.stdin.drain()

    async def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        while True:
            line = await self._process.stdout.readline()
            if not line:
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(RuntimeError("acp exited"))
                return
            try:
                message = json.loads(line.decode("utf-8", "replace").strip())
            except json.JSONDecodeError:
                continue
            await self._dispatch(message)

    async def _dispatch(self, message: dict) -> None:
        if "id" in message and ("result" in message or "error" in message):
            future = self._pending.pop(int(message["id"]), None)
            if future is None or future.done():
                return
            if "error" in message:
                future.set_exception(RuntimeError(json.dumps(message["error"])))
            else:
                future.set_result(message.get("result") or {})
            return

        method = message.get("method")
        if method == "session/update":
            self.updates.append((message.get("params") or {}).get("update") or {})
        elif method == "session/request_permission" and "id" in message:
            params = message.get("params") or {}
            self.permission_requests.append(params)
            options = params.get("options") or []
            chosen = next(
                (o["optionId"] for o in options if o.get("kind") == self.permission_answer),
                options[0]["optionId"] if options else None,
            )
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"outcome": {"outcome": "selected", "optionId": chosen}},
                }
            )
        elif method in ("fs/read_text_file", "fs/write_text_file") and "id" in message:
            await self._send({"jsonrpc": "2.0", "id": message["id"], "result": {}})

    def kinds(self, name: str) -> list[dict]:
        return [u for u in self.updates if u.get("sessionUpdate") == name]

    def text(self) -> str:
        return "".join(
            (u.get("content") or {}).get("text", "") for u in self.kinds("agent_message_chunk")
        )


@asynccontextmanager
async def acp(tmp: Path):
    client = Acp(tmp)
    await client.start()
    try:
        yield client
    finally:
        await client.close()


# ---- probes ---------------------------------------------------------------

PROBES: dict[str, Any] = {}


def probe(name: str):
    def register(func):
        PROBES[name] = func
        return func

    return register


@probe("initialize")
async def probe_initialize(tmp: Path) -> list[str]:
    """The capability flags that decide the shape of the whole pipeline."""
    findings: list[str] = []
    async with acp(tmp) as client:
        caps = client.capabilities
        prompt = caps.get("promptCapabilities", {})

        if prompt.get("audio") is not False:
            findings.append(
                f"promptCapabilities.audio is now {prompt.get('audio')!r}; audio may no longer "
                "need local Whisper — revisit stt/ and docs/adr/0004"
            )
        if prompt.get("embeddedContext") is not False:
            findings.append(
                "promptCapabilities.embeddedContext is now true; transcripts could be attached as "
                "resources instead of embedded in-band — revisit assistant/transcript.py"
            )
        if caps.get("loadSession") is not True:
            findings.append("loadSession is gone; conversations can no longer survive a restart")
        if not (caps.get("mcpCapabilities") or {}).get("http"):
            findings.append("MCP over HTTP is no longer advertised; mcp/server.py depends on it")
    return findings


@probe("session")
async def probe_session(tmp: Path) -> list[str]:
    """Session creation, streaming, and where the reply text actually comes from."""
    findings: list[str] = []
    async with acp(tmp) as client:
        session = (await client.call("session/new", {"cwd": str(tmp), "mcpServers": []}))[
            "sessionId"
        ]

        result = await client.call(
            "session/prompt",
            {
                "sessionId": session,
                "prompt": [{"type": "text", "text": "Ответь одним словом: 2+2?"}],
            },
        )

        if result.get("stopReason") != "end_turn":
            findings.append(f"unexpected stopReason: {result!r}")
        if "text" in result or "content" in result:
            findings.append(
                "session/prompt now returns reply content directly; the streaming accumulation in "
                "agent/cursor_acp.py could be simplified"
            )
        if not client.text().strip():
            findings.append("no agent_message_chunk updates arrived; the reply cannot be read")
    return findings


@probe("resume")
async def probe_resume(tmp: Path) -> list[str]:
    """session/load across two processes, which is what makes a conversation durable."""
    findings: list[str] = []
    async with acp(tmp) as first:
        session = (await first.call("session/new", {"cwd": str(tmp), "mcpServers": []}))[
            "sessionId"
        ]
        await first.call(
            "session/prompt",
            {
                "sessionId": session,
                "prompt": [{"type": "text", "text": "Запомни число 8127. Ответь 'ок'."}],
            },
        )

    async with acp(tmp) as second:
        try:
            await second.call(
                "session/load", {"sessionId": session, "cwd": str(tmp), "mcpServers": []}
            )
        except RuntimeError as exc:
            return [f"session/load failed: {exc}"]

        if not second.kinds("user_message_chunk"):
            findings.append(
                "session/load no longer replays history; the replay-suppression logic in "
                "agent/cursor_acp.py may now be unnecessary"
            )
        before = len(second.updates)
        await second.call(
            "session/prompt",
            {
                "sessionId": session,
                "prompt": [{"type": "text", "text": "Какое число я просил запомнить?"}],
            },
        )
        answer = "".join(
            (u.get("content") or {}).get("text", "")
            for u in second.updates[before:]
            if u.get("sessionUpdate") == "agent_message_chunk"
        )
        if "8127" not in answer:
            findings.append(f"resumed session lost its context (answered: {answer.strip()[:80]!r})")
    return findings


@probe("cancel")
async def probe_cancel(tmp: Path) -> list[str]:
    """Cancellation must stop the turn and leave the session usable."""
    findings: list[str] = []
    async with acp(tmp) as client:
        session = (await client.call("session/new", {"cwd": str(tmp), "mcpServers": []}))[
            "sessionId"
        ]
        task = asyncio.ensure_future(
            client.call(
                "session/prompt",
                {
                    "sessionId": session,
                    "prompt": [
                        {"type": "text", "text": "Напиши эссе на 3000 слов о почтовых голубях."}
                    ],
                },
            )
        )
        await asyncio.sleep(6)
        await client.notify("session/cancel", {"sessionId": session})

        result = await asyncio.wait_for(task, 60)
        if result.get("stopReason") != "cancelled":
            findings.append(f"cancel did not stop the turn: {result!r}")

        after = await client.call(
            "session/prompt",
            {"sessionId": session, "prompt": [{"type": "text", "text": "Скажи 'ок'."}]},
        )
        if after.get("stopReason") != "end_turn":
            findings.append("session is unusable after a cancel; /cancel would break conversations")
    return findings


@probe("plan-mode")
async def probe_plan_mode(tmp: Path) -> list[str]:
    """plan mode is the read-only boundary for non-writable projects."""
    findings: list[str] = []
    canary = tmp / "canary.txt"
    async with acp(tmp) as client:
        session = (await client.call("session/new", {"cwd": str(tmp), "mcpServers": []}))[
            "sessionId"
        ]
        await client.call("session/set_mode", {"sessionId": session, "modeId": "plan"})
        turn = asyncio.ensure_future(
            client.call(
                "session/prompt",
                {
                    "sessionId": session,
                    "prompt": [
                        {
                            "type": "text",
                            "text": f"Создай файл {canary} с текстом 'written' и запусти echo hi.",
                        }
                    ],
                },
            )
        )
        # The turn is cut short on purpose. In plan mode the agent answers with a plan document
        # that can run for many minutes, and the only thing under test is whether it wrote to
        # disk — which it would do early, not after finishing its prose.
        try:
            await asyncio.wait_for(asyncio.shield(turn), 90)
        except asyncio.TimeoutError:
            await client.notify("session/cancel", {"sessionId": session})
            await asyncio.wait_for(turn, 60)

    if canary.exists():
        findings.append(
            "plan mode wrote to disk; read-only projects are no longer protected by it — "
            "revisit the projects allowlist handling"
        )
    return findings


@probe("plan-mcp")
async def probe_plan_mcp(tmp: Path) -> list[str]:
    """Chat sessions run in plan mode; MCP tools must still be callable there.

    If this fails, do not switch Telegram chat sessions to plan — Cursor would lose calendar /
    reminder / task tools while still blocking shell. Fall back to cwd sandbox only.
    """
    findings: list[str] = []
    calls: list[str] = []
    canary = tmp / "plan_mcp_canary.txt"

    class Handler(asyncio.Protocol):
        def connection_made(self, transport):  # type: ignore[no-untyped-def]
            self.transport = transport
            self.buffer = b""

        def data_received(self, data: bytes) -> None:
            self.buffer += data
            if b"\r\n\r\n" not in self.buffer:
                return
            header, _, rest = self.buffer.partition(b"\r\n\r\n")
            length = 0
            for line in header.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":", 1)[1].strip())
            body = rest
            if len(body) < length:
                return
            body = body[:length]
            self.buffer = b""
            try:
                message = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                self._reply(400, b'{"error":"bad json"}')
                return
            method = message.get("method")
            message_id = message.get("id")
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "acp-probe-mcp", "version": "1.0.0"},
                }
                self._json(message_id, result)
            elif method in ("notifications/initialized", "notifications/cancelled"):
                self.transport.write(b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\n\r\n")
            elif method == "tools/list":
                self._json(
                    message_id,
                    {
                        "tools": [
                            {
                                "name": "probe_ping",
                                "description": "Return the magic word. No arguments.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {},
                                    "additionalProperties": False,
                                },
                            }
                        ]
                    },
                )
            elif method == "tools/call":
                name = (message.get("params") or {}).get("name", "")
                calls.append(name)
                self._json(
                    message_id,
                    {
                        "content": [{"type": "text", "text": "pong-plan"}],
                        "isError": False,
                    },
                )
            else:
                self._json(
                    message_id,
                    None,
                    error={"code": -32601, "message": f"unknown: {method}"},
                )

        def _json(self, message_id, result, error=None):  # type: ignore[no-untyped-def]
            payload: dict[str, Any] = {"jsonrpc": "2.0", "id": message_id}
            if error is not None:
                payload["error"] = error
            else:
                payload["result"] = result
            raw = json.dumps(payload).encode()
            self._reply(200, raw, content_type="application/json")

        def _reply(self, status: int, body: bytes, content_type: str = "text/plain") -> None:
            reason = {200: "OK", 400: "Bad Request"}.get(status, "OK")
            headers = (
                f"HTTP/1.1 {status} {reason}\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Mcp-Session-Id: probe\r\n"
                f"\r\n"
            ).encode()
            self.transport.write(headers + body)

    server = await asyncio.get_running_loop().create_server(Handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}/mcp"
    mcp = {
        "name": "probe",
        "type": "http",
        "url": url,
        "headers": [],
    }

    try:
        async with acp(tmp) as client:
            client.permission_answer = "allow_once"
            session = (
                await client.call(
                    "session/new", {"cwd": str(tmp), "mcpServers": [mcp]}, timeout=60
                )
            )["sessionId"]
            await client.call("session/set_mode", {"sessionId": session, "modeId": "plan"})
            await client.call(
                "session/prompt",
                {
                    "sessionId": session,
                    "prompt": [
                        {
                            "type": "text",
                            "text": (
                                "Вызови MCP-инструмент probe_ping (без аргументов) и в ответе "
                                "напиши ровно то, что он вернул. Не создавай файлы и не запускай "
                                f"shell. Не пиши в {canary}."
                            ),
                        }
                    ],
                },
                timeout=180,
            )

            execute_kinds = [
                u
                for u in client.kinds("tool_call")
                if u.get("kind") == "execute"
            ]
            if execute_kinds:
                findings.append(
                    "plan mode still emitted execute tool_call; chat set_mode(plan) would not "
                    "block shell — keep cwd sandbox only"
                )
            if canary.exists():
                findings.append("plan+MCP probe wrote a canary file; plan mode is not read-only")
            if "probe_ping" not in calls:
                findings.append(
                    "MCP tools/call never reached the probe server in plan mode; do not switch "
                    "Telegram chat sessions to plan (assistant MCP would break). "
                    f"permission_requests={len(client.permission_requests)} "
                    f"reply={client.text().strip()[:120]!r}"
                )
    finally:
        server.close()
        await server.wait_closed()
    return findings


@probe("permissions")
async def probe_permissions(tmp: Path) -> list[str]:
    """The asymmetry the whole security design rests on.

    MCP tool calls request permission; built-in writes and shell commands do not. If that ever
    changes, the sandbox-cwd requirement and the two-layer permission model should be revisited.
    """
    findings: list[str] = []
    canary = tmp / "builtin_write.txt"

    async with acp(tmp) as client:
        session = (await client.call("session/new", {"cwd": str(tmp), "mcpServers": []}))[
            "sessionId"
        ]
        await client.call(
            "session/prompt",
            {
                "sessionId": session,
                "prompt": [
                    {
                        "type": "text",
                        "text": f"Создай файл {canary} с текстом 'written'. Ничего не спрашивай.",
                    }
                ],
            },
        )

        if canary.exists() and client.permission_requests:
            findings.append(
                "built-in writes now request permission; the ACP callback could become the "
                "authoritative gate and the sandbox cwd may no longer be required"
            )
        if not canary.exists() and not client.permission_requests:
            findings.append(
                "the built-in write neither happened nor asked; the probe could not determine "
                "the permission behaviour (model may have refused for another reason)"
            )
    return findings


@probe("mcp-http")
async def probe_mcp_http(tmp: Path) -> list[str]:
    """The exact HTTP MCP object shape, which is stricter than the ACP spec suggests."""
    findings: list[str] = []
    url = "http://127.0.0.1:8/mcp"  # never connected to; only the shape is under test

    async with acp(tmp) as client:
        # Documented as rejected: no type, and type without headers.
        for bad in ({"name": "x", "url": url}, {"name": "x", "type": "http", "url": url}):
            try:
                await client.call(
                    "session/new", {"cwd": str(tmp), "mcpServers": [bad]}, timeout=60
                )
            except RuntimeError:
                pass
            else:
                findings.append(
                    f"mcpServers entry {bad!r} is now accepted; the strict shape requirement in "
                    "assistant/sessions.py may be obsolete"
                )

        good = {"name": "assistant", "type": "http", "url": url, "headers": []}
        try:
            await client.call("session/new", {"cwd": str(tmp), "mcpServers": [good]}, timeout=60)
        except RuntimeError as exc:
            findings.append(f"the documented HTTP MCP shape is now rejected: {exc}")
    return findings


# ---- runner ---------------------------------------------------------------


async def run(names: list[str]) -> int:
    root = Path(os.environ.get("ACP_PROBE_DIR", "/tmp/acp-probe")).resolve()
    failures = 0

    for name in names:
        workspace = root / name
        workspace.mkdir(parents=True, exist_ok=True)
        print(f"\n== {name} ", end="", flush=True)
        started = time.monotonic()
        try:
            findings = await PROBES[name](workspace)
        except Exception as exc:
            print(f"ERROR ({time.monotonic() - started:.1f}s)")
            print(f"   {type(exc).__name__}: {exc}")
            failures += 1
            continue

        elapsed = time.monotonic() - started
        if findings:
            print(f"CHANGED ({elapsed:.1f}s)")
            for finding in findings:
                print(f"   - {finding}")
            failures += 1
        else:
            print(f"ok ({elapsed:.1f}s)")

    print()
    if failures:
        print(
            f"{failures} probe(s) reported a change. docs/cursor-acp.md and the code paths named "
            "above need review before deploying."
        )
    else:
        print(f"all {len(names)} probe(s) match docs/cursor-acp.md")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", nargs="*", help="probe names to run")
    parser.add_argument("--all", action="store_true", help="run every probe")
    parser.add_argument("--list", action="store_true", help="list probe names")
    args = parser.parse_args()

    if args.list:
        for name, func in PROBES.items():
            summary = (func.__doc__ or "").strip().splitlines()[0]
            print(f"{name:12} {summary}")
        return 0

    names = list(PROBES) if args.all else args.probes
    if not names:
        parser.print_help()
        return 2
    unknown = [name for name in names if name not in PROBES]
    if unknown:
        print(f"unknown probe(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    return asyncio.run(run(names))


if __name__ == "__main__":
    sys.exit(main())
