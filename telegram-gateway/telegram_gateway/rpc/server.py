"""WebSocket endpoint that the Agent Core dials into.

Exactly one Core is expected. A second authenticated connection replaces the first, because the
usual cause is the old socket being half-open after a network change and the Core having given up
on it — refusing the new one would leave the assistant unreachable until a TCP timeout.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from aiohttp import WSMsgType, web
from pa_protocol import RpcError, RpcPeer, Transport, errors, methods
from pa_protocol.messages import PROTOCOL_VERSION

log = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[Any]]

GATEWAY_VERSION = "1.0.0"


class AiohttpWebSocketTransport(Transport):
    def __init__(self, ws: web.WebSocketResponse) -> None:
        self._ws = ws

    async def send_text(self, data: str) -> None:
        await self._ws.send_str(data)

    async def send_bytes(self, data: bytes) -> None:
        await self._ws.send_bytes(data)

    async def receive(self) -> str | bytes | None:
        message = await self._ws.receive()
        if message.type == WSMsgType.TEXT:
            return message.data
        if message.type == WSMsgType.BINARY:
            return message.data
        return None

    async def close(self) -> None:
        if not self._ws.closed:
            await self._ws.close()


class CoreLink:
    """Server-side view of the connected Agent Core."""

    def __init__(self, settings, store) -> None:
        self._settings = settings
        self._store = store
        self._handlers: dict[str, Handler] = {}

        self._peer: RpcPeer | None = None
        self._connected = asyncio.Event()
        self.instance_id: str | None = None
        self.capabilities: list[str] = []
        self.connected_since: datetime | None = None
        # Fired after a successful handshake so the Gateway can flush anything it queued while
        # the Core was away.
        self.on_ready: Callable[[], Awaitable[None]] | None = None

    def register(self, method: str, handler: Handler) -> None:
        self._handlers[method] = handler

    def register_all(self, handlers: dict[str, Handler]) -> None:
        self._handlers.update(handlers)

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    async def call(
        self, method: str, params: dict[str, Any], *, timeout: float | None = None
    ) -> Any:
        peer = self._peer
        if peer is None or not self._connected.is_set():
            raise RpcError(errors.NOT_READY, "not_ready", {"detail": "core is not connected"})
        return await peer.call(
            method, params, timeout=timeout or self._settings.rpc_call_timeout
        )

    async def send_binary(self, data: bytes) -> None:
        peer = self._peer
        if peer is None or not self._connected.is_set():
            raise RpcError(errors.NOT_READY, "not_ready", {"detail": "core is not connected"})
        await peer.send_binary(data)

    async def wait_connected(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ---- request handling ----------------------------------------------

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        if not self._authenticate(request):
            log.warning("rejected core connection: bad or missing service token")
            raise web.HTTPUnauthorized(text="unauthorized")

        ws = web.WebSocketResponse(heartbeat=None, max_msg_size=0, autoping=True)
        await ws.prepare(request)

        previous = self._peer
        if previous is not None:
            log.warning("replacing an existing core connection")
            await previous.close()

        peer = RpcPeer(
            AiohttpWebSocketTransport(ws),
            name="gateway",
            default_timeout=self._settings.rpc_call_timeout,
        )
        peer.register(methods.CORE_HELLO, self._on_hello)
        # Everything else stays unavailable until core.hello succeeds.
        for name, handler in self._handlers.items():
            peer.register(name, self._gate(name, handler))
        self._peer = peer

        try:
            await peer.run()
        finally:
            if self._peer is peer:
                self._peer = None
                self._connected.clear()
                self.connected_since = None
                log.info("core disconnected")
        return ws

    def _authenticate(self, request: web.Request) -> bool:
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return False
        # Constant-time comparison: a timing oracle on the service token would be enough to
        # impersonate the Core.
        return hmac.compare_digest(header[7:], self._settings.core_token)

    def _gate(self, name: str, handler: Handler) -> Handler:
        async def gated(params: dict[str, Any]) -> Any:
            if not self._connected.is_set():
                raise RpcError(
                    errors.NOT_READY, "not_ready", {"detail": f"{name} before core.hello"}
                )
            return await handler(params)

        return gated

    async def _on_hello(self, params: dict[str, Any]) -> dict[str, Any]:
        hello = methods.CoreHelloParams.model_validate(params)
        if hello.protocol_version != PROTOCOL_VERSION:
            raise RpcError(
                errors.PROTOCOL_VERSION_UNSUPPORTED,
                "protocol_version_unsupported",
                {"gateway": PROTOCOL_VERSION, "core": hello.protocol_version},
            )

        self.instance_id = hello.instance_id
        self.capabilities = hello.capabilities
        self.connected_since = datetime.now(timezone.utc)
        self._connected.set()
        log.info(
            "core %s connected (capabilities: %s)",
            hello.instance_id, ", ".join(hello.capabilities) or "none",
        )

        last_received = await self._store.get_seq("core_seq")
        if self.on_ready is not None:
            asyncio.ensure_future(self._safe_ready())

        return methods.dump(
            methods.CoreHelloResult(
                gateway_version=GATEWAY_VERSION,
                protocol_version=PROTOCOL_VERSION,
                last_received_seq=last_received,
                server_time=datetime.now(timezone.utc),
            )
        )

    async def _safe_ready(self) -> None:
        try:
            await self.on_ready()  # type: ignore[misc]
        except Exception:
            log.exception("core-ready hook failed")
