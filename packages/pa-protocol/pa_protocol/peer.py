"""Bidirectional JSON-RPC peer over an abstract duplex transport.

Both the Gateway and the Core use this class. It is transport-agnostic so tests can drive a full
conversation over an in-memory pipe without a network or a Telegram account.

Responsibilities kept deliberately narrow: request/response correlation, concurrent handler
dispatch, binary frame routing and shutdown. Reconnection, sequencing and idempotency live one
layer up, because they need persistence.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from .errors import INTERNAL_ERROR, METHOD_NOT_FOUND, RpcError
from .frames import AudioFrame, FrameError, decode_frame
from .messages import Request, Response, dumps, parse

log = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[Any]]
BinaryHandler = Callable[[AudioFrame], Awaitable[None]]


class Transport(Protocol):
    """The subset of a WebSocket connection the peer relies on."""

    async def send_text(self, data: str) -> None: ...
    async def send_bytes(self, data: bytes) -> None: ...
    async def receive(self) -> str | bytes | None:
        """Return the next message, or ``None`` once the connection is closed."""
        ...

    async def close(self) -> None: ...


class ConnectionClosed(Exception):
    """Raised into pending calls when the transport goes away."""


class RpcPeer:
    def __init__(
        self,
        transport: Transport,
        *,
        name: str = "peer",
        default_timeout: float = 30.0,
        on_binary: BinaryHandler | None = None,
    ) -> None:
        self._transport = transport
        self._name = name
        self._default_timeout = default_timeout
        self._on_binary = on_binary

        self._handlers: dict[str, Handler] = {}
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._next_id = 0
        self._closed = asyncio.Event()
        self._send_lock = asyncio.Lock()

    # ---- registration -------------------------------------------------

    def register(self, method: str, handler: Handler) -> None:
        self._handlers[method] = handler

    def register_all(self, handlers: dict[str, Handler]) -> None:
        self._handlers.update(handlers)

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    async def wait_closed(self) -> None:
        await self._closed.wait()

    # ---- outbound -----------------------------------------------------

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Send a request and await its result. Raises :class:`RpcError` on an error response."""
        if self._closed.is_set():
            raise ConnectionClosed(f"{self._name}: transport closed")

        self._next_id += 1
        request_id = f"{self._next_id}"
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future

        try:
            await self._send(dumps(Request(method=method, params=params or {}, id=request_id)))
            return await asyncio.wait_for(
                future, timeout if timeout is not None else self._default_timeout
            )
        except asyncio.TimeoutError:
            # A timeout here means the response never arrived, not that the peer did not act on
            # the request. Callers must retry with the same idempotency key rather than assume
            # nothing happened.
            raise
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self._closed.is_set():
            raise ConnectionClosed(f"{self._name}: transport closed")
        await self._send(dumps(Request(method=method, params=params or {})))

    async def send_binary(self, data: bytes) -> None:
        if self._closed.is_set():
            raise ConnectionClosed(f"{self._name}: transport closed")
        async with self._send_lock:
            await self._transport.send_bytes(data)

    async def _send(self, text: str) -> None:
        async with self._send_lock:
            await self._transport.send_text(text)

    # ---- inbound ------------------------------------------------------

    async def run(self) -> None:
        """Read from the transport until it closes. Returns after cleanup."""
        try:
            while True:
                message = await self._transport.receive()
                if message is None:
                    break
                if isinstance(message, bytes):
                    await self._handle_binary(message)
                else:
                    await self._handle_text(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("%s: receive loop failed", self._name)
        finally:
            await self._shutdown()

    async def _handle_binary(self, raw: bytes) -> None:
        if self._on_binary is None:
            log.warning("%s: binary frame received but no handler registered", self._name)
            return
        try:
            frame = decode_frame(raw)
        except FrameError:
            log.warning("%s: dropping undecodable binary frame", self._name, exc_info=True)
            return
        try:
            await self._on_binary(frame)
        except Exception:
            log.exception("%s: binary handler failed", self._name)

    async def _handle_text(self, raw: str) -> None:
        try:
            message = parse(raw)
        except RpcError as exc:
            await self._send(dumps(Response(id=None, error=exc)))
            return

        if isinstance(message, Response):
            self._resolve(message)
            return

        if message.is_notification:
            self._spawn(self._dispatch_notification(message))
        else:
            self._spawn(self._dispatch_request(message))

    def _resolve(self, response: Response) -> None:
        future = self._pending.pop(str(response.id), None)
        if future is None or future.done():
            # Late response to a call that already timed out, or a duplicate. Dropping it is
            # correct; the caller has moved on and will retry idempotently if it needs to.
            return
        if response.error is not None:
            future.set_exception(response.error)
        else:
            future.set_result(response.result)

    async def _dispatch_request(self, request: Request) -> None:
        handler = self._handlers.get(request.method)
        if handler is None:
            error = RpcError(METHOD_NOT_FOUND, "method_not_found", {"method": request.method})
            await self._send(dumps(Response(id=request.id, error=error)))
            return

        try:
            result = await handler(request.params)
        except RpcError as exc:
            await self._send(dumps(Response(id=request.id, error=exc)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("%s: handler for %s failed", self._name, request.method)
            # The exception text is not forwarded: it can contain paths or user content, and the
            # peer has no use for it beyond the stable code.
            await self._send(
                dumps(
                    Response(
                        id=request.id,
                        error=RpcError(
                            INTERNAL_ERROR, "internal_error", {"method": request.method}
                        ),
                    )
                )
            )
            del exc
        else:
            await self._send(dumps(Response(id=request.id, result=result)))

    async def _dispatch_notification(self, request: Request) -> None:
        handler = self._handlers.get(request.method)
        if handler is None:
            log.debug("%s: no handler for notification %s", self._name, request.method)
            return
        try:
            await handler(request.params)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("%s: notification handler for %s failed", self._name, request.method)

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ---- shutdown -----------------------------------------------------

    async def _shutdown(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()

        for future in self._pending.values():
            if not future.done():
                future.set_exception(ConnectionClosed(f"{self._name}: transport closed"))
        self._pending.clear()

        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

    async def close(self) -> None:
        await self._transport.close()
        await self._shutdown()
