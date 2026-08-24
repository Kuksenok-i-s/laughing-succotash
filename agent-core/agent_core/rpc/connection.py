"""The Core's outbound link to the Telegram Gateway.

The Core dials out, so it owns reconnection. Everything user-visible that the Core originates goes
through :meth:`GatewayLink.send_event`, which commits the event to SQLite before touching the
socket — that is what makes a reminder that fires during an outage still arrive afterwards.
"""

from __future__ import annotations

import asyncio
import logging
import random
import ssl
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

import websockets
from pa_protocol import ConnectionClosed, RpcError, RpcPeer, methods
from pa_protocol.messages import PROTOCOL_VERSION

from ..config import Settings
from ..storage.repositories import OutboundEventRepository
from .transport import WebSocketTransport

log = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[Any]]

# Events the Gateway may legitimately refuse forever. Retrying a send to a user who blocked the
# bot would spin until the log is pruned.
_PERMANENT_FAILURES = {-32051, -32002}


class GatewayLink:
    def __init__(
        self,
        settings: Settings,
        events: OutboundEventRepository,
        *,
        capabilities: list[str] | None = None,
    ) -> None:
        self._settings = settings
        self._events = events
        self._capabilities = capabilities or []
        self._handlers: dict[str, Handler] = {}
        # Set by the owner to receive audio frames; the control plane stays in JSON-RPC.
        self.on_binary: Callable[[Any], Awaitable[None]] | None = None

        self._peer: RpcPeer | None = None
        self._connected = asyncio.Event()
        self._stopping = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._drainer: asyncio.Task[None] | None = None
        self._wake_drainer = asyncio.Event()

        self._attempt = 0
        self._connected_since: datetime | None = None
        self.last_error: str | None = None

    # ---- lifecycle ----------------------------------------------------

    def register(self, method: str, handler: Handler) -> None:
        self._handlers[method] = handler

    def register_all(self, handlers: dict[str, Handler]) -> None:
        self._handlers.update(handlers)

    async def start(self) -> None:
        self._runner = asyncio.ensure_future(self._reconnect_loop())
        self._drainer = asyncio.ensure_future(self._drain_loop())

    async def stop(self) -> None:
        self._stopping.set()
        self._wake_drainer.set()
        for task in (self._runner, self._drainer):
            if task is not None:
                task.cancel()
        tasks = [t for t in (self._runner, self._drainer) if t is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._peer is not None:
            await self._peer.close()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def connected_since(self) -> datetime | None:
        return self._connected_since

    async def wait_connected(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ---- connection ---------------------------------------------------

    async def _reconnect_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await self._connect_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("gateway connection failed: %s", self.last_error)

            if self._stopping.is_set():
                break

            delay = self._next_delay()
            log.info("reconnecting to gateway in %.1fs (attempt %d)", delay, self._attempt)
            try:
                await asyncio.wait_for(self._stopping.wait(), delay)
                break  # stop() was called during the backoff
            except asyncio.TimeoutError:
                continue

    def _next_delay(self) -> float:
        """Exponential backoff with full jitter.

        Full jitter rather than a fixed delay: after a Gateway restart every retry would otherwise
        land at the same instant, and a thundering herd of one is still a self-inflicted spike.
        """
        self._attempt += 1
        ceiling = min(
            self._settings.reconnect_base_delay * (2 ** (self._attempt - 1)),
            self._settings.reconnect_max_delay,
        )
        return random.uniform(0, ceiling)

    async def _connect_once(self) -> None:
        ssl_context: ssl.SSLContext | None = None
        if self._settings.gateway_url.startswith("wss://"):
            ssl_context = ssl.create_default_context()
            if not self._settings.verify_tls:
                log.warning("TLS verification disabled; expected only against a test gateway")
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

        log.info("connecting to gateway")
        async with websockets.connect(
            self._settings.gateway_url,
            additional_headers={"Authorization": f"Bearer {self._settings.core_token}"},
            ssl=ssl_context,
            ping_interval=self._settings.ping_interval,
            ping_timeout=self._settings.ping_timeout,
            max_size=None,  # binary audio frames exceed the default cap
            open_timeout=30,
        ) as connection:
            peer = RpcPeer(
                WebSocketTransport(connection),
                name="core",
                default_timeout=self._settings.rpc_call_timeout,
                on_binary=self.on_binary,
            )
            peer.register_all(self._handlers)
            self._peer = peer

            runner = asyncio.ensure_future(peer.run())
            try:
                await self._handshake(peer)
                self._connected.set()
                self._connected_since = datetime.now(timezone.utc)
                self.last_error = None
                self._wake_drainer.set()
                log.info("gateway connected")

                healthy_after = asyncio.ensure_future(
                    asyncio.sleep(self._settings.reconnect_healthy_after)
                )
                done, _ = await asyncio.wait(
                    {runner, healthy_after}, return_when=asyncio.FIRST_COMPLETED
                )
                if healthy_after in done:
                    # The connection stayed up long enough to count as healthy, so a later drop
                    # starts backoff from scratch instead of inheriting an old attempt count.
                    self._attempt = 0
                    await runner
                else:
                    healthy_after.cancel()
            finally:
                self._connected.clear()
                self._connected_since = None
                self._peer = None
                runner.cancel()
                await asyncio.gather(runner, return_exceptions=True)
                log.info("gateway disconnected")

    async def _handshake(self, peer: RpcPeer) -> None:
        last_received = await self._events.get_state("gateway_seq")
        result = await peer.call(
            methods.CORE_HELLO,
            {
                "instance_id": self._settings.instance_id,
                "protocol_version": PROTOCOL_VERSION,
                "last_received_seq": last_received,
                "capabilities": self._capabilities,
            },
            timeout=30,
        )
        parsed = methods.CoreHelloResult.model_validate(result)
        if parsed.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"gateway speaks protocol v{parsed.protocol_version}, core speaks v{PROTOCOL_VERSION}"
            )
        if parsed.last_received_seq:
            # The Gateway got further than our record shows: the events arrived, only the
            # acknowledgement was lost. Marking them sent avoids a duplicate replay.
            await self._events.acknowledge_through(parsed.last_received_seq)
        log.info(
            "handshake complete with gateway %s (acked through seq %d)",
            parsed.gateway_version,
            parsed.last_received_seq,
        )

    # ---- outbound -----------------------------------------------------

    async def send_event(
        self,
        method: str,
        params: dict[str, Any],
        *,
        delivery_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Durably enqueue an event for the Gateway and nudge the sender.

        Returns as soon as the event is committed. Actual delivery happens in the drain loop, so a
        caller never blocks on the network and a disconnect costs nothing.
        """
        event = await self._events.enqueue(
            method, params, delivery_id=delivery_id, user_id=user_id
        )
        if event is None:
            log.info("skipping duplicate delivery_id=%s method=%s", delivery_id, method)
            return
        self._wake_drainer.set()

    async def call(
        self, method: str, params: dict[str, Any], *, timeout: float | None = None
    ) -> Any:
        """Transient request. Raises if the link is down — used for things not worth queueing."""
        peer = self._peer
        if peer is None or not self._connected.is_set():
            raise ConnectionClosed("gateway link is not connected")
        return await peer.call(method, params, timeout=timeout)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        """Best-effort notification. Progress updates are advisory; dropping them is correct."""
        peer = self._peer
        if peer is None or not self._connected.is_set():
            return
        try:
            await peer.notify(method, params)
        except (ConnectionClosed, Exception):
            log.debug("dropping notification %s: link unavailable", method, exc_info=True)

    # ---- durable drain ------------------------------------------------

    async def _drain_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await asyncio.wait_for(self._wake_drainer.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            self._wake_drainer.clear()

            if self._stopping.is_set():
                break
            if not self._connected.is_set():
                continue

            try:
                await self._drain_pending()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("outbound drain failed")

    async def _drain_pending(self) -> None:
        while self._connected.is_set() and not self._stopping.is_set():
            batch = await self._events.pending(limit=50)
            if not batch:
                return

            for event in batch:
                if not self._connected.is_set():
                    return
                try:
                    await self.call(event.method, event.params)
                except ConnectionClosed:
                    return  # replayed after the next handshake
                except asyncio.TimeoutError:
                    # The Gateway may well have acted on it. Leaving the event pending means it is
                    # retried, and the Gateway deduplicates on delivery_id.
                    await self._events.mark_attempt_failed(event.seq, "timeout")
                    return
                except RpcError as exc:
                    if exc.code in _PERMANENT_FAILURES:
                        log.warning(
                            "dropping event seq=%d method=%s: %s",
                            event.seq, event.method, exc.message,
                        )
                        await self._events.drop(event.seq, exc.message)
                        continue
                    await self._events.mark_attempt_failed(event.seq, exc.message)
                    return
                else:
                    await self._events.mark_sent(event.seq)
                    await self._events.set_state("core_seq", event.seq)
