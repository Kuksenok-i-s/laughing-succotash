"""In-memory duplex transport, so protocol behaviour can be tested without a network."""

from __future__ import annotations

import asyncio

from .peer import Transport


class MemoryTransport(Transport):
    """One end of a bidirectional in-memory pipe."""

    def __init__(self, name: str = "memory") -> None:
        self.name = name
        self._inbox: asyncio.Queue[str | bytes | None] = asyncio.Queue()
        self._peer: MemoryTransport | None = None
        self._closed = False
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        # Set to drop outbound traffic without closing, to model a half-open connection where
        # writes appear to succeed but nothing arrives.
        self.blackhole = False

    @classmethod
    def pair(cls, left: str = "left", right: str = "right") -> tuple[MemoryTransport, MemoryTransport]:
        a, b = cls(left), cls(right)
        a._peer, b._peer = b, a
        return a, b

    async def send_text(self, data: str) -> None:
        if self._closed:
            raise ConnectionError(f"{self.name}: transport closed")
        self.sent_text.append(data)
        if self._peer is not None and not self.blackhole:
            await self._peer._inbox.put(data)

    async def send_bytes(self, data: bytes) -> None:
        if self._closed:
            raise ConnectionError(f"{self.name}: transport closed")
        self.sent_bytes.append(data)
        if self._peer is not None and not self.blackhole:
            await self._peer._inbox.put(data)

    async def receive(self) -> str | bytes | None:
        if self._closed and self._inbox.empty():
            return None
        return await self._inbox.get()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._inbox.put(None)
        if self._peer is not None and not self._peer._closed:
            self._peer._closed = True
            await self._peer._inbox.put(None)
