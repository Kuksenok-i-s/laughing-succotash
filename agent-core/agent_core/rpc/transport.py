"""Adapts a ``websockets`` client connection to the shared :class:`Transport` protocol."""

from __future__ import annotations

import logging

import websockets
from pa_protocol import Transport

log = logging.getLogger(__name__)


class WebSocketTransport(Transport):
    def __init__(self, connection: websockets.ClientConnection) -> None:
        self._connection = connection
        self._closed = False

    async def send_text(self, data: str) -> None:
        await self._connection.send(data)

    async def send_bytes(self, data: bytes) -> None:
        await self._connection.send(data)

    async def receive(self) -> str | bytes | None:
        try:
            return await self._connection.recv()
        except websockets.ConnectionClosed:
            return None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._connection.close()
        except Exception:
            log.debug("error while closing websocket", exc_info=True)
