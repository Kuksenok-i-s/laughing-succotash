from __future__ import annotations

import asyncio

import pytest

from pa_protocol import AudioFrame, RpcError, RpcPeer, encode_frame, new_ulid
from pa_protocol.errors import INVALID_PARAMS, METHOD_NOT_FOUND
from pa_protocol.peer import ConnectionClosed
from pa_protocol.testing import MemoryTransport

pytestmark = pytest.mark.asyncio


class Pair:
    """Two connected peers with their receive loops running."""

    def __init__(self, **kwargs):
        left_t, right_t = MemoryTransport.pair()
        self.left = RpcPeer(left_t, name="left", default_timeout=2.0, **kwargs)
        self.right = RpcPeer(right_t, name="right", default_timeout=2.0)
        self._tasks: list[asyncio.Task] = []

    async def __aenter__(self):
        self._tasks = [asyncio.ensure_future(self.left.run()), asyncio.ensure_future(self.right.run())]
        return self

    async def __aexit__(self, *exc):
        await self.left.close()
        await self.right.close()
        await asyncio.gather(*self._tasks, return_exceptions=True)


async def test_request_response_roundtrip():
    async with Pair() as pair:
        pair.right.register("math.add", lambda p: _ok(p["a"] + p["b"]))
        assert await pair.left.call("math.add", {"a": 2, "b": 40}) == {"value": 42}


async def _ok(value):
    return {"value": value} if not isinstance(value, dict) else value


async def test_unknown_method_returns_method_not_found():
    async with Pair() as pair:
        with pytest.raises(RpcError) as exc:
            await pair.left.call("does.not.exist", {})
        assert exc.value.code == METHOD_NOT_FOUND


async def test_handler_rpc_error_propagates_code_and_data():
    async def boom(_params):
        raise RpcError(INVALID_PARAMS, "invalid_params", {"field": "chat_id"})

    async with Pair() as pair:
        pair.right.register("x", boom)
        with pytest.raises(RpcError) as exc:
            await pair.left.call("x", {})
        assert exc.value.code == INVALID_PARAMS
        assert exc.value.data == {"field": "chat_id"}


async def test_unexpected_handler_exception_becomes_internal_error_without_leaking_detail():
    async def boom(_params):
        raise ValueError("/Users/secret/path leaked in message")

    async with Pair() as pair:
        pair.right.register("x", boom)
        with pytest.raises(RpcError) as exc:
            await pair.left.call("x", {})
        assert exc.value.code == -32603
        assert "secret" not in str(exc.value.data)


async def test_notifications_get_no_response():
    seen = asyncio.Event()

    async def handler(params):
        seen.set()
        return {"ignored": True}

    async with Pair() as pair:
        pair.right.register("ping", handler)
        await pair.left.notify("ping", {})
        await asyncio.wait_for(seen.wait(), 1)


async def test_both_directions_can_originate_calls():
    async with Pair() as pair:
        pair.right.register("to_right", lambda p: _ok("right"))
        pair.left.register("to_left", lambda p: _ok("left"))
        a, b = await asyncio.gather(
            pair.left.call("to_right", {}),
            pair.right.call("to_left", {}),
        )
        assert a["value"] == "right"
        assert b["value"] == "left"


async def test_concurrent_calls_are_correlated_not_serialized():
    async def slow(params):
        await asyncio.sleep(0.05 * params["n"])
        return {"n": params["n"]}

    async with Pair() as pair:
        pair.right.register("slow", slow)
        results = await asyncio.gather(*(pair.left.call("slow", {"n": n}) for n in range(6)))
        assert [r["n"] for r in results] == list(range(6))


async def test_timeout_raises_and_does_not_poison_later_calls():
    async def hang(_params):
        await asyncio.sleep(10)

    async with Pair() as pair:
        pair.right.register("hang", hang)
        pair.right.register("fast", lambda p: _ok("ok"))
        with pytest.raises(asyncio.TimeoutError):
            await pair.left.call("hang", {}, timeout=0.1)
        assert (await pair.left.call("fast", {}))["value"] == "ok"


async def test_pending_calls_fail_when_transport_closes():
    async def hang(_params):
        await asyncio.sleep(10)

    pair = Pair()
    async with pair:
        pair.right.register("hang", hang)
        call = asyncio.ensure_future(pair.left.call("hang", {}, timeout=5))
        await asyncio.sleep(0.05)
        await pair.left.close()
        with pytest.raises(ConnectionClosed):
            await call


async def test_malformed_json_gets_a_parse_error_and_the_peer_survives():
    left_t, right_t = MemoryTransport.pair()
    left = RpcPeer(left_t, name="left", default_timeout=2.0)
    right = RpcPeer(right_t, name="right", default_timeout=2.0)
    right.register("ok", lambda p: _ok("still alive"))
    tasks = [asyncio.ensure_future(left.run()), asyncio.ensure_future(right.run())]
    try:
        await left_t.send_text("{not json at all")
        await asyncio.sleep(0.05)
        assert (await left.call("ok", {}))["value"] == "still alive"
    finally:
        await left.close()
        await right.close()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_binary_frames_route_to_the_binary_handler():
    received: list[AudioFrame] = []

    left_t, right_t = MemoryTransport.pair()
    left = RpcPeer(left_t, name="left")
    right = RpcPeer(right_t, name="right", on_binary=lambda f: _collect(received, f))
    tasks = [asyncio.ensure_future(left.run()), asyncio.ensure_future(right.run())]
    try:
        upload_id = new_ulid()
        await left.send_binary(encode_frame(AudioFrame(upload_id, 0, b"abc")))
        await left.send_binary(encode_frame(AudioFrame(upload_id, 3, b"def", final=True)))
        await asyncio.sleep(0.05)
        assert b"".join(f.payload for f in received) == b"abcdef"
        assert received[-1].final
    finally:
        await left.close()
        await right.close()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _collect(sink, frame):
    sink.append(frame)


async def test_corrupt_binary_frame_is_dropped_without_killing_the_connection():
    received: list[AudioFrame] = []
    left_t, right_t = MemoryTransport.pair()
    left = RpcPeer(left_t, name="left", default_timeout=2.0)
    right = RpcPeer(right_t, name="right", on_binary=lambda f: _collect(received, f))
    right.register("ok", lambda p: _ok("alive"))
    tasks = [asyncio.ensure_future(left.run()), asyncio.ensure_future(right.run())]
    try:
        await left.send_binary(b"garbage-not-a-frame")
        await left.send_binary(encode_frame(AudioFrame(new_ulid(), 0, b"good", final=True)))
        await asyncio.sleep(0.05)
        assert [f.payload for f in received] == [b"good"]
        assert (await left.call("ok", {}))["value"] == "alive"
    finally:
        await left.close()
        await right.close()
        await asyncio.gather(*tasks, return_exceptions=True)
