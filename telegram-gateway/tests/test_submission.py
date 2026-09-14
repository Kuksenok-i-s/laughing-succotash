"""Durable submission: what happens to the user's message while the Core is unreachable."""

from __future__ import annotations

import asyncio
import hashlib

import pytest
from pa_protocol import RpcError, decode_frame, errors, methods, new_ulid

from telegram_gateway.delivery.service import SubmissionService


class FakeCore:
    """A Core that can be offline, refuse, or accept — and records what it received."""

    def __init__(self) -> None:
        self.connected = True
        self.calls: list[tuple[str, dict]] = []
        self.frames: list[bytes] = []
        self.raise_on: dict[str, Exception] = {}
        self.received: dict[str, bytearray] = {}

    async def call(self, method: str, params: dict, *, timeout=None):
        self.calls.append((method, params))
        error = self.raise_on.pop(method, None)
        if error is not None:
            raise error
        if method == methods.AUDIO_BEGIN:
            return methods.dump(
                methods.AudioBeginResult(upload_id=new_ulid(), chunk_size=64, resume_offset=0)
            )
        if method == methods.ASSISTANT_SUBMIT:
            return {"job_id": "job-1", "status": "accepted"}
        return {}

    async def send_binary(self, frame: bytes) -> None:
        self.frames.append(frame)
        decoded = decode_frame(frame)
        buffer = self.received.setdefault(decoded.upload_id, bytearray())
        buffer[decoded.offset : decoded.offset + len(decoded.payload)] = decoded.payload


@pytest.fixture
def core() -> FakeCore:
    return FakeCore()


@pytest.fixture
def submissions(core, store, settings) -> SubmissionService:
    return SubmissionService(core, store, settings)


async def save_text(store, text: str = "привет") -> str:
    request_id = new_ulid()
    await store.save_request(
        request_id=request_id,
        user_id="tg:1",
        chat_id=500,
        message_id=1,
        kind="text",
        payload=methods.dump(
            methods.AssistantSubmitParams(
                request_id=request_id, user_id="tg:1", chat_id=500, message_id=1, text=text
            )
        ),
    )
    return request_id


async def test_a_queued_message_is_submitted(submissions, core, store) -> None:
    await save_text(store, "что у меня завтра?")

    await submissions._drain_requests()  # noqa: SLF001

    assert [method for method, _ in core.calls] == [methods.ASSISTANT_SUBMIT]
    assert await store.pending_request_count() == 0


async def test_a_message_survives_the_core_being_offline(submissions, core, store) -> None:
    """The Definition of Done case: accepted by the Gateway, processed after reconnect."""
    await save_text(store, "работай когда сможешь")
    core.connected = False

    await submissions._drain_requests()  # noqa: SLF001
    assert core.calls == []
    assert await store.pending_request_count() == 1

    core.connected = True
    await submissions._drain_requests()  # noqa: SLF001

    assert len(core.calls) == 1
    assert await store.pending_request_count() == 0


async def test_a_timeout_leaves_the_request_queued_for_retry(submissions, core, store) -> None:
    """A timeout is ambiguous, and assistant.submit is idempotent on request_id."""
    request_id = await save_text(store)
    core.raise_on[methods.ASSISTANT_SUBMIT] = asyncio.TimeoutError()

    await submissions._drain_requests()  # noqa: SLF001
    assert await store.pending_request_count() == 1

    await submissions._drain_requests()  # noqa: SLF001
    assert await store.pending_request_count() == 0
    assert [params["request_id"] for _, params in core.calls] == [request_id, request_id]


async def test_a_permanently_refused_request_is_dropped(submissions, core, store) -> None:
    """Retrying an unauthorized user for ever would block everything behind it."""
    await save_text(store)
    core.raise_on[methods.ASSISTANT_SUBMIT] = RpcError(
        errors.UNAUTHORIZED_USER, "unauthorized_user"
    )

    await submissions._drain_requests()  # noqa: SLF001

    assert await store.pending_request_count() == 0


async def test_requests_are_submitted_in_arrival_order(submissions, core, store) -> None:
    first = await save_text(store, "первое")
    second = await save_text(store, "второе")

    await submissions._drain_requests()  # noqa: SLF001

    assert [params["request_id"] for _, params in core.calls] == [first, second]


# ---- audio ----------------------------------------------------------------


async def test_audio_is_streamed_as_binary_frames_and_the_file_removed(
    submissions, core, store, settings
) -> None:
    data = bytes(range(256)) * 20
    path = settings.resolved_temp_dir / "voice.ogg"
    path.write_bytes(data)
    request_id = new_ulid()

    await store.save_upload(
        request_id=request_id, user_id="tg:1", chat_id=500, message_id=1,
        file_path=path, filename="voice.ogg", content_type="audio/ogg",
        size=len(data), sha256=hashlib.sha256(data).hexdigest(),
        duration_seconds=30.0, purpose="assistant",
    )

    await submissions._drain_uploads()  # noqa: SLF001

    methods_called = [method for method, _ in core.calls]
    assert methods_called == [methods.AUDIO_BEGIN, methods.AUDIO_COMMIT]
    # The bytes arrived over the binary channel, not as Base64 inside JSON.
    assert bytes(next(iter(core.received.values()))) == data
    # The Gateway keeps no audio once the Core has it.
    assert not path.exists()


async def test_a_vanished_file_does_not_wedge_the_queue(submissions, core, store, settings):
    request_id = new_ulid()
    await store.save_upload(
        request_id=request_id, user_id="tg:1", chat_id=500, message_id=1,
        file_path=settings.resolved_temp_dir / "gone.ogg", filename="gone.ogg",
        content_type="audio/ogg", size=10, sha256="x",
    )

    await submissions._drain_uploads()  # noqa: SLF001

    assert (await store.get_upload(request_id)).attempts >= 1
    assert core.calls == []


async def test_an_upload_is_abandoned_after_too_many_attempts(
    submissions, core, store, settings
) -> None:
    path = settings.resolved_temp_dir / "flaky.ogg"
    path.write_bytes(b"data")
    request_id = new_ulid()
    await store.save_upload(
        request_id=request_id, user_id="tg:1", chat_id=500, message_id=1,
        file_path=path, filename="flaky.ogg", content_type="audio/ogg",
        size=4, sha256=hashlib.sha256(b"data").hexdigest(),
    )

    for _ in range(settings.delivery_max_attempts + 1):
        core.raise_on[methods.AUDIO_BEGIN] = RpcError(errors.NOT_READY, "not_ready")
        await submissions._drain_uploads()  # noqa: SLF001

    assert await store.pending_uploads() == []
    assert not path.exists()
