"""The Gateway-facing RPC surface.

The security property under test: the Gateway is authenticated as a service, not as a person, so
a ``user_id`` in a payload is a claim the Core checks against its own allowlist.
"""

from __future__ import annotations

import hashlib

import pytest
from pa_protocol import RpcError, decode_frame, errors, iter_frames, methods, new_ulid

from agent_core.audio.storage import UploadManager
from agent_core.rpc.handlers import CoreHandlers


class StubAssistant:
    def __init__(self) -> None:
        self.submitted: list = []
        self.audio_jobs: list = []
        self.cancelled: list[str] = []
        self.resets: list[str] = []

    async def submit(self, params):
        self.submitted.append(params)
        return methods.AcceptedResult(job_id="job-1")

    async def start_audio_job(self, upload):
        self.audio_jobs.append(upload)
        return "job-audio"

    async def cancel_job(self, job_id: str) -> bool:
        self.cancelled.append(job_id)
        return True

    async def reset_session(self, user_id: str) -> str:
        self.resets.append(user_id)
        return "conv-new"

    async def status(self) -> dict:
        return {"core": {"instance_id": "test"}, "jobs": {"queued": 0, "running": 0}}


class StubConfirmations:
    def __init__(self) -> None:
        self.resolved: list = []

    async def resolve(self, action_id, user_id, choice):
        self.resolved.append((action_id, user_id, choice))
        return "applied"


class StubScheduler:
    async def snapshot(self) -> dict:
        return {"state": "ready", "pending_reminders": 0}


@pytest.fixture
def handlers(settings, repos):
    uploads = UploadManager(
        repos.uploads, settings.resolved_temp_dir,
        max_bytes=settings.max_audio_bytes, idle_timeout=300.0,
    )
    assistant = StubAssistant()
    confirmations = StubConfirmations()
    return (
        CoreHandlers(settings, repos, assistant, uploads, confirmations, StubScheduler()),
        assistant,
        confirmations,
        uploads,
    )


def submit_payload(**overrides) -> dict:
    payload = {
        "request_id": new_ulid(),
        "user_id": "tg:1",
        "chat_id": 500,
        "message_id": 1,
        "kind": "text",
        "text": "привет",
    }
    payload.update(overrides)
    return payload


async def test_a_known_user_is_accepted(handlers, repos) -> None:
    core, assistant, _, _ = handlers

    result = await core.assistant_submit(submit_payload())

    assert result["status"] == "accepted"
    assert len(assistant.submitted) == 1
    # The chat is remembered so a reminder created now can be delivered hours later.
    assert await repos.conversations.chat_for("tg:1") == 500


async def test_an_unknown_user_is_refused(handlers) -> None:
    core, assistant, _, _ = handlers

    with pytest.raises(RpcError) as excinfo:
        await core.assistant_submit(submit_payload(user_id="tg:999999"))

    assert excinfo.value.code == errors.UNAUTHORIZED_USER
    assert assistant.submitted == []


async def test_a_malformed_payload_is_rejected_with_invalid_params(handlers) -> None:
    core, _, _, _ = handlers

    with pytest.raises(RpcError) as excinfo:
        await core.assistant_submit({"user_id": "tg:1"})

    assert excinfo.value.code == errors.INVALID_PARAMS


async def test_an_unexpected_field_is_rejected(handlers) -> None:
    """Payload models forbid extras so a protocol typo fails loudly instead of being ignored."""
    core, _, _, _ = handlers

    with pytest.raises(RpcError) as excinfo:
        await core.assistant_submit(submit_payload(nonsense="x"))

    assert excinfo.value.code == errors.INVALID_PARAMS


async def test_the_full_audio_upload_handshake(handlers) -> None:
    core, assistant, _, _ = handlers
    data = b"ogg data" * 100
    request_id = new_ulid()

    begin = await core.audio_begin(
        {
            "request_id": request_id,
            "user_id": "tg:1",
            "chat_id": 500,
            "message_id": 2,
            "filename": "voice.ogg",
            "content_type": "audio/ogg",
            "size": len(data),
            "duration_seconds": 30.0,
            "purpose": "assistant",
        }
    )

    for raw in iter_frames(begin["upload_id"], data, chunk_size=begin["chunk_size"]):
        await core.on_binary(decode_frame(raw))

    commit = await core.audio_commit(
        {
            "upload_id": begin["upload_id"],
            "sha256": hashlib.sha256(data).hexdigest(),
            "total_size": len(data),
        }
    )

    assert commit["status"] == "accepted"
    assert len(assistant.audio_jobs) == 1
    assert assistant.audio_jobs[0].purpose == "assistant"


async def test_audio_from_an_unknown_user_never_opens_an_upload(handlers) -> None:
    core, _, _, _ = handlers

    with pytest.raises(RpcError) as excinfo:
        await core.audio_begin(
            {
                "request_id": new_ulid(), "user_id": "tg:evil", "chat_id": 1, "message_id": 1,
                "filename": "x.ogg", "content_type": "audio/ogg", "size": 10,
            }
        )
    assert excinfo.value.code == errors.UNAUTHORIZED_USER


async def test_audio_that_is_too_long_is_refused_up_front(handlers, settings) -> None:
    """Rejecting at begin avoids transferring hundreds of megabytes we will not process."""
    core, _, _, _ = handlers

    with pytest.raises(RpcError) as excinfo:
        await core.audio_begin(
            {
                "request_id": new_ulid(), "user_id": "tg:1", "chat_id": 1, "message_id": 1,
                "filename": "long.mp3", "content_type": "audio/mpeg", "size": 1000,
                "duration_seconds": settings.max_audio_duration_seconds + 1,
            }
        )
    assert excinfo.value.code == errors.AUDIO_TOO_LONG


async def test_an_incomplete_upload_is_reported_as_such(handlers) -> None:
    core, assistant, _, _ = handlers
    begin = await core.audio_begin(
        {
            "request_id": new_ulid(), "user_id": "tg:1", "chat_id": 1, "message_id": 1,
            "filename": "v.ogg", "content_type": "audio/ogg", "size": 500,
        }
    )

    with pytest.raises(RpcError) as excinfo:
        await core.audio_commit(
            {"upload_id": begin["upload_id"], "sha256": "", "total_size": 500}
        )

    assert excinfo.value.code == errors.UPLOAD_INCOMPLETE
    assert assistant.audio_jobs == []


async def test_committing_an_unknown_upload_is_reported(handlers) -> None:
    core, _, _, _ = handlers

    with pytest.raises(RpcError) as excinfo:
        await core.audio_commit(
            {"upload_id": new_ulid(), "sha256": "", "total_size": 1}
        )
    assert excinfo.value.code == errors.UNKNOWN_UPLOAD


async def test_confirmation_resolution_is_forwarded(handlers) -> None:
    core, _, confirmations, _ = handlers

    result = await core.confirmation_resolve(
        {"action_id": "a-1", "user_id": "tg:1", "choice": "approve"}
    )

    assert result["status"] == "applied"
    assert confirmations.resolved == [("a-1", "tg:1", "approve")]


async def test_a_confirmation_for_an_unknown_user_is_refused(handlers) -> None:
    core, _, confirmations, _ = handlers

    with pytest.raises(RpcError):
        await core.confirmation_resolve(
            {"action_id": "a-1", "user_id": "tg:hacker", "choice": "approve"}
        )
    assert confirmations.resolved == []


async def test_session_reset_and_status(handlers) -> None:
    core, assistant, _, _ = handlers

    reset = await core.session_reset({"user_id": "tg:1", "request_id": new_ulid()})
    assert reset["conversation_id"] == "conv-new"
    assert assistant.resets == ["tg:1"]

    status = await core.status_get({})
    assert status["scheduler"]["state"] == "ready"
    assert "credentials" not in str(status).lower()


async def test_job_cancellation_is_forwarded(handlers) -> None:
    core, assistant, _, _ = handlers

    result = await core.job_cancel({"job_id": "job-7"})

    assert result["cancelled"] is True
    assert assistant.cancelled == ["job-7"]
