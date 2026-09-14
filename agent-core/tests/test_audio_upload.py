"""Binary audio upload: streaming, resume, and the failure modes that must not corrupt data."""

from __future__ import annotations

import hashlib

import pytest
from pa_protocol import AudioFrame, decode_frame, iter_frames

from agent_core.audio.storage import AudioUploadError, UploadManager


@pytest.fixture
def uploads(repos, settings) -> UploadManager:
    return UploadManager(
        repos.uploads,
        settings.resolved_temp_dir,
        max_bytes=1024 * 1024,
        idle_timeout=0.0,
    )


async def _begin(uploads: UploadManager, size: int, *, request_id="req-1", purpose="assistant"):
    return await uploads.begin(
        request_id=request_id,
        user_id="tg:1",
        chat_id=500,
        message_id=1,
        filename="voice.ogg",
        content_type="audio/ogg",
        size=size,
        duration_seconds=12.0,
        purpose=purpose,
    )


async def _send(uploads: UploadManager, upload_id: str, data: bytes, chunk_size=64) -> None:
    for raw in iter_frames(upload_id, data, chunk_size=chunk_size):
        await uploads.handle_frame(decode_frame(raw))


async def test_a_streamed_file_arrives_intact(uploads) -> None:
    data = bytes(range(256)) * 40
    upload, offset = await _begin(uploads, len(data))
    assert offset == 0

    await _send(uploads, upload.upload_id, data)
    committed = await uploads.commit(
        upload.upload_id, sha256=hashlib.sha256(data).hexdigest(), total_size=len(data)
    )

    assert committed.temp_path.read_bytes() == data
    assert committed.status == "complete"


async def test_a_replayed_frame_is_ignored(uploads) -> None:
    """A reconnect can resend frames; the file must not gain duplicate bytes."""
    data = b"0123456789" * 10
    upload, _ = await _begin(uploads, len(data))

    frames = [decode_frame(raw) for raw in iter_frames(upload.upload_id, data, chunk_size=25)]
    for frame in frames:
        await uploads.handle_frame(frame)
    for frame in frames[:2]:
        await uploads.handle_frame(frame)

    committed = await uploads.commit(
        upload.upload_id, sha256=hashlib.sha256(data).hexdigest(), total_size=len(data)
    )
    assert committed.temp_path.read_bytes() == data


async def test_a_gap_in_the_stream_fails_the_upload(uploads) -> None:
    upload, _ = await _begin(uploads, 100)
    await uploads.handle_frame(AudioFrame(upload.upload_id, 0, b"abc"))
    # Skips ahead: the missing bytes would be silently zero-filled if this were accepted.
    await uploads.handle_frame(AudioFrame(upload.upload_id, 50, b"xyz"))

    with pytest.raises(AudioUploadError):
        await uploads.commit(upload.upload_id, sha256="", total_size=100)


async def test_a_wrong_checksum_is_refused(uploads) -> None:
    """Transcribing corrupted audio would produce confident nonsense."""
    data = b"real audio bytes"
    upload, _ = await _begin(uploads, len(data))
    await _send(uploads, upload.upload_id, data)

    with pytest.raises(AudioUploadError) as excinfo:
        await uploads.commit(upload.upload_id, sha256="0" * 64, total_size=len(data))
    assert excinfo.value.code == "checksum_mismatch"
    assert not upload.temp_path.exists()


async def test_a_short_upload_is_refused(uploads) -> None:
    data = b"only half"
    upload, _ = await _begin(uploads, 100)
    await _send(uploads, upload.upload_id, data)

    with pytest.raises(AudioUploadError) as excinfo:
        await uploads.commit(upload.upload_id, sha256="", total_size=100)
    assert excinfo.value.code == "size_mismatch"


async def test_an_oversized_file_is_rejected_before_any_bytes_move(uploads) -> None:
    with pytest.raises(AudioUploadError) as excinfo:
        await _begin(uploads, 50 * 1024 * 1024)
    assert excinfo.value.code == "too_large"


async def test_an_interrupted_upload_can_be_resumed(uploads, repos) -> None:
    data = b"a" * 200
    upload, _ = await _begin(uploads, len(data))
    await _send(uploads, upload.upload_id, data[:80], chunk_size=40)

    # The Gateway reconnects and starts over with the same request_id.
    resumed, offset = await _begin(uploads, len(data))
    assert resumed.upload_id == upload.upload_id
    assert offset == 80

    await uploads.handle_frame(AudioFrame(upload.upload_id, 80, data[80:], final=True))
    committed = await uploads.commit(
        upload.upload_id, sha256=hashlib.sha256(data).hexdigest(), total_size=len(data)
    )
    assert committed.temp_path.read_bytes() == data


async def test_the_recording_is_deleted_once_consumed(uploads) -> None:
    data = b"secret meeting audio"
    upload, _ = await _begin(uploads, len(data))
    await _send(uploads, upload.upload_id, data)
    committed = await uploads.commit(
        upload.upload_id, sha256=hashlib.sha256(data).hexdigest(), total_size=len(data)
    )

    await uploads.release(committed)
    assert not committed.temp_path.exists()


async def test_stale_uploads_are_swept(uploads, repos) -> None:
    upload, _ = await _begin(uploads, 500)
    await uploads.handle_frame(AudioFrame(upload.upload_id, 0, b"partial"))

    assert await uploads.sweep_stale() == 1
    assert (await repos.uploads.get(upload.upload_id)).status == "expired"
    assert not upload.temp_path.exists()


async def test_frames_for_an_unknown_upload_are_dropped(uploads) -> None:
    """A frame after an abort must not resurrect the upload or crash the receive loop."""
    from pa_protocol import new_ulid

    await uploads.handle_frame(AudioFrame(new_ulid(), 0, b"orphan"))
