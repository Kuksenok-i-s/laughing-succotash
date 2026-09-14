"""Receives audio uploads as binary frames and streams them to a temporary file.

A multi-hour recording can be hundreds of megabytes, so nothing is buffered whole in memory and
nothing is Base64-encoded. Frames carry their own offset, which lets an interrupted upload resume
and makes a duplicated frame after a reconnect detectable rather than corrupting.

An upload is only usable once ``commit`` has verified length and digest. If the connection drops
mid-transfer the Core simply never sees a commit, the partial file is swept, and the Gateway
retries — the audio is never half-processed.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import timedelta
from pathlib import Path

from pa_protocol import AudioFrame

from ..storage.database import utcnow
from ..storage.repositories import Upload, UploadRepository

log = logging.getLogger(__name__)


class AudioUploadError(RuntimeError):
    def __init__(self, message: str, *, code: str = "upload_failed") -> None:
        super().__init__(message)
        self.code = code


class UploadSink:
    """One in-flight upload: an open file handle plus the byte offset reached so far."""

    def __init__(self, upload: Upload, path: Path, max_bytes: int) -> None:
        self.upload = upload
        self.path = path
        self.max_bytes = max_bytes
        self.received = 0
        self.final_seen = False
        self._handle = path.open("wb")
        self._lock = asyncio.Lock()

    async def write(self, frame: AudioFrame) -> int:
        async with self._lock:
            if frame.offset < self.received:
                # Already have these bytes: a resend after a reconnect. Ignoring it keeps the
                # file correct without needing the sender to know what we got.
                log.debug(
                    "ignoring replayed frame for %s at offset %d (have %d)",
                    self.upload.upload_id, frame.offset, self.received,
                )
                if frame.final:
                    self.final_seen = True
                return self.received
            if frame.offset > self.received:
                raise AudioUploadError(
                    f"gap in upload {self.upload.upload_id}: expected offset {self.received}, "
                    f"got {frame.offset}",
                    code="upload_gap",
                )
            if self.received + len(frame.payload) > self.max_bytes:
                raise AudioUploadError("audio exceeds the configured size limit", code="too_large")

            await asyncio.to_thread(self._handle.write, frame.payload)
            self.received += len(frame.payload)
            if frame.final:
                self.final_seen = True
            return self.received

    async def close(self) -> None:
        async with self._lock:
            if not self._handle.closed:
                await asyncio.to_thread(self._handle.close)

    async def discard(self) -> None:
        await self.close()
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            log.debug("could not remove partial upload %s", self.path, exc_info=True)


class UploadManager:
    def __init__(
        self,
        uploads: UploadRepository,
        temp_dir: Path,
        *,
        max_bytes: int,
        idle_timeout: float = 300.0,
    ) -> None:
        self._uploads = uploads
        self._temp_dir = temp_dir
        self._max_bytes = max_bytes
        self._idle_timeout = idle_timeout
        self._sinks: dict[str, UploadSink] = {}

    async def begin(
        self,
        *,
        request_id: str,
        user_id: str,
        chat_id: int,
        message_id: int,
        filename: str,
        content_type: str,
        size: int,
        duration_seconds: float | None,
        purpose: str,
        caption: str | None = None,
        attribution: dict | None = None,
        album_id: str | None = None,
        part_index: int | None = None,
        part_count: int | None = None,
    ) -> tuple[Upload, int]:
        """Open an upload, or resume the one already open for this ``request_id``.

        Returns the upload and the offset the sender should continue from.
        """
        if size > self._max_bytes:
            raise AudioUploadError(
                f"file is {size // 1024 // 1024} MB, over the limit", code="too_large"
            )

        existing = await self._uploads.find_open_by_request(request_id)
        if existing is not None:
            sink = self._sinks.get(existing.upload_id)
            if sink is not None:
                return existing, sink.received
            # The record survived a restart but the open file handle did not. Start it over
            # rather than trust a partial file whose tail may have been lost to buffering.
            await self._uploads.set_status(existing.upload_id, "aborted")

        self._temp_dir.mkdir(parents=True, exist_ok=True)
        upload = await self._uploads.create(
            request_id=request_id,
            user_id=user_id,
            chat_id=chat_id,
            message_id=message_id,
            filename=_safe_name(filename),
            content_type=content_type,
            declared_size=size,
            temp_path=self._temp_dir / f"{request_id}-{_safe_name(filename)}",
            duration_seconds=duration_seconds,
            purpose=purpose,
            caption=caption,
            attribution=attribution,
            album_id=album_id,
            part_index=part_index,
            part_count=part_count,
        )
        self._sinks[upload.upload_id] = UploadSink(upload, upload.temp_path, self._max_bytes)
        log.info(
            "upload %s open for %s (%d bytes, purpose=%s album=%s)",
            upload.upload_id, user_id, size, purpose, album_id or "-",
        )
        return upload, 0

    async def handle_frame(self, frame: AudioFrame) -> None:
        sink = self._sinks.get(frame.upload_id)
        if sink is None:
            log.warning("binary frame for unknown upload %s", frame.upload_id)
            return
        try:
            received = await sink.write(frame)
        except AudioUploadError as exc:
            log.warning("upload %s failed: %s", frame.upload_id, exc)
            await self._fail(frame.upload_id)
            return
        # Persisted so a resume after reconnect knows where it got to.
        await self._uploads.record_progress(frame.upload_id, received)

    async def commit(self, upload_id: str, *, sha256: str, total_size: int) -> Upload:
        sink = self._sinks.get(upload_id)
        if sink is None:
            raise AudioUploadError(f"unknown upload {upload_id}", code="unknown_upload")

        await sink.close()
        actual_size = sink.path.stat().st_size

        if actual_size != total_size:
            await self._fail(upload_id)
            raise AudioUploadError(
                f"upload {upload_id} is {actual_size} bytes, expected {total_size}",
                code="size_mismatch",
            )

        if sha256:
            digest = await asyncio.to_thread(_digest, sink.path)
            if digest != sha256:
                # A wrong digest means the bytes are not what the Gateway read from Telegram.
                # Transcribing them anyway would produce confident nonsense.
                await self._fail(upload_id)
                raise AudioUploadError(
                    f"checksum mismatch for upload {upload_id}", code="checksum_mismatch"
                )

        await self._uploads.record_progress(upload_id, actual_size)
        await self._uploads.set_status(upload_id, "complete")
        self._sinks.pop(upload_id, None)

        upload = await self._uploads.get(upload_id)
        if upload is None:
            raise AudioUploadError(f"upload {upload_id} vanished", code="unknown_upload")
        log.info("upload %s committed (%d bytes)", upload_id, actual_size)
        return upload

    async def abort(self, upload_id: str, reason: str | None = None) -> None:
        log.info("aborting upload %s: %s", upload_id, reason or "no reason given")
        await self._fail(upload_id, status="aborted")

    async def _fail(self, upload_id: str, status: str = "failed") -> None:
        sink = self._sinks.pop(upload_id, None)
        if sink is not None:
            await sink.discard()
        await self._uploads.set_status(upload_id, status)

    async def release(self, upload: Upload) -> None:
        """Delete the audio once it has been transcribed.

        Recordings are not kept: the transcript is the useful artefact and the original is the
        sensitive one.
        """
        try:
            upload.temp_path.unlink(missing_ok=True)
        except OSError:
            log.debug("could not delete %s", upload.temp_path, exc_info=True)
        await self._uploads.set_status(upload.upload_id, "consumed")

    async def list_committed_album(self, album_id: str) -> list[Upload]:
        return await self._uploads.list_committed_album(album_id)

    async def sweep_stale(self) -> int:
        """Drop uploads that stopped mid-transfer, freeing their disk space."""
        cutoff = (utcnow() - timedelta(seconds=self._idle_timeout)).isoformat()
        stale = await self._uploads.stale_open(cutoff)
        for upload in stale:
            await self._fail(upload.upload_id, status="expired")
        if stale:
            log.info("swept %d stale uploads", len(stale))
        return len(stale)

    async def shutdown(self) -> None:
        for upload_id in list(self._sinks):
            await self._fail(upload_id, status="interrupted")


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _safe_name(filename: str) -> str:
    """Strip anything that could escape the temp directory or confuse ffmpeg."""
    cleaned = "".join(c for c in Path(filename).name if c.isalnum() or c in "._-")
    return cleaned[:64] or "audio.bin"
