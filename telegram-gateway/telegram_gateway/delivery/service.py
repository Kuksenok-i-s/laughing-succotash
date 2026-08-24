"""Submits queued Telegram input to the Core.

Everything the user sends is written to SQLite first and submitted second. If the Core is offline
the request simply waits, and the same loop that retries it also drains the backlog after
reconnect — there is no separate recovery path to get wrong.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from pa_protocol import RpcError, errors, iter_frames, methods

from ..storage.models import GatewayStore, PendingUpload

log = logging.getLogger(__name__)

# Errors that will never succeed on retry. Keeping the request queued forever would block the
# backlog behind a request that cannot make progress.
_PERMANENT = {
    errors.UNAUTHORIZED_USER,
    errors.AUDIO_TOO_LARGE,
    errors.AUDIO_TOO_LONG,
    errors.INVALID_PARAMS,
}


class SubmissionService:
    def __init__(self, core, store: GatewayStore, settings, bot=None) -> None:
        self._core = core
        self._store = store
        self._settings = settings
        self._bot = bot
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        self._task = asyncio.ensure_future(self._loop())

    async def stop(self) -> None:
        self._stopping.set()
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def nudge(self) -> None:
        self._wake.set()

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()

            if self._stopping.is_set():
                break
            if not self._core.connected:
                continue

            try:
                await self._drain_uploads()
                await self._drain_requests()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("submission drain failed")

    # ---- text / command requests ----------------------------------------

    async def _drain_requests(self) -> None:
        for request in await self._store.pending_requests():
            if not self._core.connected or self._stopping.is_set():
                return
            try:
                result = await self._core.call(
                    methods.ASSISTANT_SUBMIT,
                    request.payload,
                    timeout=self._settings.submit_timeout,
                )
            except RpcError as exc:
                if exc.code in _PERMANENT:
                    log.warning(
                        "dropping request %s: %s", request.request_id, exc.message
                    )
                    await self._store.mark_request_submitted(request.request_id, None)
                    continue
                await self._store.mark_request_attempt_failed(request.request_id, exc.message)
                return
            except (asyncio.TimeoutError, Exception) as exc:
                # A timeout is ambiguous: the Core may have accepted it. Retrying is safe because
                # assistant.submit is idempotent on request_id.
                await self._store.mark_request_attempt_failed(
                    request.request_id, f"{type(exc).__name__}"
                )
                return
            else:
                await self._store.mark_request_submitted(
                    request.request_id, (result or {}).get("job_id")
                )

    # ---- audio uploads ----------------------------------------------------

    async def _drain_uploads(self) -> None:
        for upload in await self._store.pending_uploads():
            if not self._core.connected or self._stopping.is_set():
                return
            attempts = await self._store.mark_upload_attempt(upload.request_id)
            if attempts > self._settings.delivery_max_attempts:
                log.warning("giving up on upload %s after %d attempts",
                            upload.request_id, attempts)
                await self._store.set_upload_status(upload.request_id, "failed")
                self._cleanup_file(upload.file_path)
                continue
            try:
                await self._upload_one(upload)
            except RpcError as exc:
                if exc.code in _PERMANENT:
                    log.warning("upload %s rejected permanently: %s",
                                upload.request_id, exc.message)
                    await self._store.set_upload_status(upload.request_id, "failed")
                    self._cleanup_file(upload.file_path)
                    await self._notify_upload_failure(upload, exc)
                    continue
                log.info("upload %s will retry: %s", upload.request_id, exc.message)
                # Back to 'pending', or the retry loop would never pick it up again: the
                # in-progress marker set by _upload_one excludes it from the pending query.
                await self._store.set_upload_status(upload.request_id, "pending")
                return
            except Exception:
                log.exception("upload %s failed", upload.request_id)
                await self._store.set_upload_status(upload.request_id, "pending")
                return

    async def _upload_one(self, upload: PendingUpload) -> None:
        if not upload.file_path.exists():
            log.warning("upload file vanished: %s", upload.file_path)
            await self._store.set_upload_status(upload.request_id, "failed")
            return

        await self._store.set_upload_status(upload.request_id, "uploading")

        begin = await self._core.call(
            methods.AUDIO_BEGIN,
            methods.dump(
                methods.AudioBeginParams(
                    request_id=upload.request_id,
                    user_id=upload.user_id,
                    chat_id=upload.chat_id,
                    message_id=upload.message_id,
                    filename=upload.filename,
                    content_type=upload.content_type or "application/octet-stream",
                    size=upload.size,
                    duration_seconds=upload.duration_seconds,
                    purpose=upload.purpose,
                )
            ),
        )
        parsed = methods.AudioBeginResult.model_validate(begin)

        # The Core may have kept part of an interrupted upload; resume rather than resend.
        await self._stream_file(
            upload.file_path,
            parsed.upload_id,
            chunk_size=parsed.chunk_size or self._settings.upload_chunk_size,
            start_offset=parsed.resume_offset,
        )

        await self._core.call(
            methods.AUDIO_COMMIT,
            methods.dump(
                methods.AudioCommitParams(
                    upload_id=parsed.upload_id,
                    sha256=upload.sha256 or "",
                    total_size=upload.size,
                )
            ),
            timeout=120,
        )

        await self._store.set_upload_status(upload.request_id, "done")
        # The Gateway keeps no audio: once the Core has it, the temporary copy is deleted.
        self._cleanup_file(upload.file_path)

    async def _stream_file(
        self, path: Path, upload_id: str, *, chunk_size: int, start_offset: int = 0
    ) -> None:
        """Stream the file in binary frames without ever holding it whole in memory."""
        loop = asyncio.get_running_loop()
        offset = start_offset
        total = path.stat().st_size

        with path.open("rb") as handle:
            if start_offset:
                handle.seek(start_offset)
            while True:
                chunk = await loop.run_in_executor(None, handle.read, chunk_size)
                if not chunk:
                    break
                offset_now = offset
                offset += len(chunk)
                for frame in iter_frames(
                    upload_id, chunk, chunk_size=chunk_size, start_offset=offset_now
                ):
                    await self._core.send_binary(frame)
                if offset >= total:
                    break

        if offset == start_offset:
            # Zero-byte file: still send the terminator so the Core sees a complete upload.
            for frame in iter_frames(upload_id, b"", start_offset=start_offset):
                await self._core.send_binary(frame)

    async def _notify_upload_failure(self, upload: PendingUpload, exc: RpcError) -> None:
        if self._bot is None:
            return
        from ..telegram.formatting import describe_error

        try:
            await self._bot.send_message(upload.chat_id, describe_error(exc.message))
        except Exception:
            log.debug("could not notify user of upload failure", exc_info=True)

    @staticmethod
    def _cleanup_file(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.debug("could not delete temp file %s", path, exc_info=True)
