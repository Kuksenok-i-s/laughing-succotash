"""Transcription on the GPU host, over its HTTP API.

Replaces an SSH pipeline that copied a script, launched it with ``nohup``, guessed whether it was
alive with ``pgrep`` and pulled ``progress.json`` over ``scp`` every thirty seconds. Here the job is
an upload, the progress is a number in a JSON body, and a dead host is a connection error rather
than a silence — which is what ``FallbackSTT`` needs to decide to use the CPU.

Everything runs on the event loop, so the progress hook is called where callers may schedule
coroutines from it, without the thread hand-off ``faster_whisper.py`` needs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import aiohttp
from pa_protocol import new_ulid

from .base import (
    NoticeHook,
    ProgressHook,
    SpeechToText,
    SttError,
    TranscriptionResult,
    TranscriptSegment,
)

log = logging.getLogger(__name__)


class GpuServiceSTT(SpeechToText):
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        language: str = "auto",
        beam_size: int = 5,
        poll_interval: float = 2.0,
        request_timeout: float = 30.0,
        upload_timeout: float = 900.0,
        # A warm large-v3 runs faster than real time, so a job that has not moved a single percent
        # in this long is not slow, it is stuck; failing lets the CPU fallback take over.
        stall_timeout: float = 900.0,
        max_concurrent: int = 1,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._language = "auto" if language in ("", None) else language
        self._beam_size = beam_size
        self._poll_interval = poll_interval
        self._request_timeout = request_timeout
        self._upload_timeout = upload_timeout
        self._stall_timeout = stall_timeout
        self._slots = asyncio.Semaphore(max(1, max_concurrent))
        self._session: aiohttp.ClientSession | None = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def model_name(self) -> str:
        return "gpu-service/faster-whisper-large-v3"

    async def warmup(self) -> None:
        """Confirm the service answers. A model still loading is fine; an absent host is not.

        The service listens before the weights are in memory, and refusing to be ready during those
        twenty seconds would send the whole process to the CPU for as long as it lives.
        """
        try:
            health = await self._request("GET", "/health", authorized=False)
        except Exception as exc:
            raise SttError(f"transcription service unreachable: {exc}") from exc
        self._ready = True
        log.info(
            "transcription service ready at %s (model=%s loaded=%s queued=%s)",
            self._base,
            health.get("model"),
            health.get("model_loaded"),
            health.get("queued"),
        )

    async def close(self) -> None:
        self._ready = False
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def transcribe(
        self,
        audio_path: Path,
        *,
        on_progress: ProgressHook | None = None,
        on_notice: NoticeHook | None = None,
    ) -> TranscriptionResult:
        if not audio_path.exists():
            raise SttError(f"audio file not found: {audio_path.name}")
        if audio_path.stat().st_size == 0:
            raise SttError("audio file is empty")

        if not self._ready:
            await self.warmup()

        async with self._slots:
            started = time.monotonic()
            job_id = new_ulid()
            try:
                await self._submit(job_id, audio_path)
                await self._await_completion(job_id, on_progress)
                payload = await self._request("GET", f"/v1/jobs/{job_id}/result")
            finally:
                # The service sweeps abandoned jobs on a timer, but asking now is what keeps an
                # hour of audio from sitting on the GPU host until the TTL expires.
                await self._forget(job_id)

        result = _to_result(payload)
        log.info(
            "transcribed %s on the GPU service in %.1fs (%.0fs audio, %d segments, lang=%s)",
            audio_path.name,
            time.monotonic() - started,
            result.duration or 0,
            len(result.segments),
            result.language,
        )
        return result

    # ---- steps -----------------------------------------------------------

    async def _submit(self, job_id: str, audio_path: Path) -> None:
        query = {
            "language": self._language,
            "beam_size": str(self._beam_size),
            "filename": audio_path.name,
        }
        with audio_path.open("rb") as handle:
            # aiohttp takes the length from the file handle, so the body is streamed rather than
            # read into memory: an hour of audio is tens of megabytes. Waiting for 100-continue
            # means a refusal (bad token, file too large) arrives as that refusal instead of as a
            # broken pipe halfway through the upload.
            await self._request(
                "PUT",
                f"/v1/jobs/{job_id}",
                params=query,
                data=handle,
                timeout=self._upload_timeout,
                expect100=True,
            )

    async def _await_completion(self, job_id: str, on_progress: ProgressHook | None) -> None:
        last_percent = -1.0
        last_change = time.monotonic()
        while True:
            status = await self._request("GET", f"/v1/jobs/{job_id}")
            state = status.get("status")
            percent = float(status.get("percent") or 0.0)

            if percent != last_percent:
                last_percent = percent
                last_change = time.monotonic()
                if on_progress is not None:
                    on_progress(min(max(percent / 100.0, 0.0), 1.0))

            if state == "done":
                return
            if state == "failed":
                raise SttError(f"GPU transcription failed: {status.get('error') or 'unknown'}")
            if time.monotonic() - last_change > self._stall_timeout:
                raise SttError(
                    f"GPU transcription stalled at {percent:.0f}% for "
                    f"{self._stall_timeout:.0f}s"
                )
            await asyncio.sleep(self._poll_interval)

    async def _forget(self, job_id: str) -> None:
        try:
            await self._request("DELETE", f"/v1/jobs/{job_id}")
        except Exception as exc:
            log.debug("could not delete job %s: %s", job_id, exc)

    # ---- transport -------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        data: Any = None,
        timeout: float | None = None,
        authorized: bool = True,
        expect100: bool = False,
    ) -> dict[str, Any]:
        session = await self._get_session()
        headers = {"Authorization": f"Bearer {self._token}"} if authorized else {}
        try:
            async with session.request(
                method,
                self._base + path,
                params=params,
                data=data,
                headers=headers,
                expect100=expect100,
                timeout=aiohttp.ClientTimeout(total=timeout or self._request_timeout),
            ) as response:
                body = await response.json(content_type=None)
                if response.status >= 400:
                    # A 4xx is about this request; a 5xx says the service itself is in trouble, so
                    # the next attempt should health-check before trusting it again.
                    if response.status >= 500:
                        self._ready = False
                    raise SttError(
                        f"{method} {path} -> {response.status} "
                        f"{(body or {}).get('message') or (body or {}).get('code') or ''}".strip()
                    )
                return body or {}
        except SttError:
            raise
        except (TimeoutError, aiohttp.ClientError) as exc:
            # A transport failure means the host is gone, not that this job is bad: forget the
            # readiness so the next attempt starts with a fresh health check.
            self._ready = False
            raise SttError(f"transcription service {method} {path} failed: {exc}") from exc


def _to_result(payload: dict[str, Any]) -> TranscriptionResult:
    segments = [
        TranscriptSegment(item["start"], item["end"], item["text"])
        for item in payload.get("segments", [])
    ]
    return TranscriptionResult(
        text=(payload.get("text") or "").strip(),
        language=payload.get("language"),
        duration=payload.get("duration"),
        segments=segments,
    )
