"""The single thread that owns the GPU.

One job at a time: two large-v3 runs on one card are slower together than one after the other, and
the memory spike risks the process. The model is loaded here rather than at startup so the HTTP
surface answers immediately. After a stretch of idle time the weights are dropped so OCR can use
the same card; the next job loads them again.
"""

from __future__ import annotations

import logging
import threading
import time
from functools import partial

from .engine import Engine
from .jobs import JobStore

log = logging.getLogger(__name__)


class TranscriptionWorker:
    def __init__(
        self,
        store: JobStore,
        engine: Engine,
        *,
        poll_interval: float = 0.5,
        idle_unload_seconds: float = 600.0,
    ) -> None:
        self._store = store
        self._engine = engine
        self._poll = poll_interval
        self._idle_unload = max(0.0, idle_unload_seconds)
        self._last_used = time.monotonic()

    def run(self, stop: threading.Event) -> None:
        if not self._engine.ready:
            try:
                self._engine.load()
                self._last_used = time.monotonic()
            except Exception:
                # Stay up: /health still answers, and the next job retries the load.
                log.exception("could not preload the whisper model; will retry on the first job")

        while not stop.is_set():
            job = self._store.next_pending(self._poll)
            if job is not None:
                self.run_job(job.job_id)
                self._last_used = time.monotonic()
                continue
            self._maybe_unload()

    def run_job(self, job_id: str) -> None:
        job = self._store.get(job_id)
        if job is None:
            return

        started = time.monotonic()
        try:
            result = self._engine.transcribe(
                job.audio_path,
                language=job.language,
                beam_size=job.beam_size,
                on_progress=partial(self._report, job_id),
            )
        except Exception as exc:
            log.exception("job %s failed", job_id)
            self._store.fail(job_id, f"{type(exc).__name__}: {exc}")
            return

        self._store.finish(job_id, result)
        log.info(
            "job %s transcribed in %.1fs (%.0fs audio, %d segments, lang=%s)",
            job_id,
            time.monotonic() - started,
            result.get("duration") or 0,
            len(result.get("segments", [])),
            result.get("language"),
        )

    def _maybe_unload(self) -> None:
        if self._idle_unload <= 0 or not self._engine.ready:
            return
        idle = time.monotonic() - self._last_used
        if idle < self._idle_unload:
            return
        try:
            self._engine.unload()
        except Exception:
            log.exception("idle unload failed")
            return
        log.info("unloaded whisper after %.0fs idle", idle)

    def _report(
        self,
        job_id: str,
        percent: float,
        position_sec: float,
        duration_sec: float | None,
        segment_count: int,
    ) -> None:
        self._store.report(
            job_id,
            percent=percent,
            position_sec=position_sec,
            duration_sec=duration_sec,
            segment_count=segment_count,
        )
