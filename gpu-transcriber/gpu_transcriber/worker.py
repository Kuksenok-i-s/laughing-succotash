"""The single thread that owns the GPU.

One job at a time: two large-v3 runs on one card are slower together than one after the other, and
the memory spike risks the process. The model is loaded here rather than at startup so the HTTP
surface answers immediately and ``/health`` can report the truth while the weights are still
loading.
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
    def __init__(self, store: JobStore, engine: Engine, *, poll_interval: float = 0.5) -> None:
        self._store = store
        self._engine = engine
        self._poll = poll_interval

    def run(self, stop: threading.Event) -> None:
        if not self._engine.ready:
            try:
                self._engine.load()
            except Exception:
                # Nothing can be transcribed without a model, but the service stays up so the Core
                # gets a clear answer instead of a connection error.
                log.exception("could not load the whisper model")
                return

        while not stop.is_set():
            job = self._store.next_pending(self._poll)
            if job is not None:
                self.run_job(job.job_id)

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
