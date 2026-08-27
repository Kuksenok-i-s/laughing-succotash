"""The single thread that owns the GPU / Ollama slot.

One job at a time: two Qwen3-VL passes already occupy the card. The model is loaded here rather
than at HTTP bind so ``/health`` answers while weights are still coming in. After a stretch of
idle time the weights are dropped so Whisper can use the same card.
"""

from __future__ import annotations

import logging
import threading
import time
from functools import partial

from .engine import Engine
from .jobs import JobStore

log = logging.getLogger(__name__)


class OcrWorker:
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
        # Best-effort preload so the first photo is faster. Failure here must not kill the
        # worker: recognize() loads lazily when a job arrives (including after unload).
        if not self._engine.ready:
            try:
                self._engine.load()
                self._last_used = time.monotonic()
            except Exception:
                log.exception("could not preload the OCR model; will retry on the first job")

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
            result = self._engine.recognize(
                job.image_path,
                on_progress=partial(self._report, job_id),
            )
        except Exception as exc:
            log.exception("job %s failed", job_id)
            self._store.fail(job_id, f"{type(exc).__name__}: {exc}")
            return

        self._store.finish(job_id, result)
        log.info(
            "job %s done in %.1fs (kind=%s passes=%d raw=%d markdown=%d desc=%d)",
            job_id,
            time.monotonic() - started,
            result.get("kind") or "text",
            int(result.get("passes") or 0),
            len(result.get("raw_text") or ""),
            len(result.get("markdown") or ""),
            len(result.get("description") or ""),
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
        log.info("unloaded OCR model after %.0fs idle", idle)

    def _report(self, job_id: str, percent: float, stage: str) -> None:
        self._store.report(job_id, percent=percent, stage=stage)
