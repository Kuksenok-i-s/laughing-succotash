"""Job registry: one record per job, one spool file, one queue.

The registry is in memory on purpose. A restart loses jobs that were in flight, the Core sees them
gone and falls back to CPU — a better outcome than a durable queue that replays an hour of GPU work
nobody is waiting for any more. What is on disk is only the audio, and only until the result has
been collected.
"""

from __future__ import annotations

import logging
import queue
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

AUDIO_FILENAME = "audio"
TERMINAL_STATES = frozenset({"done", "failed"})

# Job ids come from the Core (a ULID) and become a directory name, so anything that could climb out
# of the work directory is refused before it is ever joined to a path.
_SAFE_ID = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")


def is_safe_job_id(job_id: str) -> bool:
    return bool(_SAFE_ID.match(job_id))


@dataclass(slots=True)
class Job:
    job_id: str
    audio_path: Path
    filename: str = ""
    language: str | None = None
    beam_size: int | None = None
    status: str = "queued"
    percent: float = 0.0
    position_sec: float = 0.0
    duration_sec: float | None = None
    segment_count: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        reference = self.finished_at or time.time()
        elapsed = None if self.started_at is None else round(reference - self.started_at, 1)
        return {
            "job_id": self.job_id,
            "status": self.status,
            "percent": round(self.percent, 1),
            "position_sec": round(self.position_sec, 1),
            "duration_sec": self.duration_sec,
            "segments": self.segment_count,
            "elapsed_sec": elapsed,
            "error": self.error,
        }


class JobStore:
    def __init__(self, work_dir: Path, *, ttl_seconds: float) -> None:
        self._work_dir = work_dir
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._pending: queue.Queue[str] = queue.Queue()

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    def spool_dir(self, job_id: str) -> Path:
        return self._work_dir / job_id

    def audio_path(self, job_id: str) -> Path:
        return self.spool_dir(job_id) / AUDIO_FILENAME

    def prepare_spool(self, job_id: str) -> Path:
        """Make room for an upload and return the path to write it to."""
        spool = self.spool_dir(job_id)
        spool.mkdir(parents=True, exist_ok=True)
        return spool / AUDIO_FILENAME

    def discard_spool(self, job_id: str) -> None:
        shutil.rmtree(self.spool_dir(job_id), ignore_errors=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, job: Job) -> Job:
        """Register a job and queue it. An id already known wins; the caller's job is dropped."""
        with self._lock:
            existing = self._jobs.get(job.job_id)
            if existing is not None:
                return existing
            self._jobs[job.job_id] = job
        self._pending.put(job.job_id)
        log.info(
            "job %s queued (%s, language=%s)", job.job_id, job.filename, job.language or "auto"
        )
        return job

    def next_pending(self, timeout: float) -> Job | None:
        """Claim the next queued job, or None if the wait elapsed. Called by the worker thread."""
        try:
            job_id = self._pending.get(timeout=timeout)
        except queue.Empty:
            return None
        with self._lock:
            job = self._jobs.get(job_id)
            # Deleted while it waited: nothing to do, and no error worth reporting.
            if job is None:
                return None
            job.status = "running"
            job.started_at = time.time()
            return job

    def report(
        self,
        job_id: str,
        *,
        percent: float,
        position_sec: float,
        duration_sec: float | None,
        segment_count: int,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_STATES:
                return
            job.percent = percent
            job.position_sec = position_sec
            job.duration_sec = duration_sec
            job.segment_count = segment_count

    def finish(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "done"
            job.percent = 100.0
            job.duration_sec = result.get("duration") or job.duration_sec
            job.segment_count = len(result.get("segments", []))
            job.result = result
            job.finished_at = time.time()

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.error = error
            job.finished_at = time.time()

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        self.discard_spool(job_id)
        return job is not None

    def queue_depth(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.status == "queued")

    def sweep(self, now: float | None = None) -> int:
        """Drop jobs nobody collected, and spool directories with no job at all.

        The second half matters after a restart: the records are gone but their audio is not, and
        that audio is the only thing here that grows without bound.
        """
        moment = time.time() if now is None else now
        with self._lock:
            stale = [
                job_id
                for job_id, job in self._jobs.items()
                if moment - (job.finished_at or job.created_at) > self._ttl
            ]
            for job_id in stale:
                del self._jobs[job_id]
            known = set(self._jobs)

        removed = 0
        for job_id in stale:
            self.discard_spool(job_id)
            removed += 1

        if self._work_dir.exists():
            for spool in self._work_dir.iterdir():
                if not spool.is_dir() or spool.name in known:
                    continue
                try:
                    age = moment - spool.stat().st_mtime
                except OSError:
                    continue
                if age > self._ttl:
                    shutil.rmtree(spool, ignore_errors=True)
                    removed += 1

        if removed:
            log.info("swept %d expired job(s)", removed)
        return removed
