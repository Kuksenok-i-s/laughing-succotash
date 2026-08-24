"""The GPU thread: progress while it works, an honest failure when it cannot."""

from __future__ import annotations

import threading
from collections.abc import Callable

from helpers import Client, FakeEngine, wait_for

from gpu_transcriber.config import Settings
from gpu_transcriber.jobs import Job, JobStore
from gpu_transcriber.server import TranscriptionApp
from gpu_transcriber.worker import TranscriptionWorker


def _queued(store: JobStore, job_id: str = "01JOB") -> Job:
    audio = store.prepare_spool(job_id)
    audio.write_bytes(b"pretend audio")
    return store.submit(Job(job_id=job_id, audio_path=audio, filename="a.mp3"))


def test_a_finished_job_carries_the_transcript(store: JobStore, engine: FakeEngine) -> None:
    _queued(store)
    worker = TranscriptionWorker(store, engine)
    engine.load()

    worker.run_job("01JOB")

    job = store.get("01JOB")
    assert job.status == "done"
    assert job.percent == 100.0
    assert job.result["text"] == "первая вторая"
    assert job.result["duration"] == 60.0
    assert len(job.result["segments"]) == 2


def test_progress_is_visible_over_http_while_the_job_runs(
    settings: Settings, store: JobStore, serve: Callable[[TranscriptionApp], Client]
) -> None:
    """The whole point of the service: a percentage that moves during the transcription."""
    engine = FakeEngine(pause_after=1)
    engine.load()
    client = serve(TranscriptionApp(settings, store, engine))
    worker = TranscriptionWorker(store, engine)
    client.put_audio("01LIVE")

    thread = threading.Thread(target=worker.run_job, args=("01LIVE",), daemon=True)
    thread.start()
    try:
        assert engine.reached_pause.wait(timeout=5.0)
        midway = client.request("GET", "/v1/jobs/01LIVE")
        assert midway.payload["percent"] == 50.0
        assert midway.payload["position_sec"] == 30.0
        assert midway.payload["duration_sec"] == 60.0
        assert midway.payload["segments"] == 1
    finally:
        engine.resume.set()
        thread.join(timeout=5.0)

    assert wait_for(lambda: store.get("01LIVE").status == "done")
    result = client.request("GET", "/v1/jobs/01LIVE/result")
    assert result.status == 200
    assert result.payload["segments"][1]["text"] == "вторая"


def test_a_crash_inside_whisper_fails_only_that_job(store: JobStore) -> None:
    engine = FakeEngine(error=RuntimeError("cuda out of memory"))
    engine.load()
    _queued(store)

    TranscriptionWorker(store, engine).run_job("01JOB")

    job = store.get("01JOB")
    assert job.status == "failed"
    assert job.error == "RuntimeError: cuda out of memory"


def test_a_model_that_will_not_load_leaves_the_service_up(store: JobStore) -> None:
    """Without this the process would die and the Core would see a refused connection."""
    engine = FakeEngine(load_error=RuntimeError("no cuda device"))
    stop = threading.Event()

    TranscriptionWorker(store, engine).run(stop)

    assert engine.loads == 1
    assert not engine.ready


def test_a_job_deleted_while_queued_is_never_started(
    store: JobStore, engine: FakeEngine
) -> None:
    _queued(store)
    store.delete("01JOB")
    engine.load()

    claimed = store.next_pending(0.05)

    assert claimed is None
    assert engine.calls == []


def test_the_worker_drains_the_queue_until_told_to_stop(
    store: JobStore, engine: FakeEngine
) -> None:
    _queued(store, "01ONE")
    _queued(store, "01TWO")
    worker = TranscriptionWorker(store, engine, poll_interval=0.01)
    stop = threading.Event()
    thread = threading.Thread(target=worker.run, args=(stop,), daemon=True)
    thread.start()
    try:
        assert wait_for(
            lambda: store.get("01ONE").status == "done" and store.get("01TWO").status == "done"
        )
    finally:
        stop.set()
        thread.join(timeout=5.0)

    assert engine.loads == 1
    assert len(engine.calls) == 2
