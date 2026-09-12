"""The GPU thread: progress while it works, an honest failure when it cannot."""

from __future__ import annotations

import threading
import time
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
    worker = TranscriptionWorker(store, engine, poll_interval=0.01)
    thread = threading.Thread(target=worker.run, args=(stop,), daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: engine.loads >= 1)
    finally:
        stop.set()
        thread.join(timeout=5.0)

    assert engine.loads == 1
    assert not engine.ready


def test_idle_unload_drops_weights_and_the_next_job_reloads(
    store: JobStore, engine: FakeEngine
) -> None:
    _queued(store, "01ONE")
    worker = TranscriptionWorker(
        store, engine, poll_interval=0.01, idle_unload_seconds=0.05
    )
    stop = threading.Event()
    thread = threading.Thread(target=worker.run, args=(stop,), daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: store.get("01ONE").status == "done")
        assert wait_for(lambda: engine.unloads == 1)
        assert not engine.ready
        _queued(store, "01TWO")
        assert wait_for(lambda: store.get("01TWO").status == "done")
    finally:
        stop.set()
        thread.join(timeout=5.0)

    assert engine.loads == 2
    assert engine.unloads >= 1
    assert len(engine.calls) == 2


def test_idle_unload_can_be_disabled(store: JobStore, engine: FakeEngine) -> None:
    _queued(store)
    worker = TranscriptionWorker(
        store, engine, poll_interval=0.01, idle_unload_seconds=0
    )
    stop = threading.Event()
    thread = threading.Thread(target=worker.run, args=(stop,), daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: store.get("01JOB").status == "done")
        time.sleep(0.08)
        assert engine.unloads == 0
        assert engine.ready
    finally:
        stop.set()
        thread.join(timeout=5.0)


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


def test_shared_gpu_does_not_preload_and_keeps_weights_warm_between_jobs(store, engine, tmp_path):
    """The next slice of the same recording arrives seconds later; it must not pay a cold load."""
    from gpu_transcriber.gpu_lock import gpu_slot
    lock = str(tmp_path / "gpu.lock")
    worker = TranscriptionWorker(
        store, engine, gpu_lock_path=lock, poll_interval=0.01, idle_unload_seconds=10.0
    )
    stop = threading.Event()
    thread = threading.Thread(target=worker.run, args=(stop,))
    with gpu_slot(lock):
        thread.start()
        _queued(store, "01ONE")
        stop.wait(0.05)
        assert not engine.ready  # OCR holds the card; whisper waits without loading
    try:
        assert wait_for(lambda: store.get("01ONE").status == "done")
        time.sleep(0.05)
        assert engine.ready and engine.unloads == 0
        _queued(store, "01TWO")
        assert wait_for(lambda: store.get("01TWO").status == "done")
        assert engine.loads == 1
    finally:
        stop.set()
        thread.join(2)
    assert not thread.is_alive()


def test_shared_gpu_yields_the_weights_when_another_process_takes_the_slot(store, engine, tmp_path):
    from gpu_transcriber.gpu_lock import gpu_slot
    lock = str(tmp_path / "gpu.lock")
    worker = TranscriptionWorker(
        store, engine, gpu_lock_path=lock, poll_interval=0.01, idle_unload_seconds=0
    )
    stop = threading.Event()
    thread = threading.Thread(target=worker.run, args=(stop,))
    thread.start()
    try:
        _queued(store)
        assert wait_for(lambda: store.get("01JOB").status == "done")
        time.sleep(0.05)
        assert engine.ready
        with gpu_slot(lock):  # OCR takes the card
            assert wait_for(lambda: not engine.ready)
        time.sleep(0.05)
        assert engine.loads == 1  # released slot does not reload on its own
    finally:
        stop.set()
        thread.join(2)
    assert not thread.is_alive()


def test_idle_unload_applies_on_a_shared_gpu_too(store, engine, tmp_path):
    worker = TranscriptionWorker(
        store, engine, gpu_lock_path=str(tmp_path / "gpu.lock"), poll_interval=0.01,
        idle_unload_seconds=0.05,
    )
    stop = threading.Event()
    thread = threading.Thread(target=worker.run, args=(stop,), daemon=True)
    thread.start()
    try:
        _queued(store)
        assert wait_for(lambda: store.get("01JOB").status == "done")
        assert wait_for(lambda: engine.unloads == 1)
    finally:
        stop.set()
        thread.join(2)
