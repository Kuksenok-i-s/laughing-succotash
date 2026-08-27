from __future__ import annotations

import threading

from helpers import FakeEngine, wait_for

from handwriting_ocr.jobs import Job
from handwriting_ocr.worker import OcrWorker


def _queued(store, tmp_path, job_id: str = "01W") -> None:
    image = tmp_path / f"{job_id}.jpg"
    image.write_bytes(b"jpeg")
    store.submit(Job(job_id=job_id, image_path=image, filename="note.jpg"))


def test_worker_runs_three_passes_for_text(store, engine: FakeEngine, tmp_path) -> None:
    engine.load()
    image = tmp_path / "note.jpg"
    image.write_bytes(b"jpeg")
    store.submit(Job(job_id="01W", image_path=image, filename="note.jpg"))

    OcrWorker(store, engine).run_job("01W")

    assert wait_for(lambda: store.get("01W").status == "done")
    assert len(engine.calls) == 3
    assert store.get("01W").result["passes"] == 3
    assert store.get("01W").result["kind"] == "text"


def test_worker_stops_after_triage_for_other(store, tmp_path) -> None:
    engine = FakeEngine(kind="other", description="red mug")
    engine.load()
    image = tmp_path / "mug.jpg"
    image.write_bytes(b"jpeg")
    store.submit(Job(job_id="01O", image_path=image, filename="mug.jpg"))

    OcrWorker(store, engine).run_job("01O")

    assert wait_for(lambda: store.get("01O").status == "done")
    assert len(engine.calls) == 1
    assert store.get("01O").result["passes"] == 1
    assert store.get("01O").result["kind"] == "other"
    assert store.get("01O").result["description"] == "red mug"


def test_idle_unload_drops_weights_and_the_next_job_reloads(store, engine: FakeEngine, tmp_path) -> None:
    _queued(store, tmp_path, "01ONE")
    worker = OcrWorker(store, engine, poll_interval=0.01, idle_unload_seconds=0.05)
    stop = threading.Event()
    thread = threading.Thread(target=worker.run, args=(stop,), daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: store.get("01ONE").status == "done")
        assert wait_for(lambda: engine.unloads == 1)
        assert not engine.ready
        _queued(store, tmp_path, "01TWO")
        assert wait_for(lambda: store.get("01TWO").status == "done")
    finally:
        stop.set()
        thread.join(timeout=5.0)

    assert engine.loads == 2
    assert engine.unloads >= 1
    assert store.get("01TWO").result["kind"] == "text"
