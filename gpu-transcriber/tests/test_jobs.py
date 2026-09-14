"""The registry: what it forgets, and when.

Audio is the only thing here that grows without bound, so sweeping is the part worth testing.
"""

from __future__ import annotations

import time
from pathlib import Path

from gpu_transcriber.jobs import Job, JobStore, is_safe_job_id


def _job(store: JobStore, job_id: str) -> Job:
    audio = store.prepare_spool(job_id)
    audio.write_bytes(b"pretend audio")
    return store.submit(Job(job_id=job_id, audio_path=audio))


def test_a_collected_job_is_swept_once_it_is_old(tmp_path: Path) -> None:
    store = JobStore(tmp_path, ttl_seconds=60.0)
    _job(store, "01OLD")
    store.finish("01OLD", {"text": "x", "segments": []})
    store.get("01OLD").finished_at = time.time() - 3600

    assert store.sweep() == 1
    assert store.get("01OLD") is None
    assert not store.spool_dir("01OLD").exists()


def test_a_fresh_job_survives_the_sweep(tmp_path: Path) -> None:
    store = JobStore(tmp_path, ttl_seconds=60.0)
    _job(store, "01NEW")

    assert store.sweep() == 0
    assert store.get("01NEW") is not None


def test_audio_left_behind_by_a_previous_run_is_swept(tmp_path: Path) -> None:
    """A restart loses the records but not the files; nothing else would ever remove them."""
    store = JobStore(tmp_path, ttl_seconds=60.0)
    orphan = tmp_path / "01ORPHAN"
    orphan.mkdir()
    (orphan / "audio").write_bytes(b"50 megabytes, pretend")
    old = time.time() - 3600
    import os

    os.utime(orphan, (old, old))

    assert store.sweep() == 1
    assert not orphan.exists()


def test_a_young_orphan_is_left_alone(tmp_path: Path) -> None:
    store = JobStore(tmp_path, ttl_seconds=60.0)
    orphan = tmp_path / "01YOUNG"
    orphan.mkdir()

    assert store.sweep() == 0
    assert orphan.exists()


def test_a_spool_belonging_to_a_live_job_is_never_swept(tmp_path: Path) -> None:
    store = JobStore(tmp_path, ttl_seconds=60.0)
    _job(store, "01LIVE")
    old = time.time() - 3600
    import os

    os.utime(store.spool_dir("01LIVE"), (old, old))

    assert store.sweep() == 0
    assert store.spool_dir("01LIVE").exists()


def test_submitting_a_known_id_returns_the_first_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path, ttl_seconds=60.0)
    first = _job(store, "01SAME")
    second = store.submit(Job(job_id="01SAME", audio_path=store.audio_path("01SAME")))

    assert second is first
    assert store.queue_depth() == 1


def test_queue_depth_counts_only_what_is_waiting(tmp_path: Path) -> None:
    store = JobStore(tmp_path, ttl_seconds=60.0)
    _job(store, "01A")
    _job(store, "01B")
    store.next_pending(0.05)

    assert store.queue_depth() == 1


def test_only_plain_names_are_acceptable_job_ids() -> None:
    assert is_safe_job_id("01M0TXQWN7WR0NSQGD9M8QGQZY")
    assert is_safe_job_id("job-1_2")
    assert not is_safe_job_id("..")
    assert not is_safe_job_id("a/b")
    assert not is_safe_job_id("")
    assert not is_safe_job_id("x" * 65)
