"""Windowing and stitching for long recordings, without ffmpeg."""

from __future__ import annotations

from pathlib import Path

from helpers import FakeEngine

from gpu_transcriber.chunks import chunk_windows, merge_chunk_results, transcribe_audio
from gpu_transcriber.jobs import Job
from gpu_transcriber.worker import TranscriptionWorker


def test_a_short_file_is_a_single_window() -> None:
    assert chunk_windows(500, 600, 2) == [(0.0, 500.0)]
    assert chunk_windows(600, 600, 2) == [(0.0, 600.0)]


def test_a_long_file_overlaps_the_joins() -> None:
    windows = chunk_windows(1300, 600, 2)

    assert windows[0] == (0.0, 600.0)
    assert windows[1] == (598.0, 600.0)
    assert windows[2][0] == 1196.0
    assert windows[2][1] == 104.0


def test_zero_chunk_length_disables_splitting() -> None:
    assert chunk_windows(5000, 0, 2) == [(0.0, 5000.0)]


def test_later_chunks_drop_the_overlapped_head() -> None:
    first = {
        "language": "ru",
        "duration": 600.0,
        "segments": [
            {"start": 0.0, "end": 10.0, "text": "начало"},
            {"start": 590.0, "end": 600.0, "text": "край"},
        ],
    }
    second = {
        "language": "ru",
        "duration": 102.0,
        "segments": [
            {"start": 0.0, "end": 3.0, "text": "повтор"},
            {"start": 5.0, "end": 20.0, "text": "дальше"},
        ],
    }

    merged = merge_chunk_results([(0.0, first), (598.0, second)], overlap_seconds=2.0)

    assert [item["text"] for item in merged["segments"]] == ["начало", "край", "дальше"]
    assert merged["text"] == "начало край дальше"
    assert merged["duration"] == 700.0


def test_the_worker_splits_a_long_file_and_stitches(store) -> None:
    audio = store.prepare_spool("01LONG")
    audio.write_bytes(b"pretend audio")
    store.submit(Job(job_id="01LONG", audio_path=audio, filename="talk.mp3"))

    copies: list[tuple[float, float]] = []

    def probe(_path: Path) -> float:
        return 1200.0

    def extract(source: Path, dest: Path, start: float, length: float) -> None:
        copies.append((start, length))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())

    engine = FakeEngine(
        segments=[(0.0, 30.0, "раз"), (30.0, 60.0, "два")],
        duration=60.0,
    )
    engine.load()
    worker = TranscriptionWorker(
        store,
        engine,
        chunk_seconds=600,
        chunk_overlap_seconds=2,
        probe=probe,
        extract=extract,
    )

    worker.run_job("01LONG")

    job = store.get("01LONG")
    assert job.status == "done"
    assert copies == [(0.0, 600.0), (598.0, 600.0), (1196.0, 4.0)]
    assert len(engine.calls) == 3
    assert job.result["duration"] == 1200.0
    assert "раз" in job.result["text"]


def test_unknown_duration_does_not_split(tmp_path: Path) -> None:
    engine = FakeEngine()
    engine.load()
    audio = tmp_path / "short.mp3"
    audio.write_bytes(b"x")

    def boom(*_args):
        raise AssertionError("must not extract")

    result = transcribe_audio(
        engine,
        audio,
        language=None,
        beam_size=5,
        chunk_seconds=600,
        probe=lambda _path: None,
        extract=boom,
    )

    assert result["text"] == "первая вторая"
    assert len(engine.calls) == 1
