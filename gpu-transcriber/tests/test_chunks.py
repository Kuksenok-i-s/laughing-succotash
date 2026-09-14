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
        prepare=lambda src, _dest: src,
    )

    worker.run_job("01LONG")

    job = store.get("01LONG")
    assert job.status == "done"
    # Windows are cut in a thread pool, so only the set is deterministic.
    assert sorted(copies) == [(0.0, 600.0), (598.0, 600.0), (1196.0, 4.0)]
    assert len(engine.calls) == 3
    assert [call["audio"].name for call in engine.calls] == ["000.raw.wav", "001.raw.wav", "002.raw.wav"]
    assert job.result["duration"] == 1200.0
    assert "раз" in job.result["text"]
    assert not (audio.parent / "chunks").exists()


def test_a_prepared_upload_skips_the_filter_chain(tmp_path: Path) -> None:
    """A DualHost slice already carries highpass+loudnorm; running it again wastes CPU."""
    engine = FakeEngine()
    engine.load()
    audio = tmp_path / "slice.wav"
    audio.write_bytes(b"wav")

    def must_not_prepare(*_args):
        raise AssertionError("prepare must not run on prepared audio")

    def extract(source: Path, dest: Path, start: float, length: float) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())

    single = transcribe_audio(
        engine, audio, language="ru", beam_size=2, chunk_seconds=600,
        probe=lambda _path: 120.0, extract=extract, prepare=must_not_prepare, prepared=True,
    )
    assert single["text"] == "первая вторая"
    assert engine.calls[-1]["audio"] == audio

    split = transcribe_audio(
        engine, audio, language="ru", beam_size=2, chunk_seconds=300,
        probe=lambda _path: 600.0, extract=extract, prepare=must_not_prepare, prepared=True,
    )
    assert split["duration"] == 600.0
    assert len(engine.calls) == 4


def test_the_next_window_is_cut_while_the_current_one_decodes(tmp_path: Path) -> None:
    import threading

    engine = FakeEngine(pause_after=1)
    engine.load()
    audio = tmp_path / "long.wav"
    audio.write_bytes(b"wav")
    cut: list[int] = []
    cut_event = threading.Event()

    def extract(source: Path, dest: Path, start: float, length: float) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
        cut.append(int(dest.name.split(".")[0]))
        if len(cut) == 2:
            cut_event.set()

    def run() -> None:
        transcribe_audio(
            engine, audio, language="ru", beam_size=2, chunk_seconds=600,
            probe=lambda _path: 1200.0, extract=extract, prepare=lambda src, _dest: src,
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert engine.reached_pause.wait(timeout=5.0)
        # The GPU is mid-way through window 0 and window 1 is already on disk.
        assert cut_event.wait(timeout=5.0)
        assert sorted(cut)[:2] == [0, 1]
    finally:
        engine.resume.set()
        thread.join(timeout=5.0)
    assert not thread.is_alive()


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
        prepare=lambda src, _dest: src,
    )

    assert result["text"] == "первая вторая"
    assert len(engine.calls) == 1
