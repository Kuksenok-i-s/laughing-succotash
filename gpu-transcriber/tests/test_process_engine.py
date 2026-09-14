import os
from pathlib import Path

import pytest

from gpu_transcriber.process_engine import ProcessWhisperEngine


class ChildEngine:
    def __init__(self, **options):
        self.options = options

    def load(self):
        if self.options["model"] == "fail-load":
            raise ValueError("load failed")

    def transcribe(self, path, *, language, beam_size, on_progress):
        if path.name == "crash":
            os._exit(12)
        if path.name == "error":
            raise ValueError("bad audio")
        on_progress(100.0, 3.0, 3.0, 1)
        return {"pid": os.getpid(), "language": language, "beam_size": beam_size}


def test_release_and_reload_in_distinct_processes():
    engine = ProcessWhisperEngine(model="test", factory=ChildEngine)
    try:
        progress = []
        first = engine.transcribe(Path("audio"), language="ru", beam_size=2,
                                  on_progress=lambda *args: progress.append(args))
        child = engine._process
        assert first["pid"] != os.getpid()
        assert first["language"] == "ru"
        assert first["beam_size"] == 2
        assert progress == [(100.0, 3.0, 3.0, 1)]
        assert engine.ready
        engine.unload()
        assert not child.is_alive()
        assert not engine.ready
        second = engine.transcribe(Path("audio"), language=None, beam_size=None)
        assert second["pid"] != first["pid"]
    finally:
        engine.unload()


@pytest.mark.parametrize("path, message", [("crash", "exited unexpectedly"),
                                           ("error", "bad audio")])
def test_child_failure_is_reported_and_next_job_recovers(path, message):
    engine = ProcessWhisperEngine(model="test", factory=ChildEngine)
    try:
        with pytest.raises(RuntimeError, match=message):
            engine.transcribe(Path(path), language=None, beam_size=None)
        assert not engine.ready
        assert engine.transcribe(Path("audio"), language=None, beam_size=None)["pid"]
    finally:
        engine.unload()


def test_load_failure_does_not_leave_child():
    engine = ProcessWhisperEngine(model="fail-load", factory=ChildEngine)
    with pytest.raises(RuntimeError, match="load failed"):
        engine.load()
    assert not engine.ready
    assert engine._process is None
