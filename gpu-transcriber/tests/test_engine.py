"""WhisperEngine wiring against a stand-in faster_whisper module."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from gpu_transcriber.engine import WhisperEngine


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Info:
    language: str = "ru"
    language_probability: float = 0.9
    duration: float = 60.0


class FakeModel:
    calls: list[dict] = []

    def __init__(self, name, *, device, compute_type):
        self.name, self.device, self.compute_type = name, device, compute_type

    def transcribe(self, audio, **options):
        FakeModel.calls.append({"path": "model", "audio": audio, **options})
        return iter([Segment(0.0, 30.0, "a"), Segment(30.0, 60.0, "b")]), Info()


class FakePipeline:
    def __init__(self, model):
        self.model = model

    def transcribe(self, audio, **options):
        FakeModel.calls.append({"path": "batched", "audio": audio, **options})
        return iter([Segment(0.0, 30.0, "a"), Segment(30.0, 60.0, "b")]), Info()


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeModel
    module.BatchedInferencePipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    FakeModel.calls = []
    return module


def test_batched_pipeline_decodes_windows_together(fake_faster_whisper) -> None:
    engine = WhisperEngine(model="large-v3-turbo", batch_size=6, beam_size=2)
    engine.load()

    result = engine.transcribe(Path("a.wav"), language="ru", beam_size=None)

    (call,) = FakeModel.calls
    assert call["path"] == "batched"
    assert call["batch_size"] == 6
    assert call["beam_size"] == 2
    assert call["condition_on_previous_text"] is False
    assert call["word_timestamps"] is False
    assert result["text"] == "a b"
    assert result["duration"] == 60.0


def test_batching_falls_back_to_sequential_without_vad(fake_faster_whisper) -> None:
    engine = WhisperEngine(model="large-v3-turbo", batch_size=8, vad_filter=False)
    engine.load()

    engine.transcribe(Path("a.wav"), language=None, beam_size=3)

    (call,) = FakeModel.calls
    assert call["path"] == "model"
    assert "batch_size" not in call
    assert call["beam_size"] == 3
    assert call["condition_on_previous_text"] is False


def test_batch_size_zero_disables_batching(fake_faster_whisper) -> None:
    engine = WhisperEngine(model="large-v3-turbo", batch_size=0)
    engine.load()

    engine.transcribe(Path("a.wav"), language=None, beam_size=None)

    assert FakeModel.calls[0]["path"] == "model"


def test_unload_drops_both_model_and_pipeline(fake_faster_whisper) -> None:
    engine = WhisperEngine(model="large-v3-turbo")
    engine.load()
    assert engine.ready

    engine.unload()

    assert not engine.ready
    assert engine._batched is None
