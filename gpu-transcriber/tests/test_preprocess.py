"""ffmpeg prepare chain: high-pass, loudnorm, 16 kHz mono. No real encoder."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gpu_transcriber.chunks import transcribe_audio
from gpu_transcriber.preprocess import FILTER, WHISPER_RATE, prepare_whisper_audio
from helpers import FakeEngine


def test_missing_ffmpeg_leaves_the_original(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "quiet.ogg"
    source.write_bytes(b"ogg")
    monkeypatch.setattr("gpu_transcriber.preprocess.shutil.which", lambda _name: None)

    assert prepare_whisper_audio(source, tmp_path / "out.wav") == source


def test_ffmpeg_writes_16k_mono_with_highpass_and_loudnorm(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "talk.mp3"
    source.write_bytes(b"mp3")
    dest = tmp_path / "out.wav"
    seen: list[list[str]] = []

    def which(name: str) -> str | None:
        return "/usr/bin/ffmpeg" if name == "ffmpeg" else None

    def run(argv, **_kwargs):
        seen.append(list(argv))
        Path(argv[-1]).write_bytes(b"RIFF")
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr("gpu_transcriber.preprocess.shutil.which", which)
    monkeypatch.setattr("gpu_transcriber.preprocess.subprocess.run", run)

    assert prepare_whisper_audio(source, dest) == dest
    argv = seen[0]
    assert argv[0] == "ffmpeg"
    assert FILTER in argv
    assert argv[argv.index("-ar") + 1] == str(WHISPER_RATE)
    assert argv[argv.index("-ac") + 1] == "1"
    assert argv[argv.index("-c:a") + 1] == "pcm_s16le"


def test_a_failed_prepare_falls_back_to_the_original(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "bad.mp3"
    source.write_bytes(b"x")
    dest = tmp_path / "out.wav"

    monkeypatch.setattr("gpu_transcriber.preprocess.shutil.which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "gpu_transcriber.preprocess.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr=b"no decoder"),
    )

    assert prepare_whisper_audio(source, dest) == source
    assert not dest.exists()


def test_transcribe_uses_the_prepared_file(tmp_path: Path) -> None:
    engine = FakeEngine()
    engine.load()
    audio = tmp_path / "talk.mp3"
    audio.write_bytes(b"raw")
    prepared = tmp_path / "clean.wav"

    def prepare(source: Path, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"wav")
        prepared.write_bytes(b"wav")
        return dest

    result = transcribe_audio(
        engine,
        audio,
        language="ru",
        beam_size=5,
        probe=lambda _path: 12.0,
        extract=lambda *_args: None,
        prepare=prepare,
    )

    assert result["text"] == "первая вторая"
    assert engine.calls[0]["audio"].name == "input.wav"
    assert not (audio.parent / "prepared").exists()
