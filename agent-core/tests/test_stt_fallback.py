"""CPU fallback around a failing primary STT backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.config import Settings
from agent_core.stt.base import SttError, TranscriptionResult, TranscriptSegment
from agent_core.stt.fallback import FallbackSTT


class FakeSTT:
    def __init__(self, name: str, *, fail_warmup: bool = False, fail_transcribe: bool = False) -> None:
        self.model_name = name
        self.ready = False
        self.fail_warmup = fail_warmup
        self.fail_transcribe = fail_transcribe
        self.warmups = 0
        self.calls: list[Path] = []

    async def warmup(self) -> None:
        self.warmups += 1
        if self.fail_warmup:
            raise RuntimeError(f"{self.model_name} unreachable")
        self.ready = True

    async def close(self) -> None:
        self.ready = False

    async def transcribe(self, path: Path, *, on_progress=None) -> TranscriptionResult:
        self.calls.append(path)
        if self.fail_transcribe:
            raise RuntimeError(f"{self.model_name} ssh timeout")
        if on_progress is not None:
            on_progress(1.0)
        return TranscriptionResult(
            text=self.model_name,
            language="ru",
            duration=1.0,
            segments=[TranscriptSegment(0.0, 1.0, self.model_name)],
        )


@pytest.mark.asyncio
async def test_warmup_falls_back_when_primary_is_down(tmp_path: Path) -> None:
    gpu = FakeSTT("gpu", fail_warmup=True)
    cpu = FakeSTT("cpu")
    stt = FallbackSTT(primary=gpu, fallback=cpu)

    await stt.warmup()

    assert stt.model_name == "fallback/cpu"
    assert cpu.warmups == 1
    result = await stt.transcribe(tmp_path / "a.ogg")
    assert result.text == "cpu"
    assert gpu.calls == []


@pytest.mark.asyncio
async def test_transcribe_falls_back_and_stays_on_cpu(tmp_path: Path) -> None:
    gpu = FakeSTT("gpu", fail_transcribe=True)
    cpu = FakeSTT("cpu")
    stt = FallbackSTT(primary=gpu, fallback=cpu)
    await stt.warmup()

    first = await stt.transcribe(tmp_path / "a.ogg")
    second = await stt.transcribe(tmp_path / "b.ogg")

    assert first.text == "cpu"
    assert second.text == "cpu"
    assert gpu.calls == [tmp_path / "a.ogg"]
    assert cpu.calls == [tmp_path / "a.ogg", tmp_path / "b.ogg"]


@pytest.mark.asyncio
async def test_primary_success_does_not_touch_fallback(tmp_path: Path) -> None:
    gpu = FakeSTT("gpu")
    cpu = FakeSTT("cpu")
    stt = FallbackSTT(primary=gpu, fallback=cpu)
    await stt.warmup()

    result = await stt.transcribe(tmp_path / "a.ogg")

    assert result.text == "gpu"
    assert cpu.warmups == 0
    assert cpu.calls == []


def test_cpu_fallback_defaults_on() -> None:
    settings = Settings(
        instance_id="t",
        gateway_url="ws://localhost/rpc",
        core_token="x" * 40,
        mcp_token="y" * 40,
        allowed_users=["tg:1"],
        stt_backend="gpu",
    )
    assert settings.stt_cpu_fallback is True
    assert settings.stt_backend == "gpu"


def test_cpu_fallback_can_be_disabled() -> None:
    settings = Settings(
        instance_id="t",
        gateway_url="ws://localhost/rpc",
        core_token="x" * 40,
        mcp_token="y" * 40,
        allowed_users=["tg:1"],
        stt_backend="gpu",
        stt_cpu_fallback=False,
    )
    assert settings.stt_cpu_fallback is False


def test_stt_backend_rejects_unknown_values() -> None:
    with pytest.raises(Exception):
        Settings(
            instance_id="t",
            gateway_url="ws://localhost/rpc",
            core_token="x" * 40,
            mcp_token="y" * 40,
            allowed_users=["tg:1"],
            stt_backend="tpu",
        )
