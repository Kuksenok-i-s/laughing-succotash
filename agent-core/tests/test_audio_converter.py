"""The Whisper ffmpeg filter chain is part of the Core contract, not a comment."""

from agent_core.audio.converter import HIGHPASS_HZ, LOUDNORM, WHISPER_FILTER, WHISPER_RATE


def test_whisper_prepare_is_highpass_loudnorm_16k() -> None:
    assert HIGHPASS_HZ == 80
    assert WHISPER_RATE == 16_000
    assert LOUDNORM.startswith("loudnorm=")
    assert WHISPER_FILTER == f"highpass=f={HIGHPASS_HZ},{LOUDNORM}"
