"""Settings come from the environment and nowhere else."""

from __future__ import annotations

from pathlib import Path

from gpu_transcriber.config import DEFAULT_PORT, from_env


def test_defaults_need_only_a_token() -> None:
    settings = from_env({"GPU_STT_TOKEN": "t" * 40})

    assert settings.port == DEFAULT_PORT
    assert settings.host == "0.0.0.0"
    assert settings.device == "cuda"
    assert settings.idle_unload_seconds == 600.0
    assert settings.validate_runtime() == []


def test_a_missing_or_short_token_is_a_configuration_error() -> None:
    assert from_env({}).validate_runtime() == ["GPU_STT_TOKEN is not set"]
    assert from_env({"GPU_STT_TOKEN": "short"}).validate_runtime() == [
        "GPU_STT_TOKEN is shorter than 32 characters"
    ]


def test_the_environment_overrides_every_default(tmp_path: Path) -> None:
    settings = from_env(
        {
            "GPU_STT_TOKEN": "t" * 40,
            "GPU_STT_HOST": "10.0.7.49",
            "GPU_STT_PORT": "18000",
            "GPU_STT_MODEL": "/models/large-v3",
            "GPU_STT_COMPUTE_TYPE": "int8",
            "GPU_STT_VAD_FILTER": "no",
            "GPU_STT_WORK_DIR": str(tmp_path / "work"),
            "GPU_STT_MAX_UPLOAD_MB": "64",
            "GPU_STT_JOB_TTL_SECONDS": "120",
            "GPU_STT_IDLE_UNLOAD_SECONDS": "90",
        }
    )

    assert settings.host == "10.0.7.49"
    assert settings.port == 18000
    assert settings.model == "/models/large-v3"
    assert settings.compute_type == "int8"
    assert settings.vad_filter is False
    assert settings.work_dir == tmp_path / "work"
    assert settings.max_upload_bytes == 64 * 1024 * 1024
    assert settings.job_ttl_seconds == 120.0
    assert settings.idle_unload_seconds == 90.0


def test_a_blank_value_falls_back_to_the_default() -> None:
    """An env file with an empty line for a setting should not mean "port zero"."""
    settings = from_env({"GPU_STT_TOKEN": "t" * 40, "GPU_STT_PORT": "", "GPU_STT_HOST": ""})

    assert settings.port == DEFAULT_PORT
    assert settings.host == ""
