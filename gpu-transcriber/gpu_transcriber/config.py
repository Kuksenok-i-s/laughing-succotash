"""Service configuration.

Environment only, and a plain dataclass rather than pydantic-settings. The virtualenv on the GPU
host holds faster-whisper and its dependencies; adding wheels to a Python 3.14 environment for the
sake of five endpoints is not a trade worth making, and the service has no secrets beyond one
token.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "GPU_STT_"

# 17492 is the Gateway's RPC port on the VPS; keeping the neighbouring number makes the pair
# recognisable in a netstat listing.
DEFAULT_PORT = 17493


@dataclass(frozen=True, slots=True)
class Settings:
    # Loopback by default: Core and GPU share 10.0.7.49. Override GPU_STT_HOST only if they split.
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    token: str = ""

    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True

    work_dir: Path = Path("~/.gpu-transcriber").expanduser()
    # Long enough that a Core restart mid-job can still collect the result, short enough that an
    # abandoned hour of audio does not outlive the night.
    job_ttl_seconds: float = 6 * 3600.0
    sweep_interval_seconds: float = 60.0
    # Drop weights after this much quiet so OCR can use the same card. 0 disables.
    idle_unload_seconds: float = 600.0
    max_upload_mb: int = 512
    upload_chunk_size: int = 1024 * 1024

    log_level: str = "INFO"
    log_format: str = "text"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def validate_runtime(self) -> list[str]:
        problems: list[str] = []
        if not self.token:
            problems.append(f"{ENV_PREFIX}TOKEN is not set")
        elif len(self.token) < 32:
            problems.append(f"{ENV_PREFIX}TOKEN is shorter than 32 characters")
        return problems


def from_env(env: Mapping[str, str] | None = None) -> Settings:
    source = os.environ if env is None else env
    defaults = Settings()

    def text(name: str, fallback: str) -> str:
        raw = source.get(ENV_PREFIX + name)
        if raw is None or not raw.strip():
            return fallback
        return raw

    def number(name: str, fallback: int) -> int:
        raw = source.get(ENV_PREFIX + name)
        return fallback if raw is None or not raw.strip() else int(raw)

    def seconds(name: str, fallback: float) -> float:
        raw = source.get(ENV_PREFIX + name)
        return fallback if raw is None or not raw.strip() else float(raw)

    def flag(name: str, fallback: bool) -> bool:
        raw = source.get(ENV_PREFIX + name)
        if raw is None or not raw.strip():
            return fallback
        return raw.strip().lower() in ("1", "true", "yes", "on")

    work_dir = source.get(ENV_PREFIX + "WORK_DIR")
    return Settings(
        host=text("HOST", defaults.host),
        port=number("PORT", defaults.port),
        token=text("TOKEN", defaults.token),
        model=text("MODEL", defaults.model),
        device=text("DEVICE", defaults.device),
        compute_type=text("COMPUTE_TYPE", defaults.compute_type),
        beam_size=number("BEAM_SIZE", defaults.beam_size),
        vad_filter=flag("VAD_FILTER", defaults.vad_filter),
        work_dir=(
            Path(work_dir).expanduser().resolve() if work_dir else defaults.work_dir
        ),
        job_ttl_seconds=seconds("JOB_TTL_SECONDS", defaults.job_ttl_seconds),
        sweep_interval_seconds=seconds(
            "SWEEP_INTERVAL_SECONDS", defaults.sweep_interval_seconds
        ),
        idle_unload_seconds=seconds(
            "IDLE_UNLOAD_SECONDS", defaults.idle_unload_seconds
        ),
        max_upload_mb=number("MAX_UPLOAD_MB", defaults.max_upload_mb),
        upload_chunk_size=number("UPLOAD_CHUNK_SIZE", defaults.upload_chunk_size),
        log_level=text("LOG_LEVEL", defaults.log_level),
        log_format=text("LOG_FORMAT", defaults.log_format),
    )
