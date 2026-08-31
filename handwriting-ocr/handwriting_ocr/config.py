"""Service configuration.

Environment only, and a plain dataclass rather than pydantic-settings. The HTTP surface is
standard library; the only external dependency is the Ollama HTTP API on localhost.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "OCR_"

# Neighbouring the STT service port (17493) so a netstat listing shows the pair.
DEFAULT_PORT = 17494


@dataclass(frozen=True, slots=True)
class Settings:
    # Loopback by default: Core and OCR share 10.0.7.49. Override OCR_HOST only if they split.
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    token: str = ""

    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3-vl:2b"
    keep_alive: str = "10m"
    request_timeout: float = 600.0

    work_dir: Path = Path("~/.handwriting-ocr").expanduser()
    job_ttl_seconds: float = 6 * 3600.0
    sweep_interval_seconds: float = 60.0
    idle_unload_seconds: float = 600.0
    max_upload_mb: int = 32
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
        if not self.model.strip():
            problems.append(f"{ENV_PREFIX}MODEL is empty")
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

    work_dir = source.get(ENV_PREFIX + "WORK_DIR")
    return Settings(
        host=text("HOST", defaults.host),
        port=number("PORT", defaults.port),
        token=text("TOKEN", defaults.token),
        ollama_url=text("OLLAMA_URL", defaults.ollama_url).rstrip("/"),
        model=text("MODEL", defaults.model),
        keep_alive=text("OLLAMA_KEEP_ALIVE", defaults.keep_alive),
        request_timeout=seconds("REQUEST_TIMEOUT", defaults.request_timeout),
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
