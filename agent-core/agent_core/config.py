"""Agent Core configuration.

Secrets come from the environment or a local ``.env``; nothing sensitive is committed, and the
filesystem/project allowlists are the security boundary for everything Cursor can reach.
"""

from __future__ import annotations

import tomllib
from datetime import tzinfo
from functools import cached_property
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectConfig(BaseModel):
    """One coding project Cursor is permitted to open."""

    path: Path
    writable: bool = False
    description: str | None = None

    @field_validator("path")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value.expanduser().resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    instance_id: str = "home-macmini"

    # --- Gateway connection ---
    gateway_url: str = "wss://gateway.example.com/rpc"
    core_token: str = ""
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 60.0
    reconnect_healthy_after: float = 60.0
    ping_interval: float = 20.0
    ping_timeout: float = 20.0
    rpc_call_timeout: float = 30.0
    # TLS verification is on by default and should only ever be disabled against a local test
    # gateway with a self-signed certificate.
    verify_tls: bool = True

    # --- Authorization ---
    allowed_users: list[str] = Field(default_factory=list)

    @field_validator("allowed_users", mode="before")
    @classmethod
    def _split_users(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [u.strip() for u in value.split(",") if u.strip()]
        return value

    @field_validator("stt_backend")
    @classmethod
    def _stt_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"local", "gpu"}:
            raise ValueError("STT_BACKEND must be 'local' or 'gpu'")
        return normalized

    # --- Storage ---
    data_dir: Path = Path("~/.personal-assistant")
    database_path: Path | None = None
    temp_dir: Path | None = None

    # --- Agent ---
    agent_backend: str = "acp"  # "acp" | "cli"
    cursor_agent_binary: str = "cursor-agent"
    cursor_model: str | None = None
    agent_startup_timeout: float = 60.0
    agent_prompt_timeout: float = 1800.0
    # Sandbox workspace for ordinary assistant chat. Cursor's built-in write and shell tools do
    # not request permission (see docs/cursor-acp.md), so the conversation session must not be
    # rooted anywhere that matters.
    assistant_workspace: Path | None = None

    # --- MCP ---
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8931
    mcp_token: str = ""

    # --- STT ---
    # local = faster-whisper on this machine; gpu = the transcription service on the CUDA host.
    stt_backend: str = "local"
    stt_gpu_url: str = "http://127.0.0.1:17493"
    stt_gpu_token: str = ""
    stt_gpu_poll_interval: float = 2.0
    stt_gpu_request_timeout: float = 30.0
    stt_gpu_upload_timeout: float = 900.0
    # When STT_BACKEND=gpu, try local CPU whisper if the GPU host is unreachable. Off = fail.
    stt_cpu_fallback: bool = True
    stt_model: str = "large-v3"
    stt_device: str = "cpu"
    stt_compute_type: str = "auto"
    stt_language: str = "auto"
    stt_max_concurrent: int = 1
    stt_beam_size: int = 5
    stt_vad_filter: bool = True
    max_audio_size_mb: int = 500
    max_audio_duration_seconds: int = 14400
    upload_idle_timeout: float = 300.0

    # --- YouTube ---
    # yt-dlp runs on the proxy VPS over SSH; the toml holds that host and its key.
    youtube_config: Path | None = None

    # --- Behaviour ---
    default_timezone: str = "Europe/Moscow"
    confirmation_timeout_seconds: int = 900
    scheduler_tick_seconds: float = 5.0
    # A transcript longer than this is treated as a recording to analyse rather than a spoken
    # command, which changes both the prompt and the permission provenance.
    long_transcript_chars: int = 1200
    transcript_chunk_chars: int = 12000

    # --- Allowlists (from assistant.toml) ---
    config_file: Path | None = None

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "text"  # "text" | "json"

    @field_validator("data_dir")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return value.expanduser()

    @cached_property
    def resolved_data_dir(self) -> Path:
        path = self.data_dir.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @cached_property
    def resolved_database_path(self) -> Path:
        if self.database_path is not None:
            return self.database_path.expanduser().resolve()
        return self.resolved_data_dir / "core.sqlite3"

    @cached_property
    def resolved_temp_dir(self) -> Path:
        path = (
            self.temp_dir.expanduser().resolve()
            if self.temp_dir is not None
            else self.resolved_data_dir / "tmp"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @cached_property
    def resolved_assistant_workspace(self) -> Path:
        path = (
            self.assistant_workspace.expanduser().resolve()
            if self.assistant_workspace is not None
            else self.resolved_data_dir / "workspace"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @cached_property
    def timezone(self) -> tzinfo:
        try:
            return ZoneInfo(self.default_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                f"unknown DEFAULT_TIMEZONE {self.default_timezone!r}; "
                "timezone must be explicit for reminders to be correct"
            ) from exc

    @cached_property
    def _allowlists(self) -> dict[str, Any]:
        candidate = (
            self.config_file.expanduser().resolve()
            if self.config_file is not None
            else self.resolved_data_dir / "assistant.toml"
        )
        if not candidate.exists():
            return {}
        with candidate.open("rb") as handle:
            return tomllib.load(handle)

    @cached_property
    def allowed_files(self) -> list[Path]:
        """Directories the assistant may read. ``$HOME`` is never included wholesale."""
        raw = self._allowlists.get("files") or []
        return [Path(p).expanduser().resolve() for p in raw]

    @cached_property
    def projects(self) -> dict[str, ProjectConfig]:
        raw = self._allowlists.get("projects") or {}
        return {name: ProjectConfig(**cfg) for name, cfg in raw.items()}

    @cached_property
    def resolved_youtube_config(self) -> Path:
        """Where the YouTube worker's SSH settings live. Unrelated to the GPU service."""
        if self.youtube_config is not None:
            return self.youtube_config.expanduser().resolve()
        return self.resolved_data_dir / "youtube" / "config.toml"

    @cached_property
    def max_audio_bytes(self) -> int:
        return self.max_audio_size_mb * 1024 * 1024

    def validate_runtime(self) -> list[str]:
        """Return fatal misconfigurations. Checked at startup so failures are loud and early."""
        problems: list[str] = []
        if not self.core_token:
            problems.append("CORE_TOKEN is not set")
        elif len(self.core_token) < 32:
            problems.append("CORE_TOKEN is shorter than 32 characters")
        if not self.mcp_token:
            problems.append("MCP_TOKEN is not set")
        if self.stt_backend == "gpu" and not self.stt_gpu_token:
            problems.append("STT_GPU_TOKEN is not set but STT_BACKEND=gpu")
        if not self.allowed_users:
            problems.append("ALLOWED_USERS is empty; the Core would accept nobody")
        for user in self.allowed_users:
            if ":" not in user:
                problems.append(f"ALLOWED_USERS entry {user!r} must be namespaced, e.g. 'tg:123'")
        try:
            _ = self.timezone
        except ValueError as exc:
            problems.append(str(exc))
        for name, project in self.projects.items():
            if not project.path.exists():
                problems.append(f"project {name!r} path does not exist: {project.path}")
        return problems


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Override the process-wide settings. Used by tests."""
    global _settings
    _settings = settings
