"""Telegram Gateway configuration.

The Gateway is the exposed component, so it is given the minimum it needs: a bot token, the shared
service token, and where to keep transport state. It has no Cursor credentials and no knowledge of
the assistant's data.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    telegram_bot_token: str = ""
    # Advisory only. The Core enforces its own allowlist and never trusts this one; rejecting
    # early merely avoids waking the Mac mini for a stranger.
    allowed_users: list[str] = Field(default_factory=list)

    @field_validator("allowed_users", mode="before")
    @classmethod
    def _split(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [u.strip() for u in value.split(",") if u.strip()]
        return value

    host: str = "0.0.0.0"
    port: int = 8080
    rpc_path: str = "/rpc"
    core_token: str = ""

    data_dir: Path = Path("~/.telegram-gateway")
    database_path: Path | None = None
    temp_dir: Path | None = None

    max_download_mb: int = 500
    upload_chunk_size: int = 256 * 1024
    rpc_call_timeout: float = 30.0
    # Bounds how long a Core-bound submit may block a Telegram handler. The Core answers
    # assistant.submit in milliseconds; anything slower means it is unhealthy.
    submit_timeout: float = 15.0

    status_edit_min_interval: float = 3.0
    telegram_message_limit: int = 4096
    delivery_max_attempts: int = 5
    delivery_retry_base_delay: float = 2.0

    log_level: str = "INFO"
    log_format: str = "text"

    @cached_property
    def resolved_data_dir(self) -> Path:
        path = self.data_dir.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @cached_property
    def resolved_database_path(self) -> Path:
        if self.database_path is not None:
            return self.database_path.expanduser().resolve()
        return self.resolved_data_dir / "gateway.sqlite3"

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
    def max_download_bytes(self) -> int:
        return self.max_download_mb * 1024 * 1024

    def validate_runtime(self) -> list[str]:
        problems: list[str] = []
        if not self.telegram_bot_token:
            problems.append("TELEGRAM_BOT_TOKEN is not set")
        if not self.core_token:
            problems.append("CORE_TOKEN is not set")
        elif len(self.core_token) < 32:
            problems.append("CORE_TOKEN is shorter than 32 characters")
        return problems


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings
