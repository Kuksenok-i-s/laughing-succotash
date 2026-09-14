"""Service configuration.

Environment only, and a plain dataclass rather than pydantic-settings: this unit ships with no
runtime dependencies at all, and there is nothing sensitive here beyond the bearer token and the
upstream API key.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

ENV_PREFIX = "SEARCH_"

# Next in the run of service ports: STT 17493, OCR 17494.
DEFAULT_PORT = 17495

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


@dataclass(frozen=True, slots=True)
class Settings:
    # Loopback by default. Bind the LAN address only when Core runs on another host.
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    token: str = ""

    # brave = the hosted Search API (needs a subscription key); searxng = a self-hosted instance.
    backend: str = "brave"
    brave_api_key: str = ""
    brave_url: str = BRAVE_URL
    searxng_url: str = "http://127.0.0.1:8888"
    # SearXNG aggregates upstream engines; an empty value leaves the instance defaults alone.
    searxng_engines: str = ""

    default_limit: int = 5
    max_limit: int = 20
    # A batch cap, not a rate limit: the factcheck pass sends every claim at once.
    max_queries: int = 10
    default_lang: str = "ru"
    request_timeout: float = 10.0
    # Upstream calls in flight per batch. One by default because Brave's monthly-credit tier
    # allows one request per second, and a batch of four would simply collect 429s. The batch is
    # still worth it at this setting: what it removes is the agent's reasoning step between each
    # search, not the seconds. Raise it for a self-hosted SearXNG, which has no such limit.
    parallel: int = 1

    # Repeated queries are common: the same claim is checked twice, and a retried job re-asks.
    cache_ttl_seconds: float = 900.0
    cache_max_entries: int = 512
    sweep_interval_seconds: float = 60.0

    fetch_enabled: bool = True
    fetch_timeout: float = 15.0
    fetch_max_bytes: int = 4 * 1024 * 1024
    fetch_max_chars: int = 20_000

    user_agent: str = "personal-assistant-search/1.0"
    max_body_bytes: int = 64 * 1024

    log_level: str = "INFO"
    log_format: str = "text"

    def validate_runtime(self) -> list[str]:
        problems: list[str] = []
        if not self.token:
            problems.append(f"{ENV_PREFIX}TOKEN is not set")
        elif len(self.token) < 32:
            problems.append(f"{ENV_PREFIX}TOKEN is shorter than 32 characters")
        if self.backend not in {"brave", "searxng"}:
            problems.append(f"{ENV_PREFIX}BACKEND must be brave or searxng")
        if self.backend == "brave" and not self.brave_api_key:
            problems.append(f"{ENV_PREFIX}BRAVE_API_KEY is required for the brave backend")
        if self.backend == "searxng" and not self.searxng_url:
            problems.append(f"{ENV_PREFIX}SEARXNG_URL is required for the searxng backend")
        if self.max_queries < 1:
            problems.append(f"{ENV_PREFIX}MAX_QUERIES must be at least 1")
        if self.max_limit < 1:
            problems.append(f"{ENV_PREFIX}MAX_LIMIT must be at least 1")
        if self.default_limit > self.max_limit:
            problems.append(f"{ENV_PREFIX}DEFAULT_LIMIT is above {ENV_PREFIX}MAX_LIMIT")
        if self.parallel < 1:
            problems.append(f"{ENV_PREFIX}PARALLEL must be at least 1")
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
        return raw.strip().lower() in {"true", "1", "yes", "on"}

    return Settings(
        host=text("HOST", defaults.host),
        port=number("PORT", defaults.port),
        token=text("TOKEN", defaults.token),
        backend=text("BACKEND", defaults.backend).strip().lower(),
        brave_api_key=text("BRAVE_API_KEY", defaults.brave_api_key).strip(),
        brave_url=text("BRAVE_URL", defaults.brave_url).rstrip("/"),
        searxng_url=text("SEARXNG_URL", defaults.searxng_url).rstrip("/"),
        searxng_engines=text("SEARXNG_ENGINES", defaults.searxng_engines).strip(),
        default_limit=number("DEFAULT_LIMIT", defaults.default_limit),
        max_limit=number("MAX_LIMIT", defaults.max_limit),
        max_queries=number("MAX_QUERIES", defaults.max_queries),
        default_lang=text("DEFAULT_LANG", defaults.default_lang).strip().lower(),
        request_timeout=seconds("REQUEST_TIMEOUT", defaults.request_timeout),
        parallel=number("PARALLEL", defaults.parallel),
        cache_ttl_seconds=seconds("CACHE_TTL_SECONDS", defaults.cache_ttl_seconds),
        cache_max_entries=number("CACHE_MAX_ENTRIES", defaults.cache_max_entries),
        sweep_interval_seconds=seconds(
            "SWEEP_INTERVAL_SECONDS", defaults.sweep_interval_seconds
        ),
        fetch_enabled=flag("FETCH_ENABLED", defaults.fetch_enabled),
        fetch_timeout=seconds("FETCH_TIMEOUT", defaults.fetch_timeout),
        fetch_max_bytes=number("FETCH_MAX_BYTES", defaults.fetch_max_bytes),
        fetch_max_chars=number("FETCH_MAX_CHARS", defaults.fetch_max_chars),
        user_agent=text("USER_AGENT", defaults.user_agent),
        max_body_bytes=number("MAX_BODY_BYTES", defaults.max_body_bytes),
        log_level=text("LOG_LEVEL", defaults.log_level),
        log_format=text("LOG_FORMAT", defaults.log_format),
    )
