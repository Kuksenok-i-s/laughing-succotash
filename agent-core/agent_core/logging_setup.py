"""Logging configuration.

Structured fields are logged; content is not. Transcripts, note bodies, memory values and
credentials never reach the log — see the logging policy in the README.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_NOISY = ("aiohttp.access", "websockets.client", "websockets.server", "asyncio")

# Substrings that mark a field as secret regardless of which logger emitted it.
_SENSITIVE_KEYS = ("token", "password", "secret", "authorization", "api_key", "cookie")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in getattr(record, "__dict__", {}).items():
            if key.startswith("_") or key in logging.LogRecord("", 0, "", 0, "", (), None).__dict__:
                continue
            if key in ("args", "msg", "exc_info", "exc_text", "stack_info"):
                continue
            if any(marker in key.lower() for marker in _SENSITIVE_KEYS):
                payload[key] = "***"
            else:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
