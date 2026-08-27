"""Logging configuration.

Structured fields are logged; image contents and full transcripts never reach the log.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_SENSITIVE_KEYS = ("token", "password", "secret", "authorization", "api_key")
_NOISY = ("httpx", "httpcore", "urllib3")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        reserved = logging.LogRecord("", 0, "", 0, "", (), None).__dict__
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in reserved:
                continue
            if key in ("args", "msg", "exc_info", "exc_text", "stack_info"):
                continue
            payload[key] = "***" if any(m in key.lower() for m in _SENSITIVE_KEYS) else value
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
