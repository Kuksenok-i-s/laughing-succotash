"""JSON-RPC 2.0 message parsing and construction.

Deliberately hand-rolled rather than Pydantic: this layer must accept anything syntactically valid
and turn malformed input into a proper JSON-RPC error response instead of raising.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .errors import INVALID_REQUEST, PARSE_ERROR, RpcError

JSONRPC_VERSION = "2.0"
PROTOCOL_VERSION = 1


@dataclass(slots=True)
class Request:
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str | int | None = None

    @property
    def is_notification(self) -> bool:
        return self.id is None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": self.method}
        if self.params:
            payload["params"] = self.params
        if self.id is not None:
            payload["id"] = self.id
        return payload


@dataclass(slots=True)
class Response:
    id: str | int | None
    result: Any = None
    error: RpcError | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": self.id}
        if self.error is not None:
            payload["error"] = self.error.to_dict()
        else:
            payload["result"] = self.result if self.result is not None else {}
        return payload


def dumps(message: Request | Response) -> str:
    return json.dumps(message.to_dict(), ensure_ascii=False, separators=(",", ":"))


def parse(raw: str | bytes) -> Request | Response:
    """Parse one JSON-RPC frame.

    Raises :class:`RpcError` for input that cannot be interpreted as either a request or a
    response, so the caller can reply with a well-formed error.
    """
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise RpcError(PARSE_ERROR, "parse_error", {"detail": str(exc)}) from exc

    if not isinstance(payload, dict):
        raise RpcError(INVALID_REQUEST, "invalid_request", {"detail": "expected a JSON object"})

    if payload.get("jsonrpc") != JSONRPC_VERSION:
        raise RpcError(
            INVALID_REQUEST, "invalid_request", {"detail": "missing or bad jsonrpc version"}
        )

    if "method" in payload:
        method = payload["method"]
        if not isinstance(method, str):
            raise RpcError(INVALID_REQUEST, "invalid_request", {"detail": "method must be a string"})
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise RpcError(
                INVALID_REQUEST, "invalid_request", {"detail": "params must be an object"}
            )
        return Request(method=method, params=params, id=payload.get("id"))

    if "result" in payload or "error" in payload:
        error = payload.get("error")
        return Response(
            id=payload.get("id"),
            result=payload.get("result"),
            error=RpcError.from_dict(error) if isinstance(error, dict) else None,
        )

    raise RpcError(INVALID_REQUEST, "invalid_request", {"detail": "neither request nor response"})
