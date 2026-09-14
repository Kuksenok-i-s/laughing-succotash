"""Protocol error codes shared by both peers.

Error ``message`` values are stable machine-readable identifiers; human-facing text is produced by
the Gateway from the code, so that a wording change never breaks a caller.
"""

from __future__ import annotations

from typing import Any

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

PROTOCOL_VERSION_UNSUPPORTED = -32000
NOT_READY = -32001
UNAUTHORIZED_USER = -32002
RATE_LIMITED = -32003

UNKNOWN_UPLOAD = -32010
AUDIO_TOO_LARGE = -32011
AUDIO_TOO_LONG = -32012
UPLOAD_INCOMPLETE = -32013

JOB_NOT_FOUND = -32020
JOB_ALREADY_FINISHED = -32021

AGENT_UNAVAILABLE = -32030
AGENT_FAILED = -32031

STT_UNAVAILABLE = -32040
STT_FAILED = -32041

OCR_UNAVAILABLE = -32042
OCR_FAILED = -32043
IMAGE_TOO_LARGE = -32044
INVALID_IMAGE = -32045

TELEGRAM_SEND_FAILED = -32050
TELEGRAM_BLOCKED = -32051

NAMES = {
    PARSE_ERROR: "parse_error",
    INVALID_REQUEST: "invalid_request",
    METHOD_NOT_FOUND: "method_not_found",
    INVALID_PARAMS: "invalid_params",
    INTERNAL_ERROR: "internal_error",
    PROTOCOL_VERSION_UNSUPPORTED: "protocol_version_unsupported",
    NOT_READY: "not_ready",
    UNAUTHORIZED_USER: "unauthorized_user",
    RATE_LIMITED: "rate_limited",
    UNKNOWN_UPLOAD: "unknown_upload",
    AUDIO_TOO_LARGE: "audio_too_large",
    AUDIO_TOO_LONG: "audio_too_long",
    UPLOAD_INCOMPLETE: "upload_incomplete",
    JOB_NOT_FOUND: "job_not_found",
    JOB_ALREADY_FINISHED: "job_already_finished",
    AGENT_UNAVAILABLE: "agent_unavailable",
    AGENT_FAILED: "agent_failed",
    STT_UNAVAILABLE: "stt_unavailable",
    STT_FAILED: "stt_failed",
    OCR_UNAVAILABLE: "ocr_unavailable",
    OCR_FAILED: "ocr_failed",
    IMAGE_TOO_LARGE: "image_too_large",
    INVALID_IMAGE: "invalid_image",
    TELEGRAM_SEND_FAILED: "telegram_send_failed",
    TELEGRAM_BLOCKED: "telegram_blocked",
}

RETRYABLE = frozenset(
    {
        NOT_READY,
        RATE_LIMITED,
        UPLOAD_INCOMPLETE,
        AGENT_UNAVAILABLE,
        STT_UNAVAILABLE,
        OCR_UNAVAILABLE,
        TELEGRAM_SEND_FAILED,
    }
)


class RpcError(Exception):
    """A JSON-RPC error, raised by handlers and marshalled into an error response."""

    def __init__(self, code: int, message: str | None = None, data: Any = None) -> None:
        self.code = code
        self.message = message or NAMES.get(code, "error")
        self.data = data
        super().__init__(f"{self.code} {self.message}")

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            payload["data"] = self.data
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RpcError:
        return cls(
            code=payload.get("code", INTERNAL_ERROR),
            message=payload.get("message"),
            data=payload.get("data"),
        )
