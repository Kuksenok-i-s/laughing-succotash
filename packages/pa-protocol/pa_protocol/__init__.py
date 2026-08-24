"""Shared wire protocol for the Telegram Gateway and the Agent Core.

Kept in one package rather than duplicated per application: a framing or method-name mismatch
between the two peers would be a silent, hard-to-diagnose production failure.
"""

from . import errors, methods
from .errors import RpcError
from .frames import (
    DEFAULT_CHUNK_SIZE,
    HEADER_SIZE,
    AudioFrame,
    FrameError,
    decode_frame,
    encode_frame,
    iter_frames,
)
from .ids import new_ulid, ulid_from_bytes, ulid_to_bytes
from .messages import PROTOCOL_VERSION, Request, Response, dumps, parse
from .peer import ConnectionClosed, RpcPeer, Transport

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "HEADER_SIZE",
    "PROTOCOL_VERSION",
    "AudioFrame",
    "ConnectionClosed",
    "FrameError",
    "Request",
    "Response",
    "RpcError",
    "RpcPeer",
    "Transport",
    "decode_frame",
    "dumps",
    "encode_frame",
    "errors",
    "iter_frames",
    "methods",
    "new_ulid",
    "parse",
    "ulid_from_bytes",
    "ulid_to_bytes",
]
