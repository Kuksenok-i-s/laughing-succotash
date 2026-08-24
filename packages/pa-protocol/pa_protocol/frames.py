"""Binary frame codec for the audio data plane.

Audio bytes travel as binary WebSocket frames rather than Base64 in JSON: a 60-minute recording
would grow by a third and would have to pass through the JSON parser.

Each frame is a fixed 32-byte header followed by payload::

    offset  size  field
    0       4     magic       b"PAUP"
    4       1     version     1
    5       1     flags       bit0 = final chunk
    6       2     reserved    zero
    8       16    upload_id   binary ULID
    24      8     offset      uint64 big-endian, byte offset of this chunk
    32      ...   payload

Carrying the offset in every frame makes each one self-describing, so a duplicated or reordered
frame after a reconnect is detected rather than silently corrupting the file.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .ids import ulid_from_bytes, ulid_to_bytes

MAGIC = b"PAUP"
VERSION = 1
HEADER_SIZE = 32
FLAG_FINAL = 0x01

DEFAULT_CHUNK_SIZE = 256 * 1024

_HEADER = struct.Struct(">4sBBH16sQ")


class FrameError(ValueError):
    """Raised when a binary frame cannot be decoded."""


@dataclass(frozen=True, slots=True)
class AudioFrame:
    upload_id: str
    offset: int
    payload: bytes
    final: bool = False

    @property
    def end_offset(self) -> int:
        return self.offset + len(self.payload)


def encode_frame(frame: AudioFrame) -> bytes:
    if frame.offset < 0:
        raise FrameError("offset must be non-negative")
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        FLAG_FINAL if frame.final else 0,
        0,
        ulid_to_bytes(frame.upload_id),
        frame.offset,
    )
    return header + frame.payload


def decode_frame(raw: bytes) -> AudioFrame:
    if len(raw) < HEADER_SIZE:
        raise FrameError(f"frame shorter than header: {len(raw)} bytes")

    magic, version, flags, reserved, upload_raw, offset = _HEADER.unpack_from(raw, 0)

    if magic != MAGIC:
        raise FrameError(f"bad magic: {magic!r}")
    if version != VERSION:
        raise FrameError(f"unsupported frame version: {version}")
    if reserved != 0:
        raise FrameError("reserved field must be zero")

    return AudioFrame(
        upload_id=ulid_from_bytes(upload_raw),
        offset=offset,
        payload=raw[HEADER_SIZE:],
        final=bool(flags & FLAG_FINAL),
    )


def iter_frames(
    upload_id: str,
    data: bytes,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    start_offset: int = 0,
):
    """Split ``data`` into encoded frames, marking the last one final.

    An empty payload still yields one final frame so that a zero-byte upload has a well-defined
    terminator rather than being indistinguishable from an abandoned one.
    """
    if chunk_size <= 0:
        raise FrameError("chunk_size must be positive")

    if not data:
        yield encode_frame(AudioFrame(upload_id, start_offset, b"", final=True))
        return

    total = len(data)
    for position in range(0, total, chunk_size):
        chunk = data[position : position + chunk_size]
        yield encode_frame(
            AudioFrame(
                upload_id=upload_id,
                offset=start_offset + position,
                payload=chunk,
                final=position + len(chunk) >= total,
            )
        )
