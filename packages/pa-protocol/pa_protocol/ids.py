"""ULID generation and binary conversion.

ULIDs are used for every identifier in the protocol because they sort lexicographically by
creation time, which makes ordered replay of durable events a plain ``ORDER BY`` and makes logs
readable without a join.
"""

from __future__ import annotations

import os
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_DECODE = {c: i for i, c in enumerate(_CROCKFORD)}

_last_ms = 0
_last_randomness = 0


def new_ulid() -> str:
    """Return a new ULID string, monotonic within the same millisecond."""
    global _last_ms, _last_randomness

    ms = int(time.time() * 1000)
    if ms == _last_ms:
        # Same millisecond: increment instead of re-randomising so that IDs created in a tight
        # loop still sort in creation order.
        _last_randomness += 1
        if _last_randomness >= (1 << 80):
            ms = _last_ms + 1
            _last_ms = ms
            _last_randomness = int.from_bytes(os.urandom(10), "big")
    else:
        _last_ms = ms
        _last_randomness = int.from_bytes(os.urandom(10), "big")

    return _encode(ms, _last_randomness)


def _encode(ms: int, randomness: int) -> str:
    value = (ms << 80) | randomness
    chars = []
    for _ in range(26):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def ulid_to_bytes(value: str) -> bytes:
    """Encode a 26-character ULID into its 16-byte binary form."""
    if len(value) != 26:
        raise ValueError(f"ULID must be 26 characters, got {len(value)}")
    number = 0
    for char in value.upper():
        digit = _DECODE.get(char)
        if digit is None:
            raise ValueError(f"invalid ULID character: {char!r}")
        number = (number << 5) | digit
    return number.to_bytes(16, "big")


def ulid_from_bytes(raw: bytes) -> str:
    """Decode a 16-byte binary ULID back into its canonical string form."""
    if len(raw) != 16:
        raise ValueError(f"binary ULID must be 16 bytes, got {len(raw)}")
    return _encode(*divmod(int.from_bytes(raw, "big"), 1 << 80))
