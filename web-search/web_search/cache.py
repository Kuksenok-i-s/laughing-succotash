"""A TTL cache in front of the upstream provider.

This is the cheapest speed win available here. A factcheck pass asks about several claims at once
and often asks the same thing twice — the second turn re-checks a disputed number, a retried job
repeats the whole batch, and two videos in a playlist cite the same study. None of that should
become another billed round trip to Brave.

Entries are small and short-lived by design: a stale search result is worse than a slow one, so the
TTL is minutes rather than hours.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any


class TtlCache:
    """Thread-safe, size-bounded, least-recently-used eviction."""

    def __init__(self, *, ttl_seconds: float, max_entries: int) -> None:
        self._ttl = ttl_seconds
        self._max = max(1, max_entries)
        self._entries: OrderedDict[Any, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: Any) -> Any | None:
        if self._ttl <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            expires_at, value = entry
            if expires_at <= now:
                del self._entries[key]
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: Any, value: Any) -> None:
        if self._ttl <= 0:
            return
        with self._lock:
            self._entries[key] = (time.monotonic() + self._ttl, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def sweep(self) -> int:
        """Drop expired entries. Lookups already ignore them; this reclaims the memory."""
        now = time.monotonic()
        with self._lock:
            stale = [key for key, (expires_at, _) in self._entries.items() if expires_at <= now]
            for key in stale:
                del self._entries[key]
        return len(stale)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
            }
