"""The cache in front of the provider: expiry, bounded size, and an off switch."""

from __future__ import annotations

import threading

from web_search.cache import TtlCache


def test_a_stored_value_comes_back() -> None:
    cache = TtlCache(ttl_seconds=60.0, max_entries=10)
    cache.put(("q", 5, "ru"), [{"url": "https://example.com"}])

    assert cache.get(("q", 5, "ru")) == [{"url": "https://example.com"}]
    assert cache.stats() == {"entries": 1, "hits": 1, "misses": 0}


def test_a_different_limit_is_a_different_entry() -> None:
    """Asking for ten results must not be answered with the five already cached."""
    cache = TtlCache(ttl_seconds=60.0, max_entries=10)
    cache.put(("q", 5, "ru"), ["five"])

    assert cache.get(("q", 10, "ru")) is None


def test_an_expired_entry_is_a_miss() -> None:
    cache = TtlCache(ttl_seconds=-1.0, max_entries=10)
    cache.put("q", ["stale"])

    assert cache.get("q") is None


def test_a_zero_ttl_disables_the_cache_entirely() -> None:
    cache = TtlCache(ttl_seconds=0.0, max_entries=10)
    cache.put("q", ["value"])

    assert cache.get("q") is None
    assert cache.stats()["entries"] == 0


def test_the_least_recently_used_entry_is_evicted_first() -> None:
    cache = TtlCache(ttl_seconds=60.0, max_entries=2)
    cache.put("a", ["a"])
    cache.put("b", ["b"])
    cache.get("a")  # 'a' is now the most recent, so 'b' should go
    cache.put("c", ["c"])

    assert cache.get("a") == ["a"]
    assert cache.get("c") == ["c"]
    assert cache.get("b") is None


def test_sweeping_reclaims_expired_entries() -> None:
    cache = TtlCache(ttl_seconds=-1.0, max_entries=10)
    cache._entries["q"] = (0.0, ["stale"])  # noqa: SLF001 - the sweep is what is under test

    assert cache.sweep() == 1
    assert cache.stats()["entries"] == 0


def test_concurrent_writers_do_not_lose_entries() -> None:
    """Every batch runs on the thread pool, so the cache is written from several threads."""
    cache = TtlCache(ttl_seconds=60.0, max_entries=200)

    def fill(offset: int) -> None:
        for index in range(50):
            cache.put(f"k{offset}-{index}", [index])

    threads = [threading.Thread(target=fill, args=(offset,)) for offset in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert cache.stats()["entries"] == 200
