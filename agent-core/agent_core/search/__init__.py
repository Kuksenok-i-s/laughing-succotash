"""External search: the contract only. No provider is configured, so the tools stay unregistered."""

from .base import MAX_FETCH_CHARS, SearchError, SearchProvider, SearchResult, guard_url, truncate

__all__ = [
    "MAX_FETCH_CHARS",
    "SearchError",
    "SearchProvider",
    "SearchResult",
    "guard_url",
    "truncate",
]
