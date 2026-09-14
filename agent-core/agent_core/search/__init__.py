"""External search: the contract, and a client for the service that implements it."""

from .base import MAX_FETCH_CHARS, SearchError, SearchProvider, SearchResult, guard_url, truncate
from .remote_service import RemoteSearchProvider

__all__ = [
    "MAX_FETCH_CHARS",
    "RemoteSearchProvider",
    "SearchError",
    "SearchProvider",
    "SearchResult",
    "guard_url",
    "truncate",
]
