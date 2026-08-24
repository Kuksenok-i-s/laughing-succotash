"""The fetch guard.

`web_fetch` is the only tool that could turn a URL found in an untrusted document into a request
from this machine, so the guard is tested on its own rather than through a provider.
"""

from __future__ import annotations

import pytest

from agent_core.search.base import SearchError, guard_url, truncate


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/article",
        "http://example.com/",
        "https://example.com:8443/x?y=1",
    ],
)
def test_public_http_urls_are_allowed(url: str) -> None:
    assert guard_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8931/mcp",  # this process's own MCP server
        "http://localhost:8080/rpc",
        "http://[::1]:8931/mcp",
        "http://10.0.0.5/admin",
        "http://192.168.1.1/",
        "http://172.16.4.2/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
    ],
)
def test_non_public_addresses_are_refused(url: str) -> None:
    with pytest.raises(SearchError):
        guard_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///Users/me/.ssh/id_rsa",
        "ftp://example.com/x",
        "gopher://example.com",
        "data:text/html,<script>",
        "not-a-url",
        "https://",
    ],
)
def test_only_http_schemes_with_a_host_are_allowed(url: str) -> None:
    with pytest.raises(SearchError):
        guard_url(url)


def test_an_unresolvable_host_is_an_error_not_a_pass() -> None:
    with pytest.raises(SearchError):
        guard_url("https://this-host-does-not-exist.invalid/x")


def test_long_pages_are_truncated_visibly() -> None:
    """Silently dropping the tail would let the agent reason over a page it only half received."""
    result = truncate("а" * 500, limit=100)

    assert len(result) < 500
    assert "обрезана" in result


def test_short_pages_are_returned_untouched() -> None:
    assert truncate("коротко", limit=100) == "коротко"
