"""The external-search contract.

No provider is wired up yet, so `web_search` and `web_fetch` are simply not registered and the
assistant has no network reach at all. This module exists so that adding one later is a matter of
implementing an interface rather than deciding a policy under time pressure — and the policy is the
hard part.

Two constraints are baked into the contract rather than left to the implementation:

Results are structured. Handing the agent a raw HTML page means handing it whatever that page's
author wrote, in a form where an instruction and a paragraph look identical. A title, a URL and an
excerpt can still be malicious content, but at least it is content of a known shape and bounded
size.

Fetching is not general networking. `web_fetch` takes a URL and returns readable text; it cannot
POST, cannot follow a redirect to a private address, and cannot reach the loopback interface —
where, among other things, this process's own MCP server is listening. `guard_url` is what enforces
that, and it is deliberately here in the contract rather than in a provider that might forget it.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

# Fetched pages are truncated: an agent reasoning over half a megabyte of navigation chrome is
# both expensive and worse at it than one reading the first few thousand words.
MAX_FETCH_CHARS = 20_000


class SearchError(RuntimeError):
    """The search or fetch could not be completed. Reported to the agent as a tool error."""


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    excerpt: str

    def as_dict(self) -> dict[str, Any]:
        return {"title": self.title, "url": self.url, "excerpt": self.excerpt}


class SearchProvider(Protocol):
    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Structured results, newest-relevance first. Never raw markup."""
        ...

    async def fetch(self, url: str) -> dict[str, Any]:
        """Readable text for one URL, truncated to ``MAX_FETCH_CHARS``."""
        ...


def guard_url(url: str) -> str:
    """Validate a URL for fetching, or raise.

    Blocks anything that is not plain HTTP(S) and anything that resolves to a private, loopback or
    link-local address. Without this, `web_fetch` would be a way to ask the assistant to read the
    Mac mini's own loopback services — including its MCP endpoint and any other local daemon — from
    a URL that could have arrived inside an untrusted document.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SearchError(f"only http(s) URLs may be fetched, got {parsed.scheme or 'no'} scheme")
    if not parsed.hostname:
        raise SearchError("the URL has no host")

    for address in _resolve(parsed.hostname):
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            raise SearchError(f"refusing to fetch a non-public address ({address})")

    return url


def _resolve(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Every address the host resolves to.

    All of them are checked, not just the first: a name that resolves to both a public and a
    private address must be refused, and which one comes back first is not ours to rely on.
    """
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        return [literal]

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise SearchError(f"could not resolve {hostname}") from exc

    addresses = []
    for info in infos:
        try:
            addresses.append(ipaddress.ip_address(info[4][0]))
        except ValueError:
            continue
    if not addresses:
        raise SearchError(f"could not resolve {hostname}")
    return addresses


def truncate(text: str, limit: int = MAX_FETCH_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[…страница обрезана]"
