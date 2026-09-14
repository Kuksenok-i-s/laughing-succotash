"""The fetch guard.

A deliberate duplicate of ``agent_core.search.base.guard_url``. The Core guards before it calls,
and this service guards before it connects, because the two run on different hosts and the loopback
interface each of them protects is a different one. On this host that interface may hold a SearXNG
instance and whatever else the box runs; a URL that arrived inside an untrusted web page must not be
able to reach it.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import SplitResult, quote, urlsplit, urlunsplit

# Characters that keep their meaning in a path or query and must not be re-encoded. ``%`` is in the
# list so that an already-encoded URL survives a second pass unchanged.
_SAFE_PATH = "/%:@&=+$,;~!*'()[]-._"
_SAFE_QUERY = _SAFE_PATH + "?"


class UrlRefused(ValueError):
    """The URL is not fetchable. Reported to the caller as a 400, never attempted."""


def guard_url(url: str) -> str:
    """Validate a URL for fetching, or raise ``UrlRefused``.

    Returns the URL in its ASCII form, which is not cosmetic: ``urllib`` refuses to send a request
    for anything else, and the links worth opening in a Russian conversation are mostly Cyrillic
    Wikipedia titles.
    """
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        # urlsplit is forgiving, but .port raises on a garbage port, and that is a bad request
        # rather than a bug in this service.
        raise UrlRefused(f"the URL cannot be parsed ({exc})") from exc

    if parsed.scheme not in ("http", "https"):
        raise UrlRefused(
            f"only http(s) URLs may be fetched, got {parsed.scheme or 'no'} scheme"
        )
    if not parsed.hostname:
        raise UrlRefused("the URL has no host")
    if parsed.username or parsed.password:
        # Rebuilding the URL below would drop these silently. A page we do not trust has no
        # business handing us credentials to replay, so say no out loud instead.
        raise UrlRefused("refusing to fetch a URL that carries credentials")

    for address in _resolve(parsed.hostname):
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            raise UrlRefused(f"refusing to fetch a non-public address ({address})")

    return _ascii_form(parsed, port)


def _ascii_form(parsed: SplitResult, port: int | None) -> str:
    """The URL with an IDNA host and a percent-encoded path, ready for ``urllib``."""
    scheme, _, path, query, _ = parsed
    host = parsed.hostname or ""
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        # An IPv6 literal fails the idna codec and is already ASCII, so only a genuinely
        # unusable name reaches the raise.
        if not host.isascii():
            raise UrlRefused(f"the host is not a usable name ({host})") from exc

    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port else host
    # The fragment is dropped: it never reaches the server, so keeping it only invites a mismatch
    # between the URL we checked and the one we send.
    return urlunsplit(
        (scheme, netloc, quote(path, safe=_SAFE_PATH), quote(query, safe=_SAFE_QUERY), "")
    )


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
        raise UrlRefused(f"could not resolve {hostname}") from exc

    addresses = []
    for info in infos:
        try:
            addresses.append(ipaddress.ip_address(info[4][0]))
        except ValueError:
            continue
    if not addresses:
        raise UrlRefused(f"could not resolve {hostname}")
    return addresses
