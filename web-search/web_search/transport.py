"""Outbound HTTP, on the standard library.

Two callers with different needs share it: the search backends read a small JSON document from a
known API, and ``/v1/fetch`` reads an arbitrary page from a host that may be hostile. Hence the
byte cap, which is enforced while reading rather than trusted from ``Content-Length`` — a server
is free to lie about that, and a stream with no length at all is the normal case.
"""

from __future__ import annotations

import gzip
import json
import logging
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode

log = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    """The upstream refused, timed out or answered with something unusable."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class Response:
    __slots__ = ("status", "body", "content_type", "url")

    def __init__(self, status: int, body: bytes, content_type: str, url: str) -> None:
        self.status = status
        self.body = body
        self.content_type = content_type
        self.url = url

    def text(self) -> str:
        """Decode with the declared charset, falling back to UTF-8 with replacement.

        A page that lies about its encoding should produce mangled characters, not an exception:
        the agent can still read most of a mis-declared page, and a hard failure would lose it.
        """
        charset = ""
        for part in self.content_type.split(";")[1:]:
            name, _, value = part.strip().partition("=")
            if name.strip().lower() == "charset":
                charset = value.strip().strip('"').lower()
        for candidate in (charset, "utf-8"):
            if not candidate:
                continue
            try:
                return self.body.decode(candidate)
            except (LookupError, UnicodeDecodeError):
                continue
        return self.body.decode("utf-8", errors="replace")


class _GuardedRedirects(urllib.request.HTTPRedirectHandler):
    """Runs the caller's guard over every redirect target before following it.

    Without this, guarding the URL the caller asked for buys nothing: a page on a public host is
    free to answer 302 to ``http://127.0.0.1:8888`` or to a box on the LAN, and urllib follows it
    without asking. The guard raises, which aborts the whole request.
    """

    def __init__(self, guard: Callable[[str], str]) -> None:
        self._guard = guard

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        self._guard(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def get(
    url: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
    max_bytes: int = 1024 * 1024,
    guard: Callable[[str], str] | None = None,
) -> Response:
    """One GET. ``guard`` is applied to every redirect target when given.

    It is optional because the search backends are configured addresses, and one of them is a
    SearXNG on loopback that any sensible guard would refuse.
    """
    target = f"{url}?{urlencode(params)}" if params else url
    request = urllib.request.Request(target, headers=headers or {}, method="GET")
    opener = urllib.request.build_opener(_GuardedRedirects(guard)) if guard else None
    try:
        with (opener.open if opener else urllib.request.urlopen)(
            request, timeout=timeout
        ) as response:
            body = _read_capped(response, max_bytes)
            return Response(
                status=response.status,
                body=_decompress(body, response.headers.get("Content-Encoding", "")),
                content_type=response.headers.get("Content-Type", ""),
                # The final URL after redirects; a result should cite where it landed.
                url=response.geturl(),
            )
    except urllib.error.HTTPError as exc:
        detail = _short(exc.read())
        raise UpstreamError(f"HTTP {exc.code} {detail}".strip(), status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise UpstreamError(f"{url} unreachable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise UpstreamError(f"{url} timed out after {timeout:.0f}s") from exc


def get_json(
    url: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
    max_bytes: int = 2 * 1024 * 1024,
) -> dict[str, Any]:
    response = get(
        url, params=params, headers=headers, timeout=timeout, max_bytes=max_bytes
    )
    try:
        payload = json.loads(response.text())
    except json.JSONDecodeError as exc:
        raise UpstreamError(f"{url} returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise UpstreamError(f"{url} returned {type(payload).__name__}, expected an object")
    return payload


def _read_capped(stream: Any, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = max_bytes
    while remaining > 0:
        chunk = stream.read(min(64 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decompress(body: bytes, encoding: str) -> bytes:
    """Undo gzip or deflate if the server applied it anyway.

    We do not advertise an Accept-Encoding, so a compliant server sends identity. Not every server
    is compliant, and a gzip blob decoded as text is unreadable rather than merely wrong.
    """
    name = encoding.strip().lower()
    try:
        if name == "gzip":
            return gzip.decompress(body)
        if name == "deflate":
            return zlib.decompress(body, -zlib.MAX_WBITS)
    except (OSError, zlib.error) as exc:
        log.debug("could not decompress a %s body: %s", name, exc)
    return body


def _short(raw: bytes, limit: int = 200) -> str:
    text = raw.decode("utf-8", errors="replace").strip().replace("\n", " ")
    return text[:limit]
