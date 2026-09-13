"""Outbound HTTP: the byte cap, and the guard that has to survive a redirect."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest

from web_search.transport import UpstreamError, get
from web_search.urls import UrlRefused, guard_url


class Upstream(BaseHTTPRequestHandler):
    """Answers the handful of paths these tests need, and nothing else."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/redirect-to-loopback"):
            # The whole point: a host we were allowed to reach sending us somewhere we were not.
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9/secret")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.startswith("/redirect-to-page"):
            self.send_response(302)
            self.send_header("Location", "/page")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.startswith("/huge"):
            body = b"x" * 50_000
        else:
            body = "<p>страница</p>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Quiet: the suite's output is for failures."""


@pytest.fixture
def upstream() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def allow_everything(url: str) -> str:
    """Stands in for the real guard when the test is not about guarding."""
    return url


def test_a_plain_get_returns_the_body(upstream: str) -> None:
    response = get(f"{upstream}/page", timeout=5.0)

    assert response.status == 200
    assert "страница" in response.text()


def test_the_byte_cap_is_enforced_while_reading(upstream: str) -> None:
    """Content-Length is a claim, not a promise, so the cap cannot be based on it."""
    response = get(f"{upstream}/huge", timeout=5.0, max_bytes=1000)

    assert len(response.body) == 1000


def test_a_redirect_to_loopback_is_refused_even_from_an_allowed_host(upstream: str) -> None:
    """Guarding only the URL we were handed buys nothing if urllib then follows a 302 anywhere."""
    with pytest.raises(UrlRefused, match="non-public"):
        get(f"{upstream}/redirect-to-loopback", timeout=5.0, guard=guard_url)


def test_without_a_guard_redirects_are_followed_as_before(upstream: str) -> None:
    """The search backends pass no guard: one of them is a SearXNG on loopback."""
    response = get(f"{upstream}/redirect-to-page", timeout=5.0)

    assert "страница" in response.text()


def test_a_guard_that_allows_the_target_still_follows_the_redirect(upstream: str) -> None:
    response = get(f"{upstream}/redirect-to-page", timeout=5.0, guard=allow_everything)

    assert response.status == 200
    assert "страница" in response.text()


def test_the_guard_sees_every_hop(upstream: str) -> None:
    seen: list[str] = []

    def record(url: str) -> str:
        seen.append(url)
        return url

    get(f"{upstream}/redirect-to-page", timeout=5.0, guard=record)

    assert [u.endswith("/page") for u in seen] == [True]


def test_an_unreachable_host_is_an_upstream_error() -> None:
    with pytest.raises(UpstreamError, match="unreachable"):
        get("http://127.0.0.1:9/nothing", timeout=2.0)
