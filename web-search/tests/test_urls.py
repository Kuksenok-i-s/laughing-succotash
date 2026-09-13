"""The fetch guard: what it refuses, and what shape it returns for what it allows."""

from __future__ import annotations

import socket

import pytest

from web_search import urls
from web_search.urls import UrlRefused, guard_url

PUBLIC = "93.184.216.34"
PRIVATE = "10.0.7.127"


@pytest.fixture(autouse=True)
def no_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every name to one public address unless a test says otherwise.

    The guard calls getaddrinfo, and a test suite that reaches DNS fails on a train.
    """
    monkeypatch.setattr(
        urls.socket,
        "getaddrinfo",
        lambda host, port, *a, **kw: [(0, 0, 0, "", (PUBLIC, 0))],
    )


def resolves_to(monkeypatch: pytest.MonkeyPatch, *addresses: str) -> None:
    monkeypatch.setattr(
        urls.socket,
        "getaddrinfo",
        lambda host, port, *a, **kw: [(0, 0, 0, "", (a_, 0)) for a_ in addresses],
    )


# ---- refusals ------------------------------------------------------------


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/x", "gopher://h/1"])
def test_only_http_urls_are_fetchable(url: str) -> None:
    with pytest.raises(UrlRefused, match="http"):
        guard_url(url)


def test_a_url_with_no_host_is_refused() -> None:
    with pytest.raises(UrlRefused, match="no host"):
        guard_url("http:///just/a/path")


def test_a_url_carrying_credentials_is_refused() -> None:
    """Rebuilding the URL would drop them silently, which is worse than saying no."""
    with pytest.raises(UrlRefused, match="credentials"):
        guard_url("https://user:secret@example.com/page")


def test_a_garbage_port_is_a_bad_request_not_a_crash() -> None:
    with pytest.raises(UrlRefused, match="cannot be parsed"):
        guard_url("https://example.com:notaport/page")


def test_a_private_address_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    resolves_to(monkeypatch, PRIVATE)
    with pytest.raises(UrlRefused, match="non-public"):
        guard_url("https://internal.example.com/")


def test_a_name_resolving_to_both_public_and_private_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Which address comes back first is not ours to rely on, so one bad one is enough."""
    resolves_to(monkeypatch, PUBLIC, PRIVATE)
    with pytest.raises(UrlRefused, match="non-public"):
        guard_url("https://split-horizon.example.com/")


def test_a_blocked_domain_answered_with_the_null_address_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pi-hole answers blocked names with 0.0.0.0, which is not a public address."""
    resolves_to(monkeypatch, "0.0.0.0")
    with pytest.raises(UrlRefused, match="non-public"):
        guard_url("https://doubleclick.net/pixel")


def test_a_name_that_does_not_resolve_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **kw: object) -> None:
        raise socket.gaierror("NXDOMAIN")

    monkeypatch.setattr(urls.socket, "getaddrinfo", boom)
    with pytest.raises(UrlRefused, match="could not resolve"):
        guard_url("https://blocked.example.com/")


def test_a_loopback_literal_is_refused() -> None:
    with pytest.raises(UrlRefused, match="non-public"):
        guard_url("http://127.0.0.1:8888/search")


# ---- the ASCII form ------------------------------------------------------


def test_a_cyrillic_path_is_percent_encoded() -> None:
    """urllib will not send anything else, and these are the links worth opening here."""
    assert guard_url("https://ru.wikipedia.org/wiki/Торвальдс,_Линус") == (
        "https://ru.wikipedia.org/wiki/"
        "%D0%A2%D0%BE%D1%80%D0%B2%D0%B0%D0%BB%D1%8C%D0%B4%D1%81,_%D0%9B%D0%B8%D0%BD%D1%83%D1%81"
    )


def test_a_non_ascii_host_becomes_idna() -> None:
    assert guard_url("https://пример.рф/") == "https://xn--e1afmkfd.xn--p1ai/"


def test_an_already_encoded_url_is_not_encoded_twice() -> None:
    url = "https://ru.wikipedia.org/wiki/%D0%9B%D0%B8%D0%BD%D1%83%D1%81"
    assert guard_url(url) == url


def test_a_plain_url_is_returned_unchanged() -> None:
    url = "https://example.com/a/b?x=1&y=2"
    assert guard_url(url) == url


def test_the_port_survives() -> None:
    assert guard_url("https://example.com:8443/page") == "https://example.com:8443/page"


def test_the_fragment_is_dropped() -> None:
    """It never reaches the server, so keeping it only invites a mismatch."""
    assert guard_url("https://example.com/page#section") == "https://example.com/page"


def test_a_query_keeps_its_separators() -> None:
    assert guard_url("https://example.com/s?q=a+b&n=2") == "https://example.com/s?q=a+b&n=2"
