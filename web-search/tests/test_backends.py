"""Provider payloads reduced to title/url/excerpt, including the malformed ones."""

from __future__ import annotations

import pytest

from web_search import backends
from web_search.backends import BraveBackend, SearxngBackend, build_backend
from web_search.config import Settings
from web_search.transport import UpstreamError

BRAVE_PAYLOAD = {
    "web": {
        "results": [
            {
                "title": "ENISA <strong>Threat</strong> Landscape",
                "url": "https://enisa.europa.eu/tl",
                "description": "Phishing accounted for &gt;40% of incidents.",
            },
            {
                "title": "Второй источник",
                "url": "https://example.org/b",
                "description": "ещё выдержка",
            },
        ]
    },
    "query": {"original": "phishing share"},
}

SEARXNG_PAYLOAD = {
    "results": [
        {"title": "Первый", "url": "https://example.com/1", "content": "контент один"},
        {"title": "Второй", "url": "https://example.com/2", "content": "контент два"},
        {"title": "Третий", "url": "https://example.com/3", "content": "контент три"},
    ],
    "number_of_results": 3,
}


def _respond(monkeypatch: pytest.MonkeyPatch, payload: dict) -> list[dict]:
    """Intercept the outbound call so no test reaches the network."""
    calls: list[dict] = []

    def fake_get_json(url, *, params=None, headers=None, timeout=10.0, max_bytes=0):
        calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        return payload

    monkeypatch.setattr(backends, "get_json", fake_get_json)
    return calls


def test_brave_results_lose_their_highlight_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _respond(monkeypatch, BRAVE_PAYLOAD)
    backend = BraveBackend(api_key="k" * 32, url="https://brave.test/search")

    results = backend.search("phishing share", 5, lang="en")

    assert results[0].title == "ENISA Threat Landscape"
    assert results[0].excerpt == "Phishing accounted for >40% of incidents."
    assert results[0].url == "https://enisa.europa.eu/tl"
    assert calls[0]["headers"]["X-Subscription-Token"] == "k" * 32
    assert calls[0]["params"]["search_lang"] == "en"
    assert calls[0]["params"]["text_decorations"] == "0"


def test_brave_respects_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _respond(monkeypatch, BRAVE_PAYLOAD)
    backend = BraveBackend(api_key="k" * 32, url="https://brave.test/search")

    assert len(backend.search("q", 1, lang="ru")) == 1
    assert calls[0]["params"]["count"] == "1"


def test_brave_with_no_web_block_returns_nothing_rather_than_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _respond(monkeypatch, {"query": {"original": "ничего не найдено"}})
    backend = BraveBackend(api_key="k" * 32, url="https://brave.test/search")

    assert backend.search("ничего", 5, lang="ru") == []


def test_searxng_slices_a_full_page_to_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _respond(monkeypatch, SEARXNG_PAYLOAD)
    backend = SearxngBackend(base_url="https://searx.test", engines="google,wikipedia")

    results = backend.search("вопрос", 2, lang="ru")

    assert [result.title for result in results] == ["Первый", "Второй"]
    assert calls[0]["url"] == "https://searx.test/search"
    assert calls[0]["params"]["format"] == "json"
    assert calls[0]["params"]["language"] == "ru"
    assert calls[0]["params"]["engines"] == "google,wikipedia"


def test_searxng_omits_the_engine_filter_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _respond(monkeypatch, SEARXNG_PAYLOAD)
    backend = SearxngBackend(base_url="https://searx.test")

    backend.search("вопрос", 3, lang="ru")

    assert "engines" not in calls[0]["params"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": None},
        {"results": "строка"},
        {"results": [None, 7, "x"]},
        {"results": [{"title": "без url", "content": "x"}]},
    ],
)
def test_a_malformed_payload_yields_no_results_rather_than_an_exception(
    monkeypatch: pytest.MonkeyPatch, payload: dict
) -> None:
    """A renamed field should cost one query, not the whole batch."""
    _respond(monkeypatch, payload)
    backend = SearxngBackend(base_url="https://searx.test")

    assert backend.search("что-нибудь", 5, lang="ru") == []


def test_a_result_without_a_title_falls_back_to_its_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _respond(monkeypatch, {"results": [{"url": "https://example.com/x", "content": "тело"}]})
    backend = SearxngBackend(base_url="https://searx.test")

    assert backend.search("q", 5, lang="ru")[0].title == "https://example.com/x"


def test_health_turns_false_after_an_upstream_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real query would spend quota, so Brave health reports the last actual call."""

    def failing(*_args, **_kwargs):
        raise UpstreamError("HTTP 401 unauthorized", status=401)

    monkeypatch.setattr(backends, "get_json", failing)
    backend = BraveBackend(api_key="k" * 32, url="https://brave.test/search")
    assert backend.probe() is True

    with pytest.raises(UpstreamError):
        backend.search("q", 5, lang="ru")

    assert backend.probe() is False


def test_brave_without_a_key_reports_itself_unhealthy() -> None:
    assert BraveBackend(api_key="", url="https://brave.test/search").probe() is False


def test_the_backend_is_chosen_by_configuration() -> None:
    brave = build_backend(Settings(backend="brave", brave_api_key="k" * 32))
    searxng = build_backend(Settings(backend="searxng", searxng_url="https://searx.test"))

    assert brave.name == "brave"
    assert searxng.name == "searxng"
