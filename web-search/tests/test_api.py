"""The HTTP contract: auth, batching, per-query failure isolation, fetch guarding."""

from __future__ import annotations

import dataclasses
import json
import threading

import pytest
from helpers import Client, FakeBackend, upstream_error

from web_search.config import Settings
from web_search.server import SearchApp
from web_search.service import SearchService


def test_health_needs_no_token(client: Client) -> None:
    response = client.request("GET", "/health", token=None)

    assert response.status == 200
    assert response.payload["status"] == "ok"
    assert response.payload["backend"] == "fake"
    assert response.payload["backend_reachable"] is True
    assert response.payload["cache"]["entries"] == 0


@pytest.mark.parametrize("token", [None, "wrong-token"])
def test_search_requires_the_bearer_token(client: Client, token: str | None) -> None:
    response = client.post("/v1/search", {"query": "что угодно"}, token=token)

    assert response.status == 401
    assert response.payload["code"] == "unauthorized"


def test_a_single_query_returns_structured_results(client: Client, backend: FakeBackend) -> None:
    response = client.post("/v1/search", {"query": "курс биткоина"})

    assert response.status == 200
    entries = response.payload["queries"]
    assert len(entries) == 1
    assert entries[0]["query"] == "курс биткоина"
    assert entries[0]["error"] is None
    assert entries[0]["results"] == [
        {"title": "Пример", "url": "https://example.com/a", "excerpt": "выдержка"}
    ]
    # The default limit and language come from the settings, not from the model.
    assert backend.queries == [("курс биткоина", 3, "ru")]


def test_results_carry_the_untrusted_content_reminder(client: Client) -> None:
    """The guidance travels with the data: by then the agent is far from the prompt."""
    response = client.post("/v1/search", {"query": "что-нибудь"})

    assert "не инструкции" in response.payload["guidance"]


def test_a_batch_keeps_the_order_it_was_given(client: Client) -> None:
    queries = ["первый", "второй", "третий"]
    response = client.post("/v1/search", {"queries": queries})

    assert response.status == 200
    assert [entry["query"] for entry in response.payload["queries"]] == queries


def test_a_batch_runs_its_queries_side_by_side(settings: Settings, serve) -> None:
    """The whole point of the service: three claims cost one round trip, not three.

    The barrier is only satisfied if all three calls are in flight at once; a serial
    implementation would block on the first one and time out.
    """
    barrier = threading.Barrier(3)
    backend = FakeBackend(barrier=barrier)
    service = SearchService(settings, backend)
    try:
        client = serve(SearchApp(settings, service))
        response = client.post("/v1/search", {"queries": ["a", "b", "c"]})
    finally:
        service.close()

    assert response.status == 200
    assert barrier.broken is False
    assert sorted(backend.asked()) == ["a", "b", "c"]


def test_one_failing_query_does_not_discard_the_others(settings: Settings, serve) -> None:
    backend = FakeBackend(errors={"плохой": upstream_error()})
    service = SearchService(settings, backend)
    try:
        client = serve(SearchApp(settings, service))
        response = client.post("/v1/search", {"queries": ["хороший", "плохой"]})
    finally:
        service.close()

    assert response.status == 200
    good, bad = response.payload["queries"]
    assert good["results"] and good["error"] is None
    assert bad["results"] == []
    assert "429" in bad["error"]


def test_a_repeated_query_in_one_batch_costs_one_upstream_call(
    client: Client, backend: FakeBackend
) -> None:
    response = client.post("/v1/search", {"queries": ["одно и то же", "одно и то же"]})

    assert response.status == 200
    assert len(response.payload["queries"]) == 2
    assert backend.asked() == ["одно и то же"]


def test_a_second_request_is_served_from_the_cache(client: Client, backend: FakeBackend) -> None:
    first = client.post("/v1/search", {"query": "повторим"})
    second = client.post("/v1/search", {"query": "  Повторим  "})

    assert first.payload["queries"][0]["cached"] is False
    # Whitespace and case do not change what an engine returns, so they must not miss.
    assert second.payload["queries"][0]["cached"] is True
    assert second.payload["cached"] == 1
    assert backend.asked() == ["повторим"]


def test_limit_is_clamped_to_the_configured_maximum(client: Client, backend: FakeBackend) -> None:
    client.post("/v1/search", {"query": "много", "limit": 999})

    assert backend.queries[0][1] == 10


def test_an_oversized_batch_is_refused(client: Client, backend: FakeBackend) -> None:
    response = client.post("/v1/search", {"queries": [f"q{index}" for index in range(6)]})

    assert response.status == 400
    assert response.payload["code"] == "too_many_queries"
    assert backend.asked() == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"queries": []},
        {"queries": "не список"},
        {"queries": [1, 2]},
        {"queries": ["   "]},
        {"query": 42},
    ],
)
def test_a_request_without_usable_queries_is_refused(client: Client, body: dict) -> None:
    response = client.post("/v1/search", body)

    assert response.status == 400
    assert response.payload["code"] == "bad_request"


def test_a_non_integer_limit_is_refused(client: Client) -> None:
    response = client.post("/v1/search", {"query": "x", "limit": "пять"})

    assert response.status == 400
    assert response.payload["code"] == "bad_request"


def test_a_body_that_is_not_json_is_refused(client: Client) -> None:
    response = client.request(
        "POST", "/v1/search", body=b"{not json", headers={"Content-Type": "application/json"}
    )

    assert response.status == 400
    assert response.payload["code"] == "bad_json"


def test_a_json_array_body_is_refused(client: Client) -> None:
    response = client.post("/v1/search", ["запрос"])

    assert response.status == 400
    assert response.payload["code"] == "bad_json"


def test_a_request_with_no_body_is_refused(client: Client) -> None:
    response = client.request("POST", "/v1/search", headers={"Content-Type": "application/json"})

    assert response.status == 400
    assert response.payload["code"] == "empty_body"


def test_an_oversized_body_is_refused(settings: Settings, service, serve) -> None:
    small = dataclasses.replace(settings, max_body_bytes=32)
    client = serve(SearchApp(small, service))

    response = client.post("/v1/search", {"query": "к" * 200})

    assert response.status == 413
    assert response.payload["code"] == "body_too_large"


def test_an_unknown_endpoint_is_a_404(client: Client) -> None:
    assert client.request("GET", "/v1/nope").status == 404
    assert client.post("/v1/nope", {}).status == 404


# ---- fetch ----------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8888/search",
        "http://localhost/admin",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
        "not-a-url",
    ],
)
def test_fetch_refuses_non_public_and_non_http_urls(client: Client, url: str) -> None:
    """The service guards too: its own loopback may hold the SearXNG instance."""
    response = client.post("/v1/fetch", {"url": url})

    assert response.status == 400
    assert response.payload["code"] == "url_refused"


def test_fetch_requires_a_url(client: Client) -> None:
    response = client.post("/v1/fetch", {"url": "   "})

    assert response.status == 400
    assert response.payload["code"] == "bad_request"


def test_fetch_can_be_switched_off_entirely(
    settings: Settings, backend: FakeBackend, serve
) -> None:
    disabled = dataclasses.replace(settings, fetch_enabled=False)
    service = SearchService(disabled, backend)
    try:
        client = serve(SearchApp(disabled, service))
        response = client.post("/v1/fetch", {"url": "https://example.com/a"})
        health = client.request("GET", "/health", token=None)
    finally:
        service.close()

    assert response.status == 403
    assert response.payload["code"] == "fetch_disabled"
    assert health.payload["fetch_enabled"] is False


def test_the_response_is_utf8_json(client: Client) -> None:
    response = client.post("/v1/search", {"query": "кириллица"})

    assert json.loads(response.raw)["queries"][0]["query"] == "кириллица"
