"""Configuration comes from SEARCH_* and is checked before the port is bound."""

from __future__ import annotations

import pytest

from web_search.config import DEFAULT_PORT, Settings, from_env

VALID = {
    "SEARCH_TOKEN": "t" * 40,
    "SEARCH_BRAVE_API_KEY": "k" * 32,
}


def test_defaults_apply_when_the_environment_is_empty() -> None:
    settings = from_env({})

    assert settings.host == "127.0.0.1"
    assert settings.port == DEFAULT_PORT
    assert settings.backend == "brave"
    assert settings.default_lang == "ru"
    assert settings.fetch_enabled is True


def test_every_field_reads_its_prefixed_variable() -> None:
    settings = from_env(
        {
            **VALID,
            "SEARCH_HOST": "0.0.0.0",
            "SEARCH_PORT": "19000",
            "SEARCH_BACKEND": "SearXNG",
            "SEARCH_SEARXNG_URL": "https://searx.test/",
            "SEARCH_SEARXNG_ENGINES": " google,wikipedia ",
            "SEARCH_MAX_QUERIES": "3",
            "SEARCH_DEFAULT_LANG": "EN",
            "SEARCH_PARALLEL": "8",
            "SEARCH_CACHE_TTL_SECONDS": "30.5",
            "SEARCH_FETCH_ENABLED": "no",
        }
    )

    assert settings.host == "0.0.0.0"
    assert settings.port == 19000
    assert settings.backend == "searxng"
    # A trailing slash would double up in every request path.
    assert settings.searxng_url == "https://searx.test"
    assert settings.searxng_engines == "google,wikipedia"
    assert settings.max_queries == 3
    assert settings.default_lang == "en"
    assert settings.parallel == 8
    assert settings.cache_ttl_seconds == 30.5
    assert settings.fetch_enabled is False


@pytest.mark.parametrize("value", ["true", "1", "YES", "on"])
def test_truthy_flags(value: str) -> None:
    assert from_env({"SEARCH_FETCH_ENABLED": value}).fetch_enabled is True


def test_a_blank_variable_falls_back_to_the_default() -> None:
    assert from_env({"SEARCH_HOST": "   "}).host == "127.0.0.1"


def test_a_valid_configuration_has_no_complaints() -> None:
    assert from_env(VALID).validate_runtime() == []


def test_a_missing_token_is_refused() -> None:
    problems = from_env({"SEARCH_BRAVE_API_KEY": "k" * 32}).validate_runtime()

    assert any("SEARCH_TOKEN is not set" in problem for problem in problems)


def test_a_short_token_is_refused() -> None:
    problems = from_env({**VALID, "SEARCH_TOKEN": "short"}).validate_runtime()

    assert any("shorter than 32" in problem for problem in problems)


def test_brave_without_a_key_is_refused() -> None:
    problems = from_env({"SEARCH_TOKEN": "t" * 40}).validate_runtime()

    assert any("BRAVE_API_KEY" in problem for problem in problems)


def test_searxng_without_a_url_is_refused() -> None:
    settings = Settings(token="t" * 40, backend="searxng", searxng_url="")

    assert any("SEARXNG_URL" in problem for problem in settings.validate_runtime())


def test_an_unknown_backend_is_refused() -> None:
    problems = from_env({**VALID, "SEARCH_BACKEND": "google"}).validate_runtime()

    assert any("must be brave or searxng" in problem for problem in problems)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"max_queries": 0}, "MAX_QUERIES"),
        ({"max_limit": 0}, "MAX_LIMIT"),
        ({"default_limit": 50, "max_limit": 20}, "DEFAULT_LIMIT"),
        ({"parallel": 0}, "PARALLEL"),
    ],
)
def test_nonsensical_limits_are_refused(overrides: dict, expected: str) -> None:
    settings = Settings(token="t" * 40, brave_api_key="k" * 32, **overrides)

    assert any(expected in problem for problem in settings.validate_runtime())
