"""The two search backends.

Both are reduced to the same three fields — title, URL, excerpt — because that is the whole
contract the assistant is allowed to see. Handing a model a provider's full payload means handing
it ranking metadata, ad slots and sponsored blocks in a shape where an instruction and a snippet
look identical.

Brave is the hosted API: one key, one request, a bounded JSON document. SearXNG is a self-hosted
aggregator, which trades the key for an instance to run and needs ``json`` enabled in its
``search.formats``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from .extract import strip_markup
from .transport import UpstreamError, get_json

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Result:
    title: str
    url: str
    excerpt: str

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "excerpt": self.excerpt}


class Backend(Protocol):
    @property
    def name(self) -> str: ...

    def search(self, query: str, limit: int, *, lang: str) -> list[Result]: ...

    def probe(self) -> bool: ...


class BraveBackend:
    def __init__(
        self,
        *,
        api_key: str,
        url: str,
        timeout: float = 10.0,
        user_agent: str = "personal-assistant-search/1.0",
    ) -> None:
        self._api_key = api_key
        self._url = url
        self._timeout = timeout
        self._user_agent = user_agent
        self._healthy = bool(api_key)

    @property
    def name(self) -> str:
        return "brave"

    def probe(self) -> bool:
        # A real query would spend quota, so health reports the outcome of the last actual call.
        return self._healthy

    def search(self, query: str, limit: int, *, lang: str) -> list[Result]:
        params = {
            "q": query,
            "count": str(limit),
            # Brave wraps matched terms in <strong> unless asked not to.
            "text_decorations": "0",
            "safesearch": "off",
        }
        if lang:
            params["search_lang"] = lang
        try:
            payload = get_json(
                self._url,
                params=params,
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self._api_key,
                    "User-Agent": self._user_agent,
                },
                timeout=self._timeout,
            )
        except UpstreamError:
            self._healthy = False
            raise
        self._healthy = True
        return _results(payload.get("web"), "description", limit)


class SearxngBackend:
    def __init__(
        self,
        *,
        base_url: str,
        engines: str = "",
        timeout: float = 10.0,
        user_agent: str = "personal-assistant-search/1.0",
    ) -> None:
        self._base = base_url.rstrip("/")
        self._engines = engines
        self._timeout = timeout
        self._user_agent = user_agent
        self._healthy = True

    @property
    def name(self) -> str:
        return "searxng"

    def probe(self) -> bool:
        try:
            get_json(
                f"{self._base}/config",
                headers={"User-Agent": self._user_agent},
                timeout=min(self._timeout, 5.0),
            )
        except UpstreamError as exc:
            log.debug("searxng probe failed: %s", exc)
            self._healthy = False
            return False
        self._healthy = True
        return True

    def search(self, query: str, limit: int, *, lang: str) -> list[Result]:
        params = {
            "q": query,
            "format": "json",
            "pageno": "1",
            "safesearch": "0",
        }
        if lang:
            params["language"] = lang
        if self._engines:
            params["engines"] = self._engines
        try:
            payload = get_json(
                f"{self._base}/search",
                params=params,
                headers={
                    "Accept": "application/json",
                    "User-Agent": self._user_agent,
                },
                timeout=self._timeout,
            )
        except UpstreamError:
            self._healthy = False
            raise
        self._healthy = True
        # SearXNG returns a whole page of aggregated hits and has no count parameter.
        return _results(payload, "content", limit)


def build_backend(settings) -> Backend:
    if settings.backend == "searxng":
        return SearxngBackend(
            base_url=settings.searxng_url,
            engines=settings.searxng_engines,
            timeout=settings.request_timeout,
            user_agent=settings.user_agent,
        )
    return BraveBackend(
        api_key=settings.brave_api_key,
        url=settings.brave_url,
        timeout=settings.request_timeout,
        user_agent=settings.user_agent,
    )


def _results(container: Any, excerpt_key: str, limit: int) -> list[Result]:
    """Pull the result list out of a provider payload, skipping anything malformed.

    A provider that renames a field or returns a null entry should cost us that one result, not
    the whole query: a partial answer is still usable and an exception here is not.
    """
    items = container.get("results") if isinstance(container, dict) else None
    if not isinstance(items, list):
        return []

    results: list[Result] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip()
        if not url:
            continue
        results.append(
            Result(
                title=strip_markup(item.get("title") or "") or url,
                url=url,
                excerpt=strip_markup(item.get(excerpt_key) or ""),
            )
        )
        if len(results) >= limit:
            break
    return results
