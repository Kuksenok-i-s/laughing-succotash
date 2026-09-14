"""The search provider, over the web-search service's HTTP API.

Mirrors ``ocr.remote_service.RemoteOcrService``: one bearer token, one aiohttp session, errors
translated into the domain exception. There is no local fallback — a search that cannot be made is
reported to the agent as a failed tool call, which is a better outcome than a confident answer built
on nothing.

``search_many`` is the reason the service exists. ``search`` satisfies the ``SearchProvider``
protocol for the MCP tool, where the agent asks one question at a time because that is how tool
calls work; ``search_many`` is for the Core's own passes — factchecking a transcript has a list of
claims that are independent, and sending them together turns a sequence of round trips into one.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp

from .base import SearchError, guard_url

log = logging.getLogger(__name__)


class RemoteSearchProvider:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        default_limit: int = 5,
        lang: str = "ru",
        request_timeout: float = 20.0,
        fetch_timeout: float = 40.0,
        max_concurrent: int = 4,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._default_limit = default_limit
        self._lang = lang
        self._request_timeout = request_timeout
        self._fetch_timeout = fetch_timeout
        self._slots = asyncio.Semaphore(max(1, max_concurrent))
        self._session: aiohttp.ClientSession | None = None
        self._ready = False
        self._backend = "unknown"

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def backend(self) -> str:
        return self._backend

    async def warmup(self) -> None:
        try:
            health = await self._request("GET", "/health", authorized=False)
        except SearchError as exc:
            raise SearchError(f"search service unreachable: {exc}") from exc
        self._ready = True
        self._backend = health.get("backend") or self._backend
        log.info(
            "search service ready at %s (backend=%s reachable=%s fetch=%s)",
            self._base,
            health.get("backend"),
            health.get("backend_reachable"),
            health.get("fetch_enabled"),
        )

    async def close(self) -> None:
        self._ready = False
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """One query's results. Satisfies ``SearchProvider`` for the ``web_search`` tool."""
        answers = await self.search_many([query], limit=limit)
        answer = answers[0]
        if answer.get("error"):
            raise SearchError(f"search failed: {answer['error']}")
        return answer.get("results") or []

    async def search_many(
        self, queries: list[str], *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Every query in one round trip, answered in the order given.

        A query that failed upstream keeps its own ``error`` and an empty result list; the caller
        decides whether a partial answer is usable, because for a factcheck it usually is.
        """
        wanted = [query for query in queries if query and query.strip()]
        if not wanted:
            return []

        started = time.monotonic()
        async with self._slots:
            payload = await self._request(
                "POST",
                "/v1/search",
                json={
                    "queries": wanted,
                    "limit": limit or self._default_limit,
                    "lang": self._lang,
                },
            )

        answers = payload.get("queries")
        if not isinstance(answers, list) or len(answers) != len(wanted):
            raise SearchError("the search service returned a malformed batch")

        log.info(
            "searched %d quer%s in %.2fs (backend=%s, %s cached)",
            len(wanted),
            "y" if len(wanted) == 1 else "ies",
            time.monotonic() - started,
            payload.get("backend"),
            payload.get("cached"),
        )
        return [
            {
                "query": answer.get("query") or query,
                "results": _results_of(answer),
                "error": answer.get("error"),
            }
            for query, answer in zip(wanted, answers, strict=True)
        ]

    async def fetch(self, url: str) -> dict[str, Any]:
        """Readable text for one page.

        Guarded here as well as in the service: this side knows which loopback belongs to the Core,
        and a URL reaching this method may have arrived inside an untrusted document.
        """
        async with self._slots:
            payload = await self._request(
                "POST",
                "/v1/fetch",
                json={"url": guard_url(url)},
                timeout=self._fetch_timeout,
            )
        return {
            "url": payload.get("url") or url,
            "title": payload.get("title") or "",
            "text": payload.get("text") or "",
            "truncated": bool(payload.get("truncated")),
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
        authorized: bool = True,
    ) -> dict[str, Any]:
        session = await self._get_session()
        headers = {"Authorization": f"Bearer {self._token}"} if authorized else {}
        try:
            async with session.request(
                method,
                self._base + path,
                json=json,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout or self._request_timeout),
            ) as response:
                body = await response.json(content_type=None)
                if response.status >= 400:
                    if response.status >= 500:
                        self._ready = False
                    raise SearchError(
                        f"{method} {path} -> {response.status} "
                        f"{(body or {}).get('message') or (body or {}).get('code') or ''}".strip()
                    )
                return body or {}
        except SearchError:
            raise
        except (TimeoutError, aiohttp.ClientError) as exc:
            self._ready = False
            detail = str(exc).strip() or type(exc).__name__
            raise SearchError(f"search service {method} {path} failed: {detail}") from exc


def _results_of(answer: dict[str, Any]) -> list[dict[str, str]]:
    """Keep the three agreed fields and drop the rest.

    The service already reduces provider payloads to this shape. Doing it again here means a
    service that grows a field does not silently start feeding it to the model.
    """
    raw = answer.get("results")
    if not isinstance(raw, list):
        return []
    results = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        results.append(
            {
                "title": str(item.get("title") or url),
                "url": url,
                "excerpt": str(item.get("excerpt") or ""),
            }
        )
    return results
