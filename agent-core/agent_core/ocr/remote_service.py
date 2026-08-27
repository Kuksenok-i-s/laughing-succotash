"""Handwriting OCR on the GPU host, over its HTTP API.

Mirrors ``stt.gpu_service.GpuServiceSTT``: upload, poll, collect, delete. There is no local
fallback — if the OCR host is down the job fails and the user is told.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import aiohttp
from pa_protocol import new_ulid

from .base import HandwritingOCR, OcrError, OcrResult, ProgressHook

log = logging.getLogger(__name__)


class RemoteOcrService(HandwritingOCR):
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        poll_interval: float = 2.0,
        request_timeout: float = 30.0,
        upload_timeout: float = 300.0,
        stall_timeout: float = 900.0,
        max_concurrent: int = 1,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._poll_interval = poll_interval
        self._request_timeout = request_timeout
        self._upload_timeout = upload_timeout
        self._stall_timeout = stall_timeout
        self._slots = asyncio.Semaphore(max(1, max_concurrent))
        self._session: aiohttp.ClientSession | None = None
        self._ready = False
        self._model = "remote-ocr"

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def model_name(self) -> str:
        return self._model

    async def warmup(self) -> None:
        try:
            health = await self._request("GET", "/health", authorized=False)
        except Exception as exc:
            raise OcrError(f"OCR service unreachable: {exc}") from exc
        self._ready = True
        self._model = health.get("model") or self._model
        log.info(
            "OCR service ready at %s (model=%s loaded=%s queued=%s)",
            self._base,
            health.get("model"),
            health.get("model_loaded"),
            health.get("queued"),
        )

    async def close(self) -> None:
        self._ready = False
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def recognize(
        self,
        image_path: Path,
        *,
        content_type: str | None = None,
        on_progress: ProgressHook | None = None,
    ) -> OcrResult:
        if not image_path.exists():
            raise OcrError(f"image file not found: {image_path.name}")
        if image_path.stat().st_size == 0:
            raise OcrError("image file is empty")

        if not self._ready:
            await self.warmup()

        async with self._slots:
            started = time.monotonic()
            job_id = new_ulid()
            try:
                await self._submit(job_id, image_path, content_type=content_type)
                await self._await_completion(job_id, on_progress)
                payload = await self._request("GET", f"/v1/jobs/{job_id}/result")
            finally:
                await self._forget(job_id)

        result = _to_result(payload)
        log.info(
            "recognized %s on the OCR service in %.1fs (kind=%s passes=%d raw=%d markdown=%d model=%s)",
            image_path.name,
            time.monotonic() - started,
            result.kind,
            result.passes,
            len(result.raw_text),
            len(result.markdown),
            result.model,
        )
        return result

    async def _submit(
        self, job_id: str, image_path: Path, *, content_type: str | None
    ) -> None:
        query = {
            "filename": image_path.name,
            "content_type": content_type or "application/octet-stream",
        }
        with image_path.open("rb") as handle:
            await self._request(
                "PUT",
                f"/v1/jobs/{job_id}",
                params=query,
                data=handle,
                timeout=self._upload_timeout,
                expect100=True,
            )

    async def _await_completion(self, job_id: str, on_progress: ProgressHook | None) -> None:
        last_percent = -1.0
        last_change = time.monotonic()
        while True:
            status = await self._request("GET", f"/v1/jobs/{job_id}")
            state = status.get("status")
            percent = float(status.get("percent") or 0.0)
            stage = status.get("stage") or "recognizing"

            if percent != last_percent:
                last_percent = percent
                last_change = time.monotonic()
                if on_progress is not None:
                    on_progress(min(max(percent / 100.0, 0.0), 1.0), stage)

            if state == "done":
                return
            if state == "failed":
                raise OcrError(f"OCR failed: {status.get('error') or 'unknown'}")
            if time.monotonic() - last_change > self._stall_timeout:
                raise OcrError(
                    f"OCR stalled at {percent:.0f}% for {self._stall_timeout:.0f}s"
                )
            await asyncio.sleep(self._poll_interval)

    async def _forget(self, job_id: str) -> None:
        try:
            await self._request("DELETE", f"/v1/jobs/{job_id}")
        except Exception as exc:
            log.debug("could not delete OCR job %s: %s", job_id, exc)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        data: Any = None,
        timeout: float | None = None,
        authorized: bool = True,
        expect100: bool = False,
    ) -> dict[str, Any]:
        session = await self._get_session()
        headers = {"Authorization": f"Bearer {self._token}"} if authorized else {}
        try:
            async with session.request(
                method,
                self._base + path,
                params=params,
                data=data,
                headers=headers,
                expect100=expect100,
                timeout=aiohttp.ClientTimeout(total=timeout or self._request_timeout),
            ) as response:
                body = await response.json(content_type=None)
                if response.status >= 400:
                    if response.status >= 500:
                        self._ready = False
                    raise OcrError(
                        f"{method} {path} -> {response.status} "
                        f"{(body or {}).get('message') or (body or {}).get('code') or ''}".strip()
                    )
                return body or {}
        except OcrError:
            raise
        except (TimeoutError, aiohttp.ClientError) as exc:
            self._ready = False
            raise OcrError(f"OCR service {method} {path} failed: {exc}") from exc


def _to_result(payload: dict[str, Any]) -> OcrResult:
    kind = (payload.get("kind") or "text").strip().lower()
    if kind not in {"text", "other"}:
        kind = "text"
    return OcrResult(
        raw_text=(payload.get("raw_text") or "").strip(),
        markdown=(payload.get("markdown") or "").strip(),
        model=payload.get("model"),
        elapsed_seconds=payload.get("elapsed_seconds"),
        passes=int(payload.get("passes") or (1 if kind == "other" else 3)),
        kind=kind,
        description=(payload.get("description") or "").strip(),
    )
