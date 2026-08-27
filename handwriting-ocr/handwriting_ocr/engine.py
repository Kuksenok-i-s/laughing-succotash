"""Qwen3-VL via Ollama: triage, then one or three passes.

Pass 1 (always): decide whether the image is mainly readable text.
- ``kind=other`` → stop after this pass (description only).
- ``kind=text`` → pass 2 corrects the raw draft against the image, pass 3 structures Markdown.

Triage uses the original colours (a cat is not a shopping list). OCR passes 2–3 use the
grayscale+contrast preprocess. The Ollama client stays standard-library-only besides Pillow
preprocess.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from .preprocess import prepare_image_bytes

log = logging.getLogger(__name__)

ProgressHook = Callable[[float, str], None]

_TRIAGE_JSON = re.compile(r"\{.*\}", re.DOTALL)

TRIAGE_PROMPT = """\
Look at this image and decide what it is for a personal-assistant bot.

If the image is mainly handwritten or printed text the user would want transcribed \
(notes, lists, whiteboards, screenshots of text, receipts, labels with readable text), \
reply with ONLY this JSON (no markdown fences, no commentary):
{"kind":"text","raw_text":"<literal transcription, line breaks preserved, uncertain bits as [?…?]>"}

If it is NOT mainly text to transcribe (photo of a person/place/object, meme, diagram without \
prose, blank/blurry, decorative image), reply with ONLY:
{"kind":"other","description":"<short factual description of what is visible>"}

Rules:
- Do not invent text that is not visible.
- Prefer "other" when text is incidental (logo on a shirt, street sign in a landscape).
- Output JSON only.
"""

PASS2_PROMPT = """\
You are correcting a handwriting/print OCR draft against the original image.

You receive the image and a raw transcription. Fix misread words, missed lines and wrong order.
Keep uncertain fragments as [?…?]. Plain text only, no Markdown. Output only the corrected \
transcription.

RAW TRANSCRIPTION:
"""

PASS3_PROMPT = """\
You are structuring a corrected OCR transcription into clean Markdown.

You receive the image and the corrected plain text. Restore headings, lists and tables when \
clearly present. Keep [?…?] markers. Do not invent content. Output only the Markdown document.

CORRECTED TRANSCRIPTION:
"""


class Engine(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def model_name(self) -> str: ...

    @property
    def ollama_reachable(self) -> bool: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def recognize(
        self,
        image_path: Path,
        *,
        on_progress: ProgressHook | None = None,
    ) -> dict[str, Any]: ...


class OllamaEngine:
    def __init__(
        self,
        *,
        ollama_url: str,
        model: str,
        keep_alive: str = "10m",
        request_timeout: float = 600.0,
    ) -> None:
        self._base = ollama_url.rstrip("/")
        self._model = model
        self._keep_alive = keep_alive
        self._timeout = request_timeout
        self._ready = False
        self._ollama_ok = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def ollama_reachable(self) -> bool:
        return self._ollama_ok

    def probe(self) -> bool:
        """Cheap reachability check used by /health. Does not load weights."""
        try:
            self._request("GET", "/api/tags")
            self._ollama_ok = True
            return True
        except Exception as exc:
            self._ollama_ok = False
            log.debug("ollama probe failed: %s", exc)
            return False

    def load(self) -> None:
        started = time.monotonic()
        self.probe()
        if not self._ollama_ok:
            raise RuntimeError(f"ollama is unreachable at {self._base}")
        self._chat(messages=[], keep_alive=self._keep_alive)
        self._ready = True
        log.info(
            "ollama model %s loaded in %.1fs (keep_alive=%s)",
            self._model,
            time.monotonic() - started,
            self._keep_alive,
        )

    def unload(self) -> None:
        try:
            self._chat(messages=[], keep_alive=0)
        finally:
            self._ready = False
        log.info("ollama model %s unloaded", self._model)

    def recognize(
        self,
        image_path: Path,
        *,
        on_progress: ProgressHook | None = None,
    ) -> dict[str, Any]:
        if not self._ready:
            self.load()

        # Colour for triage; grayscale+contrast only if we continue into OCR.
        original_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        started = time.monotonic()

        if on_progress is not None:
            on_progress(5.0, "recognizing")
        triage_raw = self._chat(
            messages=[
                {
                    "role": "user",
                    "content": TRIAGE_PROMPT,
                    "images": [original_b64],
                }
            ],
            keep_alive=self._keep_alive,
        )
        triage = parse_triage(triage_raw)

        if triage["kind"] == "other":
            if on_progress is not None:
                on_progress(100.0, "completed")
            description = triage["description"]
            elapsed = round(time.monotonic() - started, 2)
            return {
                "kind": "other",
                "raw_text": "",
                "markdown": "",
                "description": description,
                "model": self._model,
                "elapsed_seconds": elapsed,
                "passes": 1,
            }

        raw_text = triage["raw_text"]
        ocr_bytes, _mime = prepare_image_bytes(image_path)
        ocr_b64 = base64.b64encode(ocr_bytes).decode("ascii")

        if on_progress is not None:
            on_progress(35.0, "recognizing")
        raw_text = self._chat(
            messages=[
                {
                    "role": "user",
                    "content": PASS2_PROMPT + raw_text,
                    "images": [ocr_b64],
                }
            ],
            keep_alive=self._keep_alive,
        ).strip() or raw_text

        if on_progress is not None:
            on_progress(70.0, "structuring")
        markdown = self._chat(
            messages=[
                {
                    "role": "user",
                    "content": PASS3_PROMPT + raw_text,
                    "images": [ocr_b64],
                }
            ],
            keep_alive=self._keep_alive,
        ).strip() or raw_text

        if on_progress is not None:
            on_progress(100.0, "completed")

        elapsed = round(time.monotonic() - started, 2)
        return {
            "kind": "text",
            "raw_text": raw_text,
            "markdown": markdown,
            "description": "",
            "model": self._model,
            "elapsed_seconds": elapsed,
            "passes": 3,
        }

    def _chat(
        self,
        *,
        messages: list[dict[str, Any]],
        keep_alive: str | int,
    ) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
        }
        body = self._request("POST", "/api/chat", payload)
        message = body.get("message") or {}
        return (message.get("content") or "").strip()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"ollama {method} {path} -> {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            self._ollama_ok = False
            raise RuntimeError(f"ollama {method} {path} failed: {exc.reason}") from exc

        self._ollama_ok = True
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))


def parse_triage(text: str) -> dict[str, str]:
    """Parse the triage JSON; fall back to treating free text as a transcription."""
    blob = (text or "").strip()
    if not blob:
        return {"kind": "other", "description": "empty image or no visible content"}

    candidate = blob
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", blob, re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1)
    else:
        match = _TRIAGE_JSON.search(blob)
        if match:
            candidate = match.group(0)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        log.warning("triage JSON parse failed; treating reply as text transcription")
        return {"kind": "text", "raw_text": blob}

    kind = str(data.get("kind") or "").strip().lower()
    if kind in {"other", "image", "scene", "photo", "non_text", "non-text"}:
        description = str(data.get("description") or data.get("caption") or "").strip()
        if not description:
            description = "image without readable note text"
        return {"kind": "other", "description": description}

    raw_text = str(data.get("raw_text") or data.get("text") or "").strip()
    if not raw_text and kind == "text":
        # Model said text but forgot the field — keep going with empty and let later passes try.
        return {"kind": "text", "raw_text": ""}
    if kind == "text" or raw_text:
        return {"kind": "text", "raw_text": raw_text or blob}

    # Unknown shape: if there is any descriptive field, treat as other.
    description = str(data.get("description") or "").strip()
    if description and not raw_text:
        return {"kind": "other", "description": description}
    return {"kind": "text", "raw_text": blob}
