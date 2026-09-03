"""Qwen-VL: triage, then one or three passes.

Pass 1 (always): decide whether the image is mainly readable text.
- ``kind=other`` → stop after this pass (description only).
- ``kind=text`` → pass 2 corrects the raw draft against the image, pass 3 structures Markdown.

Triage uses the original colours (a cat is not a shopping list). OCR passes 2–3 use the
grayscale+contrast preprocess. The HTTP client stays standard-library-only besides Pillow
preprocess. Two transports: Ollama ``/api/chat`` and llama.cpp ``/v1/chat/completions``.
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

from .preprocess import (
    DEFAULT_MAX_EDGE,
    prepare_image_bytes,
    prepare_ocr_variants,
    prepare_triage_bytes,
)

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

# Dedicated OCR VLMs (OvisOCR2, GLM-OCR) are trained on this shape, not on triage JSON.
OCR_PROMPT = (
    "Extract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. Format formulas as LaTeX. "
    "Format tables as HTML: <table>...</table>. Transcribe all other text as standard "
    "Markdown. Preserve the original text without translation or paraphrasing."
)


class Engine(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def model_name(self) -> str: ...

    @property
    def backend_reachable(self) -> bool: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def recognize(
        self,
        image_path: Path,
        *,
        on_progress: ProgressHook | None = None,
    ) -> dict[str, Any]: ...


def strip_repeats(text: str, *, min_repeat_times: int = 5) -> str:
    """Drop a trailing loop (``00\\n00\\n…``) so a 2048-token cap does not keep the junk."""
    blob = (text or "").rstrip()
    if not blob:
        return blob

    lines = blob.splitlines()
    while len(lines) >= min_repeat_times:
        last = lines[-1].strip()
        if not last or len(last) > 16:
            break
        count = 0
        for line in reversed(lines):
            if line.strip() == last:
                count += 1
            else:
                break
        if count < min_repeat_times:
            break
        lines = lines[:-count]
        log.info("stripped %d trailing repeated lines %r", count, last[:16])

    blob = "\n".join(lines).rstrip()
    n = len(blob)
    if n < 80:
        return blob
    max_period = min(80, n // min_repeat_times)
    for unit_len in range(1, max_period + 1):
        unit = blob[-unit_len:]
        times = 0
        pos = n
        while pos >= unit_len and blob[pos - unit_len : pos] == unit:
            times += 1
            pos -= unit_len
        if times >= min_repeat_times and times * unit_len >= 20:
            trimmed = blob[: pos + unit_len].rstrip()
            log.info("stripped trailing %d-char unit repeated %d times", unit_len, times)
            return trimmed
    return blob


def score_draft(text: str) -> tuple[int, int, int]:
    """Prefer real page text over a leaked merge prompt or an empty stub."""
    blob = (text or "").strip()
    leak = "DRAFT COLOR" in blob or "DRAFT GRAY" in blob or "DRAFT INK" in blob
    letters = sum(ch.isalpha() for ch in blob)
    digits = sum(ch.isdigit() for ch in blob)
    return (0 if leak else 1, letters + digits, len(blob))


def pick_best_draft(drafts: dict[str, str]) -> str:
    nonempty = {name: text for name, text in drafts.items() if text.strip()}
    if not nonempty:
        return ""
    name, text = max(nonempty.items(), key=lambda item: score_draft(item[1]))
    color = nonempty.get("color")
    if color and name != "color":
        winner_alnum = score_draft(text)[1]
        color_alnum = score_draft(color)[1]
        # Colour is the natural page. Switch only if another view recovered much more text.
        if winner_alnum < color_alnum * 1.2:
            name, text = "color", color
    log.info("chose ocr draft %s (%d chars)", name, len(text))
    return text


def merge_ocr_drafts(drafts: dict[str, str]) -> str:
    return pick_best_draft(drafts)


def _ocr_one(
    image_path: Path,
    chat: Callable[[str, str, str], str],
    image_max_edge: int,
) -> str:
    page_bytes, page_mime = prepare_triage_bytes(image_path, max_edge=image_max_edge)
    page_b64 = base64.b64encode(page_bytes).decode("ascii")
    return strip_repeats(chat(OCR_PROMPT, page_b64, page_mime)).strip()


def _ocr_ensemble(
    image_path: Path,
    chat: Callable[[str, str, str], str],
    image_max_edge: int,
    on_progress: ProgressHook | None,
) -> tuple[str, int]:
    variants = prepare_ocr_variants(image_path, max_edge=image_max_edge)
    drafts: dict[str, str] = {}
    marks = (10.0, 40.0, 70.0)
    for index, (name, payload, mime) in enumerate(variants):
        if on_progress is not None:
            on_progress(marks[index], "recognizing")
        b64 = base64.b64encode(payload).decode("ascii")
        drafts[name] = strip_repeats(chat(OCR_PROMPT, b64, mime)).strip()
        log.info("ocr variant %s produced %d chars", name, len(drafts[name]))
    if on_progress is not None:
        on_progress(90.0, "structuring")
    return merge_ocr_drafts(drafts), 3


def run_recognition(
    *,
    model: str,
    image_path: Path,
    chat: Callable[[str, str, str], str],
    on_progress: ProgressHook | None = None,
    image_max_edge: int = DEFAULT_MAX_EDGE,
    max_passes: int = 3,
    pipeline: str = "triage",
) -> dict[str, Any]:
    """Shared 1-or-3-pass pipeline. ``chat(prompt, image_b64, mime)`` is the only backend hook."""
    started = time.monotonic()
    if pipeline == "ocr":
        if max_passes <= 1:
            if on_progress is not None:
                on_progress(5.0, "recognizing")
            text = _ocr_one(image_path, chat, image_max_edge)
            passes = 1
        else:
            text, passes = _ocr_ensemble(image_path, chat, image_max_edge, on_progress)
        if on_progress is not None:
            on_progress(100.0, "completed")
        if not text:
            return {
                "kind": "other",
                "raw_text": "",
                "markdown": "",
                "description": "empty image or no visible content",
                "model": model,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "passes": passes,
            }
        return {
            "kind": "text",
            "raw_text": text,
            "markdown": text,
            "description": "",
            "model": model,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "passes": passes,
        }

    triage_bytes, triage_mime = prepare_triage_bytes(image_path, max_edge=image_max_edge)
    triage_b64 = base64.b64encode(triage_bytes).decode("ascii")

    if on_progress is not None:
        on_progress(5.0, "recognizing")
    triage = parse_triage(chat(TRIAGE_PROMPT, triage_b64, triage_mime))

    if triage["kind"] == "other":
        if on_progress is not None:
            on_progress(100.0, "completed")
        return {
            "kind": "other",
            "raw_text": "",
            "markdown": "",
            "description": triage["description"],
            "model": model,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "passes": 1,
        }

    raw_text = strip_repeats(triage["raw_text"])
    if max_passes <= 1:
        if on_progress is not None:
            on_progress(100.0, "completed")
        return {
            "kind": "text",
            "raw_text": raw_text,
            "markdown": raw_text,
            "description": "",
            "model": model,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "passes": 1,
        }

    ocr_bytes, ocr_mime = prepare_image_bytes(image_path, max_edge=image_max_edge)
    ocr_b64 = base64.b64encode(ocr_bytes).decode("ascii")

    if on_progress is not None:
        on_progress(35.0, "recognizing")
    raw_text = strip_repeats(chat(PASS2_PROMPT + raw_text, ocr_b64, ocr_mime)).strip() or raw_text

    if on_progress is not None:
        on_progress(70.0, "structuring")
    markdown = strip_repeats(chat(PASS3_PROMPT + raw_text, ocr_b64, ocr_mime)).strip() or raw_text

    if on_progress is not None:
        on_progress(100.0, "completed")
    return {
        "kind": "text",
        "raw_text": raw_text,
        "markdown": markdown,
        "description": "",
        "model": model,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "passes": 3,
    }


class OllamaEngine:
    def __init__(
        self,
        *,
        ollama_url: str,
        model: str,
        keep_alive: str = "10m",
        request_timeout: float = 600.0,
        image_max_edge: int = DEFAULT_MAX_EDGE,
        max_passes: int = 3,
        pipeline: str = "triage",
    ) -> None:
        self._base = ollama_url.rstrip("/")
        self._model = model
        self._keep_alive = keep_alive
        self._timeout = request_timeout
        self._image_max_edge = image_max_edge
        self._max_passes = max_passes
        self._pipeline = pipeline
        self._ready = False
        self._backend_ok = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def backend_reachable(self) -> bool:
        return self._backend_ok

    def probe(self) -> bool:
        """Cheap reachability check used by /health. Does not load weights."""
        try:
            self._request("GET", "/api/tags")
            self._backend_ok = True
            return True
        except Exception as exc:
            self._backend_ok = False
            log.debug("ollama probe failed: %s", exc)
            return False

    def load(self) -> None:
        started = time.monotonic()
        self.probe()
        if not self._backend_ok:
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
        return run_recognition(
            model=self._model,
            image_path=image_path,
            chat=self._vision_chat,
            on_progress=on_progress,
            image_max_edge=self._image_max_edge,
            max_passes=self._max_passes,
            pipeline=self._pipeline,
        )

    def _vision_chat(self, prompt: str, image_b64: str, _mime: str) -> str:
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if image_b64:
            message["images"] = [image_b64]
        return self._chat(messages=[message], keep_alive=self._keep_alive)

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
            self._backend_ok = False
            raise RuntimeError(f"ollama {method} {path} failed: {exc.reason}") from exc

        self._backend_ok = True
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))


class LlamaCppEngine:
    """Qwen-VL via llama-server's OpenAI-compatible ``/v1/chat/completions``."""

    def __init__(
        self,
        *,
        llama_url: str,
        model: str,
        request_timeout: float = 600.0,
        max_tokens: int = 2048,
        image_max_edge: int = DEFAULT_MAX_EDGE,
        max_passes: int = 3,
        pipeline: str = "triage",
    ) -> None:
        self._base = llama_url.rstrip("/")
        self._model = model
        self._timeout = request_timeout
        self._max_tokens = max_tokens
        self._image_max_edge = image_max_edge
        self._max_passes = max_passes
        self._pipeline = pipeline
        self._ready = False
        self._backend_ok = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def backend_reachable(self) -> bool:
        return self._backend_ok

    def probe(self) -> bool:
        """Cheap reachability check used by /health. Does not load weights."""
        try:
            self._request("GET", "/v1/models")
            self._backend_ok = True
            return True
        except Exception as exc:
            self._backend_ok = False
            log.debug("llama-server probe failed: %s", exc)
            return False

    def load(self) -> None:
        started = time.monotonic()
        self.probe()
        if not self._backend_ok:
            raise RuntimeError(f"llama-server is unreachable at {self._base}")
        self._ready = True
        log.info(
            "llama-server %s reachable in %.1fs",
            self._model,
            time.monotonic() - started,
        )

    def unload(self) -> None:
        # llama-server owns the weights; we only drop the ready flag so /health is honest
        # after an explicit unload. The next recognize() probes again.
        self._ready = False
        log.info("llama-server %s marked unloaded (process keeps the weights)", self._model)

    def recognize(
        self,
        image_path: Path,
        *,
        on_progress: ProgressHook | None = None,
    ) -> dict[str, Any]:
        if not self._ready:
            self.load()
        return run_recognition(
            model=self._model,
            image_path=image_path,
            chat=self._vision_chat,
            on_progress=on_progress,
            image_max_edge=self._image_max_edge,
            max_passes=self._max_passes,
            pipeline=self._pipeline,
        )

    def _vision_chat(self, prompt: str, image_b64: str, mime: str) -> str:
        text_part = {"type": "text", "text": prompt}
        if image_b64:
            image_part = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{image_b64}"},
            }
            content = (
                [image_part, text_part]
                if self._pipeline == "ocr"
                else [text_part, image_part]
            )
        else:
            content = [text_part]
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "temperature": 0.0 if self._pipeline == "ocr" else 0.1,
            "max_tokens": self._max_tokens,
            "repeat_penalty": 1.15,
        }
        body = self._request("POST", "/v1/chat/completions", payload)
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict)
            ]
            return "".join(parts).strip()
        return ""

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
            raise RuntimeError(f"llama-server {method} {path} -> {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            self._backend_ok = False
            raise RuntimeError(f"llama-server {method} {path} failed: {exc.reason}") from exc

        self._backend_ok = True
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
