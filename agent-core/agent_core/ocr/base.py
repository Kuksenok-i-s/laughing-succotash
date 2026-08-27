"""Remote handwriting OCR contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

ProgressHook = Callable[[float, str], None]


@dataclass(slots=True)
class OcrResult:
    raw_text: str
    markdown: str
    model: str | None = None
    elapsed_seconds: float | None = None
    passes: int = 3
    kind: str = "text"  # "text" | "other"
    description: str = ""

    @property
    def empty(self) -> bool:
        if self.kind == "other":
            return not self.description.strip()
        return not (self.markdown or self.raw_text).strip()


class OcrError(RuntimeError):
    """OCR failed. The job fails; the Core stays up and the temp file is removed."""


class HandwritingOCR(Protocol):
    async def recognize(
        self,
        image_path: Path,
        *,
        content_type: str | None = None,
        on_progress: ProgressHook | None = None,
    ) -> OcrResult: ...

    async def warmup(self) -> None: ...

    async def close(self) -> None: ...

    @property
    def ready(self) -> bool: ...

    @property
    def model_name(self) -> str: ...
