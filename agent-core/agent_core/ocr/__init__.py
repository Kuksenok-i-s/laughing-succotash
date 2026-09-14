"""Remote handwriting OCR client."""

from .base import HandwritingOCR, OcrError, OcrResult
from .remote_service import RemoteOcrService

__all__ = ["HandwritingOCR", "OcrError", "OcrResult", "RemoteOcrService"]
