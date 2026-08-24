"""Speech-to-text abstraction and backends."""

from .base import (
    STT_CPU_FALLBACK,
    NoticeHook,
    ProgressHook,
    SpeechToText,
    SttError,
    TranscriptionResult,
    TranscriptSegment,
)

__all__ = [
    "STT_CPU_FALLBACK",
    "NoticeHook",
    "ProgressHook",
    "SpeechToText",
    "SttError",
    "TranscriptSegment",
    "TranscriptionResult",
]
