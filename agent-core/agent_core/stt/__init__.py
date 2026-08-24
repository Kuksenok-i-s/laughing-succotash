"""Speech-to-text abstraction and backends."""

from .base import SpeechToText, SttError, TranscriptionResult, TranscriptSegment

__all__ = ["SpeechToText", "SttError", "TranscriptSegment", "TranscriptionResult"]
