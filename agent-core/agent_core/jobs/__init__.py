"""Job scheduling: one queue per conversation, cancellation, shutdown."""

from .manager import JobManager

__all__ = ["JobManager"]
