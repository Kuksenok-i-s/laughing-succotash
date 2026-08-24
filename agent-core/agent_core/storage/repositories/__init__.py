"""Repository aggregate.

A single object carrying every repository keeps constructor signatures short and gives services
one thing to depend on instead of eight.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..database import Database
from .actions import CalendarRepository, OperationLedger, PendingAction, PendingActionRepository
from .assistant import (
    ContactRepository,
    MemoryRepository,
    NoteRepository,
    Reminder,
    ReminderRepository,
    TaskRepository,
    TimerRepository,
)
from .conversations import Conversation, ConversationRepository, CursorSession, User
from .events import OutboundEvent, OutboundEventRepository
from .jobs import Job, JobRepository, TranscriptionMetadataRepository, Upload, UploadRepository

__all__ = [
    "CalendarRepository",
    "ContactRepository",
    "Conversation",
    "ConversationRepository",
    "CursorSession",
    "Job",
    "JobRepository",
    "MemoryRepository",
    "NoteRepository",
    "OperationLedger",
    "OutboundEvent",
    "OutboundEventRepository",
    "PendingAction",
    "PendingActionRepository",
    "Reminder",
    "ReminderRepository",
    "Repositories",
    "TaskRepository",
    "TimerRepository",
    "TranscriptionMetadataRepository",
    "Upload",
    "UploadRepository",
    "User",
]


@dataclass(slots=True)
class Repositories:
    conversations: ConversationRepository
    jobs: JobRepository
    uploads: UploadRepository
    transcriptions: TranscriptionMetadataRepository
    events: OutboundEventRepository
    reminders: ReminderRepository
    timers: TimerRepository
    tasks: TaskRepository
    notes: NoteRepository
    memory: MemoryRepository
    contacts: ContactRepository
    calendar: CalendarRepository
    pending_actions: PendingActionRepository
    operations: OperationLedger

    @classmethod
    def build(cls, db: Database, default_timezone: str) -> "Repositories":
        return cls(
            conversations=ConversationRepository(db, default_timezone),
            jobs=JobRepository(db),
            uploads=UploadRepository(db),
            transcriptions=TranscriptionMetadataRepository(db),
            events=OutboundEventRepository(db),
            reminders=ReminderRepository(db),
            timers=TimerRepository(db),
            tasks=TaskRepository(db),
            notes=NoteRepository(db),
            memory=MemoryRepository(db),
            contacts=ContactRepository(db),
            calendar=CalendarRepository(db),
            pending_actions=PendingActionRepository(db),
            operations=OperationLedger(db),
        )
