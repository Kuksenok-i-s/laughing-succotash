"""Agent Core schema.

Migrations are ordered, idempotent statements applied at startup and recorded in
``schema_migrations``. This is the source of truth for the whole system; the Gateway's database
holds transport state only and can be deleted without losing anything here.
"""

from __future__ import annotations

MIGRATIONS: list[tuple[str, str]] = [
    (
        "0001_core",
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id       TEXT PRIMARY KEY,          -- namespaced, e.g. 'tg:123456789'
            timezone      TEXT,                      -- per-user override of DEFAULT_TIMEZONE
            display_name  TEXT,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            status          TEXT NOT NULL DEFAULT 'active',  -- active | archived
            title           TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_conversations_user
            ON conversations(user_id, status, updated_at DESC);

        CREATE TABLE IF NOT EXISTS cursor_sessions (
            session_id      TEXT PRIMARY KEY,        -- our id
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            backend         TEXT NOT NULL,           -- acp | cli
            external_id     TEXT,                    -- Cursor's sessionId
            workspace       TEXT NOT NULL,
            mode            TEXT NOT NULL DEFAULT 'agent',
            status          TEXT NOT NULL DEFAULT 'active',
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cursor_sessions_conversation
            ON cursor_sessions(conversation_id, status);

        CREATE TABLE IF NOT EXISTS jobs (
            job_id       TEXT PRIMARY KEY,
            request_id   TEXT NOT NULL UNIQUE,       -- idempotency: one job per Telegram interaction
            user_id      TEXT NOT NULL,
            chat_id      INTEGER,
            message_id   INTEGER,
            kind         TEXT NOT NULL,              -- text | command | audio | transcribe_only
            status       TEXT NOT NULL,              -- queued|running|completed|failed|cancelled
            stage        TEXT,
            payload      TEXT NOT NULL DEFAULT '{}',
            result       TEXT,
            error_code   TEXT,
            error_detail TEXT,
            created_at   TEXT NOT NULL,
            started_at   TEXT,
            finished_at  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
        CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS uploads (
            upload_id     TEXT PRIMARY KEY,
            request_id    TEXT NOT NULL,
            user_id       TEXT NOT NULL,
            chat_id       INTEGER,
            message_id    INTEGER,
            filename      TEXT NOT NULL,
            content_type  TEXT,
            declared_size INTEGER NOT NULL,
            received_size INTEGER NOT NULL DEFAULT 0,
            duration_seconds REAL,
            purpose       TEXT NOT NULL DEFAULT 'assistant',
            temp_path     TEXT NOT NULL,
            status        TEXT NOT NULL,             -- open|committed|aborted|expired
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_uploads_request ON uploads(request_id);
        CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status, updated_at);

        -- Durable Core -> Gateway events. Persisted before being written to the socket so a
        -- reminder that fires while the Gateway is down is still delivered after reconnect.
        CREATE TABLE IF NOT EXISTS outbound_events (
            seq          INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id     TEXT NOT NULL UNIQUE,
            delivery_id  TEXT,
            method       TEXT NOT NULL,
            params       TEXT NOT NULL,
            user_id      TEXT,
            status       TEXT NOT NULL DEFAULT 'pending',  -- pending|sent|failed|dropped
            attempts     INTEGER NOT NULL DEFAULT 0,
            last_error   TEXT,
            created_at   TEXT NOT NULL,
            sent_at      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_outbound_pending ON outbound_events(status, seq);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_outbound_delivery
            ON outbound_events(delivery_id) WHERE delivery_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS delivery_state (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS schema_migrations (
            name       TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        );
        """,
    ),
    (
        "0002_assistant_objects",
        """
        CREATE TABLE IF NOT EXISTS reminders (
            reminder_id  TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL,
            text         TEXT NOT NULL,
            due_at       TEXT,                       -- UTC ISO-8601; null for pure recurrence
            timezone     TEXT NOT NULL,
            rrule        TEXT,                       -- RFC 5545 RRULE for recurring reminders
            status       TEXT NOT NULL DEFAULT 'scheduled',  -- scheduled|fired|cancelled|completed
            last_fired_at TEXT,
            fire_count   INTEGER NOT NULL DEFAULT 0,
            operation_id TEXT UNIQUE,                -- idempotency for the creating tool call
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(status, due_at);
        CREATE INDEX IF NOT EXISTS idx_reminders_user ON reminders(user_id, status, due_at);

        CREATE TABLE IF NOT EXISTS timers (
            timer_id     TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL,
            label        TEXT,
            duration_seconds INTEGER NOT NULL,
            fires_at     TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'running',  -- running|fired|cancelled
            operation_id TEXT UNIQUE,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_timers_fires ON timers(status, fires_at);

        CREATE TABLE IF NOT EXISTS tasks (
            task_id      TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL,
            title        TEXT NOT NULL,
            details      TEXT,
            status       TEXT NOT NULL DEFAULT 'open',  -- open|done|cancelled
            priority     TEXT NOT NULL DEFAULT 'normal',
            due_at       TEXT,
            owner        TEXT,
            tags         TEXT NOT NULL DEFAULT '[]',
            source       TEXT NOT NULL DEFAULT 'user',  -- user|transcript|agent
            operation_id TEXT UNIQUE,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id, status, due_at);

        CREATE TABLE IF NOT EXISTS notes (
            note_id      TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL,
            title        TEXT,
            body         TEXT NOT NULL,
            tags         TEXT NOT NULL DEFAULT '[]',
            source       TEXT NOT NULL DEFAULT 'user',
            operation_id TEXT UNIQUE,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_notes_user ON notes(user_id, updated_at DESC);
        CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
            title, body, note_id UNINDEXED, tokenize='unicode61'
        );

        -- Long-term memory, deliberately separate from notes: written only on an explicit
        -- instruction or a confirmed proposal (see ADR 7).
        CREATE TABLE IF NOT EXISTS memory (
            memory_id    TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL,
            content      TEXT NOT NULL,
            category     TEXT,
            confidence   REAL NOT NULL DEFAULT 1.0,
            source       TEXT NOT NULL DEFAULT 'explicit',  -- explicit|confirmed
            operation_id TEXT UNIQUE,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            last_used_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_memory_user ON memory(user_id, updated_at DESC);
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            content, category, memory_id UNINDEXED, tokenize='unicode61'
        );

        CREATE TABLE IF NOT EXISTS contacts (
            contact_id   TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL,
            display_name TEXT NOT NULL,
            aliases      TEXT NOT NULL DEFAULT '[]',
            emails       TEXT NOT NULL DEFAULT '[]',
            phones       TEXT NOT NULL DEFAULT '[]',
            note         TEXT,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_contacts_user ON contacts(user_id, display_name);

        CREATE TABLE IF NOT EXISTS calendar_events (
            event_id     TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL,
            calendar     TEXT NOT NULL DEFAULT 'default',
            title        TEXT NOT NULL,
            starts_at    TEXT NOT NULL,
            ends_at      TEXT NOT NULL,
            timezone     TEXT NOT NULL,
            location     TEXT,
            description  TEXT,
            attendees    TEXT NOT NULL DEFAULT '[]',
            external_id  TEXT,
            status       TEXT NOT NULL DEFAULT 'confirmed',
            operation_id TEXT UNIQUE,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_calendar_range
            ON calendar_events(user_id, status, starts_at, ends_at);

        -- A proposed action waiting on the user. Persisted so a confirmation survives a restart.
        CREATE TABLE IF NOT EXISTS pending_actions (
            action_id    TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL,
            chat_id      INTEGER,
            job_id       TEXT,
            tool_name    TEXT NOT NULL,
            arguments    TEXT NOT NULL,              -- exact validated args, executed verbatim
            operation_id TEXT NOT NULL,
            tier         TEXT NOT NULL,              -- safe_write | dangerous
            prompt_text  TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending',  -- pending|approved|rejected|expired
            resolved_at  TEXT,
            expires_at   TEXT NOT NULL,
            created_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_pending_actions_status
            ON pending_actions(status, expires_at);

        -- Records that a side-effecting tool call already ran, so a retry after a lost response
        -- returns the original result instead of acting twice.
        CREATE TABLE IF NOT EXISTS operations (
            operation_id TEXT PRIMARY KEY,
            tool_name    TEXT NOT NULL,
            user_id      TEXT NOT NULL,
            result       TEXT NOT NULL,
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS transcription_metadata (
            transcription_id TEXT PRIMARY KEY,
            job_id       TEXT,
            user_id      TEXT NOT NULL,
            filename     TEXT,
            language     TEXT,
            duration     REAL,
            segment_count INTEGER,
            char_count   INTEGER,
            model        TEXT,
            elapsed_seconds REAL,
            created_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_transcription_user
            ON transcription_metadata(user_id, created_at DESC);
        """,
    ),
    (
        # A reminder that fires hours after the last message still has to reach a chat, and the
        # scheduler has no request to read one from. The last chat the user wrote from is the only
        # honest answer the Core can give.
        "0003_user_last_chat",
        """
        ALTER TABLE users ADD COLUMN last_chat_id INTEGER;
        """,
    ),
    (
        "0004_journal",
        """
        CREATE TABLE IF NOT EXISTS journal_entries (
            entry_id     TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL,
            local_date   TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'open',
            step         TEXT NOT NULL DEFAULT 'offer',
            answers      TEXT NOT NULL DEFAULT '{}',
            prompted_at  TEXT,
            completed_at TEXT,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            UNIQUE(user_id, local_date)
        );
        CREATE INDEX IF NOT EXISTS idx_journal_user_date
            ON journal_entries(user_id, local_date);

        CREATE TABLE IF NOT EXISTS journal_summaries (
            summary_id   TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL,
            period       TEXT NOT NULL,
            body         TEXT NOT NULL DEFAULT '',
            entry_count  INTEGER NOT NULL DEFAULT 0,
            skipped_count INTEGER NOT NULL DEFAULT 0,
            status       TEXT NOT NULL DEFAULT 'pending',
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            UNIQUE(user_id, period)
        );
        CREATE INDEX IF NOT EXISTS idx_journal_summaries_user
            ON journal_summaries(user_id, period);
        """,
    ),
    (
        "0005_upload_caption",
        """
        ALTER TABLE uploads ADD COLUMN caption TEXT;
        """,
    ),
    (
        "0006_upload_attribution",
        """
        ALTER TABLE uploads ADD COLUMN attribution TEXT;
        """,
    ),
    (
        "0007_upload_album",
        """
        ALTER TABLE uploads ADD COLUMN album_id TEXT;
        ALTER TABLE uploads ADD COLUMN part_index INTEGER;
        ALTER TABLE uploads ADD COLUMN part_count INTEGER;
        CREATE INDEX IF NOT EXISTS idx_uploads_album
            ON uploads(album_id, part_index) WHERE album_id IS NOT NULL;
        """,
    ),
    (
        "0008_contact_operation_id",
        """
        ALTER TABLE contacts ADD COLUMN operation_id TEXT;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_contacts_operation
            ON contacts(operation_id) WHERE operation_id IS NOT NULL;
        """,
    ),
    (
        # Trainer journal: per-Telegram-user long-term store for athletes, programmes,
        # scheduled sessions and structured workout logs. Distinct from the evening diary.
        "0009_training",
        """
        CREATE TABLE IF NOT EXISTS training_profiles (
            user_id     TEXT PRIMARY KEY,
            enabled     INTEGER NOT NULL DEFAULT 1,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS training_athletes (
            athlete_id    TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL,
            display_name  TEXT NOT NULL,
            aliases       TEXT NOT NULL DEFAULT '[]',
            note          TEXT,
            is_self       INTEGER NOT NULL DEFAULT 0,
            status        TEXT NOT NULL DEFAULT 'active',
            operation_id  TEXT,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_training_athletes_user
            ON training_athletes(user_id, status, display_name);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_training_athletes_operation
            ON training_athletes(operation_id) WHERE operation_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS training_programs (
            program_id    TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL,
            athlete_id    TEXT NOT NULL,
            title         TEXT NOT NULL,
            goal          TEXT,
            days_per_week INTEGER,
            weekly_plan   TEXT NOT NULL DEFAULT '[]',
            notes         TEXT,
            status        TEXT NOT NULL DEFAULT 'active',
            started_on    TEXT,
            ended_on      TEXT,
            operation_id  TEXT,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_training_programs_athlete
            ON training_programs(user_id, athlete_id, status);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_training_programs_operation
            ON training_programs(operation_id) WHERE operation_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS training_sessions (
            session_id    TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL,
            athlete_id    TEXT NOT NULL,
            program_id    TEXT,
            local_date    TEXT NOT NULL,
            title         TEXT NOT NULL,
            plan          TEXT NOT NULL DEFAULT '[]',
            notes         TEXT,
            status        TEXT NOT NULL DEFAULT 'planned',
            operation_id  TEXT,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_training_sessions_athlete_date
            ON training_sessions(user_id, athlete_id, local_date);
        CREATE INDEX IF NOT EXISTS idx_training_sessions_user_date
            ON training_sessions(user_id, local_date);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_training_sessions_operation
            ON training_sessions(operation_id) WHERE operation_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS training_logs (
            log_id        TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL,
            athlete_id    TEXT NOT NULL,
            session_id    TEXT,
            local_date    TEXT NOT NULL,
            title         TEXT,
            raw_text      TEXT,
            exercises     TEXT NOT NULL DEFAULT '[]',
            notes         TEXT,
            duration_minutes INTEGER,
            operation_id  TEXT,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_training_logs_athlete_date
            ON training_logs(user_id, athlete_id, local_date DESC);
        CREATE INDEX IF NOT EXISTS idx_training_logs_user_date
            ON training_logs(user_id, local_date DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_training_logs_operation
            ON training_logs(operation_id) WHERE operation_id IS NOT NULL;
        """,
    ),
    (
        "0010_training_mode",
        """
        ALTER TABLE training_profiles ADD COLUMN mode TEXT NOT NULL DEFAULT 'self';
        UPDATE training_profiles SET mode = 'trainer'
        WHERE user_id IN (
            SELECT user_id FROM training_athletes WHERE is_self = 0
        );
        """,
    ),
    (
        "0011_training_progress",
        """
        ALTER TABLE training_programs ADD COLUMN total_sessions INTEGER;
        ALTER TABLE training_programs ADD COLUMN weeks INTEGER;
        """,
    ),
]
