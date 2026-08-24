# 1. Split the system into Telegram Gateway and Agent Core

Status: accepted

## Context

Telegram must be reachable from a network position that is stable and unblocked, which in practice
means a VPS. The AI runtime must be where Cursor is authenticated and where the user's files,
projects, calendar and contacts live, which is a personal Mac mini. Those are different machines
with very different exposure: the VPS accepts traffic from the public internet, the Mac mini holds
the user's entire digital life.

## Decision

Two independently deployable applications.

**Telegram Gateway** is a transport adapter and nothing else. It knows only
`telegram_user_id`, `telegram_chat_id`, `telegram_message_id`, `request_id`, `delivery_id`,
opaque payloads and delivery status. It has no Cursor, no Whisper, no MCP tools, no memory, no
notes, no tasks, no calendar logic, no reasoning, and it is not the scheduler's source of truth.

**Agent Core** is the source of truth for everything else.

## Consequences

A compromise of the exposed component yields the attacker a message relay, not the assistant. The
Gateway holds no Cursor credentials, no memory database, no shell and no filesystem access. Losing
the Gateway's SQLite loses in-flight transport state only; reminders, tasks, notes and memory are
untouched.

The cost is a real network boundary in the middle of every interaction, which forces the async job
model (ADR 2) and idempotency work that a single process would not need. This is accepted: the
alternative is putting a public listener in front of the machine that has the user's files on it.

Because the Core is authoritative, it re-checks user authorization itself rather than trusting a
`user_id` sent by an authenticated Gateway.
