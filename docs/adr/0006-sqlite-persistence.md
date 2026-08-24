# 6. SQLite on both sides, with distinct roles

Status: accepted

## Context

This is a single-user personal assistant running on one Mac mini and one small VPS. It needs
durable reminders, tasks, notes, memory, job state and delivery queues. It does not need
horizontal scale.

## Decision

SQLite on both machines, in WAL mode with foreign keys enforced. No PostgreSQL, Redis, RabbitMQ,
Kafka, Celery, Kubernetes or vector database.

The two databases have deliberately unequal status:

- **Core** — the source of truth: users, conversations, cursor sessions, jobs, uploads, reminders,
  tasks, notes, memory, pending actions, outbound events, delivery state, transcription metadata.
- **Gateway** — transport state only: pending requests, pending uploads, outbound delivery,
  processed event IDs, RPC sequence state.

Deleting the Gateway's database must never lose a task, note, memory, reminder or calendar entry.
This is a schema-level invariant, not a convention: there is nowhere in the Gateway schema to
store such a thing.

Migrations are ordered, idempotent SQL applied at startup and tracked in a `schema_migrations`
table.

## Consequences

Backup is copying a file. Failure modes are few and well understood, and there is no broker to be
down at 3am when a reminder must fire.

Writes are serialized per database. For one user this is irrelevant; concurrency comes from
per-conversation queues in the application layer, not from the storage engine.

WAL mode allows the scheduler to read while a job writes. All SQLite access happens off the event
loop so a slow write cannot stall the WebSocket.

Vector search for memory is deliberately absent. Keyword and recency search over a personal-scale
memory table is adequate, and an embedding store would add a model, an index and a rebuild path to
maintain.
