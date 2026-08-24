# 2. JSON-RPC 2.0 over a Core-initiated WebSocket

Status: accepted

## Context

The Mac mini sits behind a residential NAT with no stable public address. Both sides need to
originate calls: the Gateway forwards user input, and the Core independently pushes reminder
notifications that are triggered by its scheduler rather than by any user request. Work items last
from a second to over an hour.

## Decision

One persistent WebSocket Secure connection, dialled **outbound** by the Core to the Gateway,
carrying bidirectional JSON-RPC 2.0. Authentication is a bearer service token at the HTTP upgrade.
Binary WebSocket frames on the same connection carry audio (ADR: see `docs/protocol.md` §6).

Rejected alternatives:

- **REST + polling** — either wasteful or laggy, and gives the Core no way to push a reminder.
- **REST + callback URL** — requires an inbound listener on the Mac mini, which is the thing we
  are specifically avoiding.
- **WireGuard / reverse SSH / mTLS** — solves a problem we do not have. WSS plus a long random
  token is sufficient, and every extra layer is another thing that can silently break and leave
  the assistant unreachable.
- **A message broker** — Redis or RabbitMQ for exactly two peers is infrastructure for its own
  sake.

## Consequences

No inbound exposure on the Mac mini. The Core owns reconnection with exponential backoff and full
jitter, since it is the only side that can re-dial.

Because a long operation cannot hold an RPC request open, everything slow becomes a job:
`assistant.submit` returns `{job_id, accepted}` in milliseconds and progress arrives as separate
notifications. A dropped connection then loses no work.

The connection is the only channel, so its failure modes must be handled explicitly rather than
retried blindly — hence sequence numbers, replay of unacknowledged durable events, and idempotency
keys on every state-changing method.
