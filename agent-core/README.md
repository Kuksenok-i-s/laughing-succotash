# Agent Core

The source of truth of the personal assistant. Runs on an Intel Mac mini, where Cursor, Whisper and
the user's files are.

It listens on nothing public. It dials out to the Gateway over WSS and keeps that connection alive,
which is why the Mac needs no port forwarding, no VPN and no reverse tunnel.

## Responsibilities

Everything that is not Telegram transport: conversation sessions, transcription, hierarchical
analysis of recordings, the MCP tool surface (reminders, calendar, tasks, notes, memory, contacts,
timers, system status), the permission model and confirmation flow, the scheduler, and all durable
state.

## Layout

```
agent_core/
├── main.py            composition, startup order, ordered shutdown
├── config.py          settings plus the filesystem/project allowlists
├── agent/             Cursor: acp_client.py is the wire, cursor_acp.py the backend
├── assistant/
│   ├── service.py     inbound request → job → reply
│   ├── sessions.py    conversation ↔ Cursor session, per-session MCP tokens
│   ├── prompts.py     the trusted/untrusted boundary, in text
│   ├── transcript.py  chunking and hierarchical extraction
│   └── confirmations.py
├── mcp/               the loopback HTTP MCP server, tools, permission tiers
├── stt/               local faster-whisper or the GPU service, behind one protocol
├── audio/             streamed uploads, ffprobe
├── search/            the external-search contract (no provider configured)
├── jobs/              per-conversation serial execution
├── scheduler/         reminders, timers, expiry, cleanup
├── reminders/         buttoned follow-up after a fire (done / snooze / reschedule)
├── calendar/          CalendarProvider protocol + local SQLite implementation
├── rpc/               the outbound link and the Gateway-facing handlers
└── storage/           SQLite schema, migrations, repositories
```

## Running

```bash
pip install -e ../packages/pa-protocol -r requirements.txt
pip install -r requirements-stt.txt     # only where Whisper actually runs
cp .env.example .env                    # CORE_TOKEN, MCP_TOKEN, ALLOWED_USERS
cp assistant.toml.example ~/.personal-assistant/assistant.toml
python -m agent_core.main
pytest
```

Installation under launchd is covered in [`../docs/operations.md`](../docs/operations.md).

## Notes worth knowing

**The Gateway is authenticated as a service, not as a person.** A `user_id` in a request is a claim.
Every handler in `rpc/handlers.py` re-checks it against `ALLOWED_USERS`, so a compromised Gateway
can talk to the Core but cannot act as a user the Core does not know.

**The permission gate lives in the MCP server, not in the ACP callback.** ACP identifies a tool only
through a display title with its arguments rendered as a Markdown code fence; deciding
READ/SAFE_WRITE/DANGEROUS by parsing that would be fragile and security-critical. The authoritative
decision is made in-process with exact tool names and validated Pydantic arguments. See
[`../docs/cursor-acp.md`](../docs/cursor-acp.md).

**The chat session runs in a sandbox directory.** Cursor's built-in file writes and shell commands
do *not* trigger a permission request — this was verified, not assumed — so the conversation session
is rooted in a throwaway workspace, and coding work happens only in allowlisted project paths, in
`plan` mode when the project is not writable.

**Provenance decides whether a write may run unattended.** A reminder the user asked for is created
immediately; the same reminder inferred from a meeting recording asks first. Provenance is set by
the Core from where the turn came from and is never read from anything the model says.

**Work is a job, never a held-open RPC.** Whisper and Cursor take minutes; `assistant.submit` returns
an id in milliseconds. Progress is advisory notifications, results are durable events. Jobs are
serial per conversation and concurrent across users, so one user's hour-long transcription never
blocks another's question — and control commands like `/cancel` run on their own lane so they are
not queued behind the job they are meant to cancel.

**Search is not enabled.** `search/base.py` defines the contract — structured results, one URL per
fetch, no private or loopback addresses — but no provider is wired up, so `web_search` and
`web_fetch` are not registered and the assistant has no network reach through MCP. The guard is
written and tested ahead of the provider because the policy is the part that is easy to get wrong,
and loopback on this machine includes the MCP server itself.

**Reminders do not depend on Cursor.** The scheduler reads SQLite and emits events; a reminder fires
whether or not Cursor is running and whether or not the Gateway is reachable. If the Gateway is away,
the event waits in the outbound log and is delivered exactly once after the handshake.
