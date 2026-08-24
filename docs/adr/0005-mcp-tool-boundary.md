# 5. Capability-based MCP server, hosted over loopback HTTP

Status: accepted

## Context

Cursor needs to create reminders, read the calendar, manage tasks and notes, search memory and
look up contacts. It must not acquire a general-purpose remote-execution primitive in the process.

## Decision

The Core exposes one MCP server, `personal-assistant-mcp`, offering **named capabilities with
typed arguments**. There is deliberately no `shell(command)` tool and no unrestricted
`http_request`. Web access is limited to `web_search` and `web_fetch` returning structured,
size-capped results.

Transport is **HTTP on loopback**, passed to `session/new` as
`{"name","type":"http","url","headers":[...]}` — the exact shape the installed CLI accepts, which
requires both `type` and a present `headers` array (see `docs/cursor-acp.md`). A bearer token in
`headers` means only Cursor can reach the endpoint.

stdio MCP also works and was verified, but was rejected: a stdio server is a separate process, and
every tool call would then need its own IPC channel back to the Core merely to reach the SQLite
repositories, the scheduler and — crucially — the confirmation service that has to ask the user a
question and wait for the answer. Over loopback HTTP the handlers run in-process and can simply
await that answer.

Filesystem reach is an explicit allowlist of directories and named projects. `$HOME` is never
indexed wholesale.

## Consequences

Every capability is enumerable, which is what makes the three-tier permission model of ADR 7
expressible at all: a permission engine can classify `calendar_delete`, but it cannot meaningfully
classify `shell("...")`.

Arguments arrive validated by Pydantic, so the permission decision is made against exact,
well-typed values rather than a parsed string.

Adding a capability means adding a tool with a schema and a permission tier — the extension path
is uniform, which is what "extensible MCP architecture" has to mean in practice.

The loopback endpoint is a listening socket on the Mac mini. It binds to `127.0.0.1` only and
requires the bearer token, so it is not reachable off-host.
