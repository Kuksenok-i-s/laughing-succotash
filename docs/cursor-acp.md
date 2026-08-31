# Cursor Agent ACP — verified capabilities

Everything in this document was obtained by running the actual binary and recording the
traffic. Nothing here is recalled from memory or inferred from upstream ACP docs. Where the
installed Cursor build deviates from the generic Agent Client Protocol specification, the
observed behaviour wins and the deviation is called out explicitly.

## Environment probed

| Item | Value |
| --- | --- |
| `cursor-agent --version` | `2026.08.11-e8db854` |
| Install path | `~/.local/share/cursor-agent/versions/2026.08.11-e8db854/` |
| Auth | `cursor-agent status` → logged in (`cursor_login` method) |
| Probe host | Linux x86_64 (dev machine, not the target Mac mini) |
| Probe date | 2026-08-23 |

The probe scripts live in `agent-core/tools/acp_probe/` so the findings can be re-verified on the
Mac mini before deployment. Re-run them there: the CLI is the same JS bundle on both platforms,
but the ACP surface is version-gated and must be re-checked after any CLI upgrade.

## Discovery: ACP is a hidden subcommand

`cursor-agent --help` does **not** list ACP, and `--acp` is rejected with
`error: unknown option '--acp'`. ACP is exposed as an undocumented subcommand:

```
$ cursor-agent acp --help
Usage: agent acp [options]

Start the Cursor Agent as an ACP (Agent Client Protocol) server
```

It speaks **newline-delimited JSON-RPC 2.0 over stdin/stdout**. There is no `Content-Length`
framing (unlike LSP). One JSON object per line. Diagnostics go to stderr.

Because the subcommand is undocumented, treat its presence as a runtime capability to be probed,
not as a guarantee. `CursorACPBackend` checks for it at startup and falls back to
`CursorCLIBackend` if `cursor-agent acp` is missing or fails its handshake.

## `initialize`

Request:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
  "protocolVersion":1,
  "clientCapabilities":{"fs":{"readTextFile":true,"writeTextFile":true},"terminal":false}}}
```

Verbatim response from the installed build:

```json
{"protocolVersion":1,
 "agentCapabilities":{
   "loadSession":true,
   "mcpCapabilities":{"http":true,"sse":true},
   "promptCapabilities":{"audio":false,"embeddedContext":false,"image":true},
   "sessionCapabilities":{"list":{}}},
 "authMethods":[{"id":"cursor_login","name":"Cursor Login",
   "description":"Authenticate using existing Cursor login credentials. Run 'agent login' first if not logged in."}]}
```

Consequences for this project:

- `promptCapabilities.audio == false`. Audio **cannot** be handed to Cursor as a prompt content
  block. Whisper must run first and only text reaches Cursor. This independently confirms the
  STT-then-Cursor pipeline; it is not merely a design preference.
- `promptCapabilities.embeddedContext == false`. There is no resource-block content type, so long
  meeting transcripts must be embedded in the text prompt itself. This is what forces the
  hierarchical chunking in `assistant/transcript.py` — we cannot attach a transcript as a
  separate untrusted resource and must instead delimit it in-band.
- `loadSession: true`. Sessions survive a CLI restart and can be replayed (see `session/load`).
- No `authenticate` call is needed while `cursor-agent status` reports a logged-in user.

## `session/new`

```json
{"method":"session/new","params":{"cwd":"<abs path>","mcpServers":[...]}}
```

`cwd` must be absolute. Returns:

- `sessionId` — a UUID string.
- `modes` — `currentModeId` plus `agent` / `plan` / `ask`.
- `models` — `currentModelId` (`default[]` = Auto) and ~35 available model IDs.
- `configOptions` — the same mode/model choices in generic form.

## `session/prompt`

```json
{"method":"session/prompt","params":{"sessionId":"...","prompt":[{"type":"text","text":"..."}]}}
```

Returns only `{"stopReason": "end_turn" | "cancelled"}`. **The assistant's reply text is not in
the response.** It arrives beforehand as a stream of `session/update` notifications, and the
client must accumulate it. Observed `sessionUpdate` variants:

| `sessionUpdate` | Meaning |
| --- | --- |
| `session_info_update` | Auto-generated chat title |
| `available_commands_update` | Slash commands / skills list (large; ignored by this project) |
| `agent_thought_chunk` | Reasoning tokens — **never** shown to the Telegram user |
| `agent_message_chunk` | The actual reply, streamed in small pieces |
| `user_message_chunk` | Replayed user turns (only during `session/load`) |
| `tool_call` | Tool started: `toolCallId`, `title`, `kind`, `status` |
| `tool_call_update` | Status transition to `in_progress` / `completed`, plus `rawOutput` |
| `current_mode_update` | Emitted after `session/set_mode` |

`tool_call.kind` observed: `read`, `edit`, `execute`, `search`, `other`. These drive the
`job.progress` stage reporting sent to the Gateway.

Chunks are small (2–4 characters is common), so the reply must be buffered and flushed on a
timer rather than forwarded to Telegram per chunk.

## `session/load` — resume works

`session/load` with a previously returned `sessionId` succeeds and **replays the entire
conversation** as `user_message_chunk` / `agent_message_chunk` / `tool_call` notifications before
returning. Conversational context genuinely persists: after a restart the agent still answered a
follow-up question that depended only on the earlier turn.

The replay is a real cost. A long conversation replays in full on every load, so
`CursorACPBackend` loads a session lazily — only when a prompt actually arrives for a session it
has not seen since process start — and it suppresses replayed chunks by ignoring all updates
until the `session/load` response is received.

## `session/list`

Returns `{"sessions":[{"sessionId","cwd","title","updatedAt"}]}`. Useful for reconciling the
Core's SQLite session table against reality after a crash, and for garbage-collecting sessions.

## `session/cancel` — notification, not a request

Sent as a JSON-RPC **notification** (no `id`). The in-flight `session/prompt` then returns
`{"stopReason":"cancelled"}`. Verified that the session remains fully usable afterwards: the very
next `session/prompt` on the same session returned `end_turn` normally. This is what makes
`/cancel` safe to expose.

## `session/set_mode` — and it is genuinely enforced

`{"method":"session/set_mode","params":{"sessionId":"...","modeId":"plan"}}` returns `{}` and
emits a `current_mode_update`.

This was tested rather than assumed. In `plan` mode the agent was directly instructed to write a
file and run a shell command. It did **neither** — no file was created on disk, no `execute` tool
call was emitted, and it produced a plan document instead. `plan` mode is therefore a real
read-only enforcement boundary and is used for projects marked `writable: false`.

## Permissions: the critical asymmetry

This is the single most important finding, and it is the opposite of what the ACP spec would
suggest.

**MCP tool calls always request permission.** Every call into a connected MCP server triggers a
client-bound `session/request_permission` request:

```json
{"jsonrpc":"2.0","id":0,"method":"session/request_permission","params":{
  "sessionId":"...",
  "toolCall":{"toolCallId":"tool_...","title":"probe-probe_get_magic_word: probe_get_magic_word",
    "kind":"other","status":"pending",
    "content":[{"type":"content","content":{"type":"text","text":"```json\n{}\n```"}}]},
  "options":[{"optionId":"allow-once","name":"Allow once","kind":"allow_once"},
             {"optionId":"allow-always","name":"Allow always","kind":"allow_always"},
             {"optionId":"reject-once","name":"Reject","kind":"reject_once"}]}}
```

Answering `{"outcome":{"outcome":"selected","optionId":"reject-once"}}` cleanly denies the call:
the tool reports `rawOutput: {"rejected": true}` and the agent does not retry.

**Built-in file writes and shell commands do NOT request permission.** In `agent` mode the agent
created a file and ran `echo` with zero `session/request_permission` traffic. The file was really
written to disk.

Two consequences drive the whole security design:

1. The permission model cannot be enforced through ACP for Cursor's *built-in* tools. The
   assistant conversation session therefore runs with `cwd` set to a dedicated throwaway sandbox
   directory containing nothing sensitive, **and** `session/set_mode` to `plan` after every
   `session/new` / successful `session/load` (see `assistant/sessions.py`). `plan` refuses
   built-in write/shell but still allows MCP (verified by `tools.acp_probe plan-mcp`). Coding
   sessions are confined to explicitly allowlisted project paths (`plan` when not writable).
2. Permission enforcement for *assistant capabilities* is reliable, because every one of those
   capabilities is an MCP tool. See below for why the gate is nonetheless placed inside the MCP
   server rather than in the ACP permission callback.

### Why the authoritative gate lives in the MCP server

The `session/request_permission` payload identifies the tool only through
`toolCall.title` (`"<server>-<tool>: <tool>"`) and renders its arguments as a fenced JSON code
block inside `content[].content.text`. `rawInput` was `{}` for MCP calls. Deciding
READ / SAFE_WRITE / DANGEROUS by string-parsing a Markdown code fence out of a title-cased
display field would be fragile and security-critical.

So permissions are two-layered:

- **Layer 1 — ACP callback** (`agent/cursor_acp.py`): coarse and non-authoritative. Allows calls
  into our own MCP server (Layer 2 will gate them properly) and rejects unknown MCP servers.
- **Layer 2 — inside the MCP server** (`mcp/permissions.py`): authoritative. It runs in-process
  with exact tool names and already-validated Pydantic arguments, and it is the thing that raises
  a Telegram confirmation and blocks on the answer.

## MCP transport: HTTP works, and the exact shape matters

Both stdio and HTTP were verified end-to-end (`tools/list` and `tools/call` both reached the test
server and the agent used the returned value).

Despite `mcpCapabilities` advertising only `{"http":true,"sse":true}`, **stdio MCP servers work**
and are accepted as `{"name","command","args","env"}`.

For HTTP the accepted object shape is stricter than the ACP spec suggests, and getting it wrong
produces an opaque error. Both of these were **rejected**:

```json
{"name":"x","url":"http://127.0.0.1:8931/mcp"}
{"name":"x","type":"http","url":"http://127.0.0.1:8931/mcp"}
```

with `-32603 Internal error` and a Zod `invalid_union` detail. The working shape requires **both**
`type` and an explicitly present `headers` array:

```json
{"name":"assistant","type":"http","url":"http://127.0.0.1:8931/mcp","headers":[]}
```

`headers` is mandatory even when empty. This project sends a bearer token through it so that only
Cursor can reach the loopback MCP endpoint.

Observed client behaviour: Cursor sends `Accept: application/json, text/event-stream`, honours the
`Mcp-Session-Id` response header, negotiates `protocolVersion 2024-11-05`, and accepts a plain
`application/json` response body (an SSE stream is not required). A `202` with an empty body is
the correct reply to `notifications/initialized`.

**HTTP is the transport this project uses**, because it lets the MCP tool handlers run in the same
process as the Core's SQLite repositories, scheduler and confirmation service. A stdio MCP server
would be a separate process needing its own IPC channel back to the Core just to ask the user a
confirmation question.

## Errors

Standard JSON-RPC codes. Unknown method:

```json
{"code":-32601,"message":"\"Method not found\": nonexistent/method","data":{"method":"..."}}
```

Malformed params surface as `-32603 Internal error` with a Zod issue array in `data` — verbose,
but it does identify the offending field path.

## Capability summary

| Capability | Status | Where relied upon |
| --- | --- | --- |
| Session creation | works | `agent/cursor_acp.py` |
| Session resume across restart | works (`session/load`, replays history) | `assistant/sessions.py` |
| Streaming | works (`session/update`, needs buffering) | `agent/cursor_acp.py` |
| Cancellation | works, session stays usable | `/cancel` command |
| Tool calls visible to client | works | `job.progress` stages |
| Permission prompts — MCP tools | works | Layer 1 gate |
| Permission prompts — built-in write/shell | **absent** | forces sandbox cwd + plan mode |
| Read-only enforcement | works via `plan` mode | chat sessions + non-writable projects |
| MCP tools usable in `plan` mode | works (`plan-mcp` probe) | Telegram chat uses plan |
| Workspaces | per-session `cwd` | project allowlist |
| MCP over stdio | works | not used |
| MCP over HTTP | works (`type` + `headers` required) | `mcp/server.py` |
| Audio prompt input | **unsupported** | forces local Whisper |
| Embedded context resources | **unsupported** | forces in-band transcripts |

## Re-verification

`agent-core/tools/acp_probe/` reproduces the claims above. Run it after any Cursor CLI upgrade, and
on the Mac mini before deploying:

```bash
cd agent-core && python -m tools.acp_probe --all
python -m tools.acp_probe --list          # what each probe checks
python -m tools.acp_probe initialize      # or run one
```

| Probe | Checks |
| --- | --- |
| `initialize` | The capability flags: no audio input, no embedded context, `loadSession`, MCP over HTTP |
| `session` | `session/new`, streaming, and that reply text arrives only as updates |
| `resume` | `session/load` across two processes, and that context survived |
| `cancel` | The turn stops and the session is still usable afterwards |
| `plan-mode` | `plan` mode really refuses to write to disk |
| `plan-mcp` | In `plan` mode MCP `tools/call` still reaches a local HTTP server; execute does not fire |
| `permissions` | MCP calls ask; built-in writes do not |
| `mcp-http` | The strict `type` + `headers` shape, and that the loose forms are still rejected |

It exits non-zero if a previously verified capability has changed, and names the code path that
depended on it — the probe is deliberately standalone rather than built on `agent_core.agent`, so it
cannot pass because both sides share a wrong assumption.

Last full run: `2026-08-23`, `cursor-agent 2026.08.11-e8db854` — all probes matched this document.
