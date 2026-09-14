# 3. Cursor Agent is the only AI runtime

Status: accepted

## Context

The assistant needs intent understanding, context, reasoning, tool selection, document analysis,
transcript summarization and genuine codebase work. Cursor Agent already does all of this and is
already authenticated on the target machine. Adding a second LLM layer would mean two places that
decide what to do.

## Decision

Cursor Agent is the sole reasoning engine. No DeepSeek, OpenAI or Anthropic API is used for the
main chat, and no orchestration framework is layered on top of it. The Core supplies capabilities
through MCP and otherwise stays out of the way.

Integration is over ACP (`cursor-agent acp`), whose real behaviour is documented in
`docs/cursor-acp.md` — probed, not assumed.

The rest of the system talks to an `AgentBackend` protocol, never to ACP directly.
`CursorACPBackend` is the implementation; `CursorCLIBackend` (`cursor-agent --print
--output-format stream-json`) is a fallback for a CLI build without a working `acp` subcommand.
The fallback changes only the transport — sessions, permissions and job semantics are unchanged.

## Consequences

Conversation state lives inside Cursor sessions, verified to survive a CLI restart via
`session/load`. The Core stores the session ID and the mapping to a namespaced user identity, not
the transcript.

Two probed facts constrain the design rather than being worked around:

- ACP reports `promptCapabilities.audio == false`, so audio can never be handed to Cursor
  directly. Local STT is a requirement, not a preference (ADR 4).
- ACP reports `embeddedContext == false`, so a transcript cannot be attached as a separate
  untrusted resource and must be delimited in-band in the prompt text.

ACP is an undocumented subcommand of a CLI that auto-updates, so its capabilities are re-probed at
startup and the probe suite is re-run after upgrades.
