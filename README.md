# Personal assistant

A personal AI assistant with a Telegram interface, split across two machines.

```
        Telegram
           │
  ┌────────▼─────────┐        JSON-RPC 2.0        ┌──────────────────────┐
  │ Telegram Gateway │◄───  over persistent  ────►│      Agent Core      │
  │   (Linux VPS)    │           WSS              │ (Jetson Xavier)      │
  │                  │   (the Core dials out)     │                      │
  │ aiogram          │                            │ Cursor Agent (ACP)   │
  │ delivery queue   │                            │ Whisper large-v3     │
  │ temporary audio  │                            │ MCP tools, scheduler │
  │ transport SQLite │                            │ memory, SQLite       │
  └──────────────────┘                            └──────────────────────┘
```

The Gateway is a transport adapter and nothing more. It knows Telegram ids, request ids and
delivery status; it does not know what a session, a reminder or a tool is. The Core is the source
of truth for everything else, and it is the only machine that holds Cursor credentials, the
assistant's data, or any reasoning.

The split exists because Telegram needs a host with reliable access to it, while Cursor and Whisper
need to run where the files and the CPU are. Losing the Gateway's disk costs nothing but a few
in-flight messages.

## What it does

Text and voice conversation; transcription of long recordings with extraction of decisions, tasks,
owners and deadlines; reminders and timers that fire from the Core's own scheduler; calendar,
tasks, notes, contacts and controlled long-term memory as MCP tools; and Cursor acting as a coding
agent over an allowlist of local projects.

Nothing inferred from a recording, a file or a web page is ever executed without the user saying
yes. See [ADR 0007](docs/adr/0007-tool-permission-model.md).

## Layout

| Path | What it is |
| --- | --- |
| `telegram-gateway/` | Deploy unit A — the Telegram side |
| `agent-core/` | Deploy unit B — everything else |
| `gpu-transcriber/` | Deploy unit C — optional; whisper on a GPU host behind an HTTP API |
| `handwriting-ocr/` | Deploy unit D — optional; OvisOCR2 handwriting OCR via llama.cpp on `10.0.7.98` |
| `packages/pa-protocol/` | The wire protocol both sides share |
| `tests/` | End-to-end tests that run both units together |
| `deploy/` | systemd, launchd and nginx examples |
| `docs/` | Protocol specification, ACP findings, ADRs, operations |

The Gateway and the Core are independently deployable and never import each other. The only thing
they share is `pa-protocol`, which contains the frame codec, the JSON-RPC peer and the Pydantic
models for every method — so a protocol change breaks both sides at import time rather than at
runtime.

`gpu-transcriber/` is optional and shares nothing at all: the Core reaches it over plain HTTP on the
LAN, and falls back to transcribing on its own CPU when it is absent. See
[ADR 0008](docs/adr/0008-transcription-service-on-the-gpu-host.md).

`handwriting-ocr/` is likewise optional and remote-only: photos are recognised by OvisOCR2 through
llama.cpp (`llama-server`) on `10.0.7.98`, with no local fallback. See
[ADR 0009](docs/adr/0009-remote-handwriting-ocr.md).

## Documentation

- [`docs/protocol.md`](docs/protocol.md) — the wire contract: methods, framing, sequence numbers,
  upload lifecycle, idempotency, reconnect.
- [`docs/cursor-acp.md`](docs/cursor-acp.md) — what `cursor-agent acp` actually does, established
  by probing the binary. Several design decisions follow directly from findings here.
- [`docs/operations.md`](docs/operations.md) — installing, running, upgrading, and what to do when
  something breaks.
- [`docs/testing.md`](docs/testing.md) — how the suites are organised and what each one proves.
- [`docs/adr/`](docs/adr/) — why the architecture is the way it is.

## Quick start (development, one machine)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e packages/pa-protocol
pip install -r agent-core/requirements.txt -r telegram-gateway/requirements.txt
pip install pytest pytest-asyncio

./scripts/test.sh          # every suite: protocol, both units, end-to-end
```

Whisper and Cursor are not needed to run the tests: the end-to-end suite fakes exactly those two
runtimes and Telegram itself, and runs everything else for real.

To run the thing rather than test it, copy `telegram-gateway/.env.example` and
`agent-core/.env.example`, fill in the bot token and a shared `CORE_TOKEN`, then see
[`docs/operations.md`](docs/operations.md).

## Priorities

When requirements conflict, this order decides:

1. Reliability
2. Correct tool execution
3. No double execution
4. Security
5. STT quality
6. State durability
7. Architectural simplicity
8. Response latency

Response latency is last on purpose. An hour of audio transcribed accurately and slowly on a CPU
is the intended behaviour, not a limitation to be optimised away.
