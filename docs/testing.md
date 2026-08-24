# Testing

```bash
./scripts/test.sh              # all five suites
./scripts/test.sh -v -k audio  # arguments are forwarded to pytest
```

Five separate pytest runs, because each deploy unit owns a top-level `tests` package and must be
testable on its own machine without this repository's root config:

| Suite | Command | What it covers |
| --- | --- | --- |
| Protocol | `pytest packages/pa-protocol/tests` | Frame codec, JSON-RPC peer, ids, errors |
| Core | `pytest agent-core` | Storage, permissions, jobs, STT plumbing, MCP, scheduler, transcripts |
| Gateway | `pytest telegram-gateway` | Handlers, renderer, submission queue, formatting |
| GPU service | `pytest gpu-transcriber` | The HTTP contract, job registry, TTL sweep, GPU worker |
| End-to-end | `pytest tests` | Both units running together over a real WebSocket |

Nothing requires Cursor, Whisper, a GPU, a Telegram token or network access beyond loopback.

## What is faked, and what deliberately is not

Only three things are ever faked: Telegram, Cursor and Whisper. Everything else runs for real, in
every suite — including SQLite, the aiohttp RPC endpoint, the service-token handshake, both durable
queues, the job manager, the scheduler and the confirmation flow.

That is a considered choice. The properties this system needs to hold are all about what happens
when a message is delivered twice, a response is lost, or one machine disappears mid-request. A
mocked queue or a mocked socket cannot exhibit those situations, so tests built on them would only
be testing the mocks.

The fakes model semantics rather than recording calls. The Core's fake Gateway enqueues through the
real durable event log and deduplicates on `delivery_id` exactly as the real link does, so a test
of "a reminder fires while the Gateway is offline" exercises the production code path.

## The end-to-end suite

`tests/test_end_to_end.py` starts a real Gateway (aiohttp server, SQLite, renderer, submission
service) and a real Core (`agent_core.main.Core` with only `_build_backend` and `_build_stt`
overridden), and lets the Core dial in over a real WebSocket on a loopback port. Assertions are
made on what a Telegram user would see.

It covers each item of the Definition of Done:

| Scenario | Test |
| --- | --- |
| Text chat with conversation context | `test_the_conversation_keeps_its_context` |
| Voice command end to end | `test_a_voice_command_is_transcribed_and_answered` |
| Reminder fired by the Core's own scheduler | `test_a_reminder_fires_on_its_own` |
| Gateway offline when a reminder fires | `test_a_reminder_that_fires_during_an_outage_is_delivered_once` |
| Core offline when a message arrives | `test_a_message_sent_while_the_core_is_down_is_processed_later` |
| Long recording analysed, nothing executed | `test_a_long_recording_is_analysed_not_obeyed` |
| Confirmation before a dangerous action | `test_nothing_dangerous_happens_without_a_button_press` |
| Sessions isolated per user | `test_two_users_never_share_a_session` |

Outages are produced by making the Gateway reject the handshake for the duration of a block, not by
cutting the socket alone — the Core reconnects within milliseconds, so a cut socket is not an
outage. See `Harness.outage`.

## Reliability properties

The awkward cases have named tests rather than prose:

- **Duplicate delivery** — the same `delivery_id` sends one Telegram message
  (`test_the_same_delivery_twice_sends_one_telegram_message`).
- **Lost response after a write** — the same `operation_id` creates one object
  (`test_the_same_reminder_operation_twice_creates_one`, plus the ledger tests in
  `agent-core/tests/test_storage.py`).
- **Duplicate submit** — the same `request_id` maps to the same job, no second answer.
- **Interrupted upload** — replayed frames are ignored, gaps are refused, a checksum mismatch fails
  the upload rather than transcribing wrong bytes (`agent-core/tests/test_audio_upload.py`).
- **Cursor crash / Whisper failure** — the job fails, the Core stays up and keeps answering
  (`test_a_whisper_failure_fails_the_job_and_the_core_survives`).
- **Core restart** — jobs left "running" by a killed process are failed at startup instead of
  lying in `/status` forever.

## The GPU service suite

`gpu-transcriber/tests` runs the real `ThreadingHTTPServer` on an ephemeral port and fakes only the
whisper engine. Everything that broke in the SSH pipeline this service replaced was in that layer:
`test_progress_is_visible_over_http_while_the_job_runs` holds the fake engine mid-transcription and
asserts the percentage over HTTP, `test_a_job_id_that_is_not_a_plain_name_is_refused` covers path
traversal through the job id, and `test_audio_left_behind_by_a_previous_run_is_swept` covers the
only thing on that machine that grows without bound.

The Core's half is `agent-core/tests/test_stt_gpu_service.py`, against a scripted stub service. It
asserts the progress hook arrives on the event loop thread — the exact contract whose violation sent
every GPU job silently to the CPU — that a collected or failed job is deleted, and that a service
failure surfaces as `SttError` so `FallbackSTT` takes over.

## Permission and injection boundary

`agent-core/tests/test_permissions.py` and `test_mcp_server.py` assert the three tiers directly:
READ runs unattended, an explicitly requested SAFE_WRITE runs unattended, the *same* SAFE_WRITE
with `UNTRUSTED_CONTENT` provenance asks first, and DANGEROUS always asks. Provenance is set by the
Core from its own knowledge of where the turn came from and is never taken from anything the model
says.

`test_confirmations.py` covers the answer side: silence expires as a refusal, a second button press
changes nothing, another user cannot resolve someone else's prompt, and shutdown refuses everything
outstanding rather than leaving it ambiguous.

## The real Cursor tests

`agent-core/tests/test_real_cursor.py` talks to the actual Cursor Agent and is skipped unless
`PA_REAL_CURSOR_TESTS=1`, so an ordinary run never spends Cursor usage:

```bash
PA_REAL_CURSOR_TESTS=1 pytest agent-core/tests/test_real_cursor.py -v
```

Related but separate: `agent-core/tools/acp_probe` re-verifies the protocol findings in
`docs/cursor-acp.md` and exits non-zero if one has changed, naming the code path that depended on
it. Run it after every Cursor CLI upgrade — the ACP surface is undocumented and version-gated.

## Style

Test names are sentences describing the property being protected, not the function being called:
`test_a_replayed_delivery_does_not_send_twice` rather than `test_send_dedup`. When a test exists
because of a specific hazard, the comment says what the hazard is — a future reader deciding whether
a failure matters needs to know what would break in production, not what the assertion does.
