# Telegram Gateway

The Telegram side of the personal assistant. Runs on a Linux host with reliable access to Telegram.

This process is deliberately incapable of doing anything interesting. It has no Cursor credentials,
no Whisper, no MCP tools, no memory, no notes and no scheduler. Compromising it gets an attacker a
bot token and the ability to talk to the Core as a *service* — not as a user, because the Core
re-checks every `user_id` against its own allowlist.

## Responsibilities

- Receive Telegram updates; persist them to SQLite before anything else can go wrong.
- Download voice and audio files and stream them to the Core as binary WebSocket frames.
- Accept the Core's inbound WSS connection and serve it JSON-RPC.
- Render what the Core asks for: messages, edits, inline keyboards, chat actions.
- Retry deliveries, and never send the same `delivery_id` twice.

Everything else — what a session is, whether an action needs confirming, what to do with a
transcript — belongs to the Core.

## Layout

```
telegram_gateway/
├── main.py            composition and graceful shutdown
├── config.py          settings; two secrets and some limits
├── telegram/
│   ├── handlers.py    inbound updates → durable queue
│   ├── keyboard.py    reply keyboard and bot command menu
│   ├── renderer.py    Core intent → Bot API calls, with deduplication
│   └── formatting.py  message splitting, Markdown escaping, error wording
├── rpc/
│   ├── server.py      the WebSocket endpoint the Core dials into
│   └── transport.py
├── delivery/
│   └── service.py     durable submission with retry and backoff
└── storage/
    ├── database.py    SQLite in a worker thread
    └── models.py      transport state only
```

## Running

```bash
pip install -e ../packages/pa-protocol -r requirements.txt
cp .env.example .env      # bot token + shared CORE_TOKEN
python -m telegram_gateway.main
pytest
```

Deployment, TLS and systemd are covered in [`../docs/operations.md`](../docs/operations.md).

## Notes worth knowing

**The database is disposable.** It holds pending requests, pending uploads, delivery state, callback
tokens and sequence numbers. Losing it costs a few in-flight messages and nothing else. This is
what makes the exposed host cheap to rebuild.

**Handlers never wait for an answer.** A Telegram message is written to SQLite and the submitter is
nudged; the reply arrives later as a `telegram.send` from the Core. So a slow Cursor turn or a
disconnected Core cannot block Telegram polling or lose a message.

**Callback data carries a random token.** Telegram callback data is attacker-visible and capped at
64 bytes, so a button carries an opaque token that maps to a pending action in SQLite — never an
action id or a payload.

**One status message per job.** Progress notifications edit a single message rather than sending a
new one, and identical consecutive stages are not re-sent, because Telegram rejects a no-op edit and
rate-limits the rest.
