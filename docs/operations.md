# Operations

Installing, running and repairing the two halves of the system.

## Prerequisites

| Machine | Needs |
| --- | --- |
| Gateway (Linux) | Python 3.12+, a domain with a TLS certificate, a Telegram bot token |
| Core (Intel Mac mini) | Python 3.12+, `ffmpeg`, `cursor-agent` logged in, ~4 GB free for model weights |

The Gateway must be reachable from the Core; the Core needs no inbound access at all. That
asymmetry is deliberate — see [ADR 0002](adr/0002-jsonrpc-over-websocket.md).

## Shared secret

One token authenticates the Core to the Gateway. Generate it once:

```bash
openssl rand -base64 48
```

Put the same value in `CORE_TOKEN` on both machines. It is compared in constant time on the
Gateway; a timing oracle on it would be enough to impersonate the Core.

The Core additionally needs `MCP_TOKEN` (any long random string), which authenticates Cursor's tool
calls to the loopback MCP server. Neither token is ever logged or included in `/status`.

## Gateway (Linux)

```bash
sudo useradd --system --create-home --home-dir /opt/telegram-gateway assistant
sudo -u assistant git clone <repo> /opt/telegram-gateway/src
cd /opt/telegram-gateway/src

sudo -u assistant python3.12 -m venv /opt/telegram-gateway/.venv
sudo -u assistant /opt/telegram-gateway/.venv/bin/pip install -e packages/pa-protocol
sudo -u assistant /opt/telegram-gateway/.venv/bin/pip install -r telegram-gateway/requirements.txt
sudo -u assistant /opt/telegram-gateway/.venv/bin/pip install -e telegram-gateway

sudo install -d -m 700 -o assistant /etc/telegram-gateway
sudo install -m 600 -o assistant telegram-gateway/.env.example /etc/telegram-gateway/gateway.env
sudo -u assistant "${EDITOR:-vi}" /etc/telegram-gateway/gateway.env
```

TLS termination goes in front of the RPC port; `deploy/nginx/gateway.conf` is a working example.
The long read timeout matters: without it nginx closes the idle link every minute and the Core
reconnects for nothing.

```bash
sudo cp deploy/systemd/telegram-gateway.service /etc/systemd/system/
sudo systemctl enable --now telegram-gateway
journalctl -u telegram-gateway -f
```

`GET /health` answers `{"status":"ok","core_connected":false}` before the Core connects. That is
the correct state, not an error — the Gateway accepts and queues messages regardless.

## Core (Intel Mac mini)

```bash
git clone <repo> ~/agent-core-src && cd ~/agent-core-src
python3.12 -m venv ~/agent-core/.venv
~/agent-core/.venv/bin/pip install -e packages/pa-protocol
~/agent-core/.venv/bin/pip install -r agent-core/requirements.txt
~/agent-core/.venv/bin/pip install -r agent-core/requirements-stt.txt   # Whisper
~/agent-core/.venv/bin/pip install -e agent-core

brew install ffmpeg
cursor-agent login && cursor-agent status
```

Configuration:

```bash
mkdir -p ~/agent-core
cp agent-core/.env.example ~/agent-core/.env && chmod 600 ~/agent-core/.env
cp agent-core/assistant.toml.example ~/.personal-assistant/assistant.toml
```

`assistant.toml` is the filesystem and project allowlist. Nothing outside it is reachable; `$HOME`
is never opened wholesale. A project listed with `writable = false` is opened in Cursor's `plan`
mode, which was verified to genuinely refuse writes (see `docs/cursor-acp.md`).

Before installing the service, confirm the ACP findings still hold on this machine and this CLI
build:

```bash
cd ~/agent-core-src/agent-core && python -m tools.acp_probe --all
```

It exits non-zero and names the affected code path if a capability the Core depends on has changed.

Then install the launch agent:

```bash
cp deploy/launchd/com.assistant.agent-core.plist ~/Library/LaunchAgents/
# edit the paths inside first
launchctl load -w ~/Library/LaunchAgents/com.assistant.agent-core.plist
tail -f ~/Library/Logs/agent-core.log
```

It is a **user agent, not a daemon**, because `cursor-agent` authenticates as the logged-in user.
The consequence is that the Mac must be set to log this user in automatically after a reboot
(System Settings → Users & Groups → Automatic login), or the Core will not start — and a Core that
is not running is a reminder that does not fire.

The first voice message downloads the large-v3 weights (~3 GB). Warm them up deliberately if you
would rather not have that happen mid-conversation.

## First run checklist

1. `journalctl -u telegram-gateway` shows `core ... connected (capabilities: ...)`.
2. `/start` in Telegram returns the help text.
3. A text message gets an answer.
4. `/status` shows `Cursor: ready` and `Whisper: idle` (idle is correct until first use).
5. A voice message gets transcribed and answered.
6. "Напомни через 2 минуты проверить бот" fires two minutes later.

## Upgrading

Both units tolerate the other being absent, so they can be restarted independently and in any
order. Restarting the Core mid-request is safe: the Gateway still holds the request and resubmits
it with the same `request_id`, which the Core deduplicates.

```bash
# Gateway
cd /opt/telegram-gateway/src && sudo -u assistant git pull
sudo systemctl restart telegram-gateway

# Core
cd ~/agent-core-src && git pull
launchctl kickstart -k gui/$(id -u)/com.assistant.agent-core
```

Database migrations run automatically at startup and are additive only.

After a Cursor CLI upgrade, re-run `python -m tools.acp_probe --all`. The ACP surface is
undocumented and version-gated; assuming it is unchanged is how a silent breakage happens.

## Rotating the service token

Set the new `CORE_TOKEN` on the Gateway, restart it, then set it on the Core and restart that. In
between, the Core's connections are rejected and the Gateway queues everything, so the only cost is
delay. Do not rotate in the other order unless a few minutes of rejected reconnects is acceptable.

## Diagnosing

**"Ядро сейчас недоступно"** — the Gateway has no Core connection. Check the Core's log for a
handshake failure: a `protocol_version_unsupported` error means the two sides are on different
commits, and a 401 means the tokens differ.

**Replies stop arriving but Telegram works** — look for `dropping event seq=... ` in the Core log.
The Gateway refused an event permanently, which happens when the bot is blocked by the user.

**A job is stuck** — `/status` shows what is running. `/cancel` stops it; the conversation survives
cancellation (verified against the real Cursor). If the Core is restarted while a job runs, that
job is marked failed at next startup rather than left claiming to run.

**Transcription is very slow** — expected. large-v3 on an Intel CPU runs at roughly real time or
slower. `STT_MAX_CONCURRENT=1` is deliberate: two parallel runs on this hardware are slower than
two sequential ones and risk the memory of the whole process. If it is unusable, `STT_MODEL=medium`
trades accuracy for speed, against the stated priority.

**Reminders fire late** — check that the Core was running. The scheduler catches up on startup and
fires anything overdue, so a late reminder after a reboot is the designed behaviour rather than a
lost one.

## What lives where

| Data | Machine | Lost if the disk dies |
| --- | --- | --- |
| Reminders, tasks, notes, memory, contacts, calendar, sessions | Core | Everything the assistant knows |
| Pending requests, pending uploads, delivery state, callback tokens | Gateway | A few in-flight messages |

Back up `~/.personal-assistant/core.sqlite3`. The Gateway's database is not worth backing up, which
is the point of keeping it purely transport state.

Audio is never kept. The Gateway deletes its copy once the Core acknowledges the upload, and the
Core deletes its copy once transcription finishes — success or failure. The transcript is the useful
artefact; the recording is the sensitive one.
