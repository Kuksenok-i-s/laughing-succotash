# Operations

Installing, running and repairing the parts of the system.

Hostnames, addresses and account names for this deployment live in
[`docs/infra.local.md`](infra.local.md) (gitignored). Placeholders such as `<gateway-host>`
and `<core-host>` refer to that file.

## Prerequisites

| Machine | Needs |
| --- | --- |
| Gateway (Linux VPS) | Python 3.12+, Telegram bot token; RPC on port 17492 |
| Core + STT | Python 3.13 (uv), `ffmpeg`, `cursor-agent` logged in |
| OCR host | llama.cpp `llama-server` + `handwriting-ocr` (OvisOCR2) |
| Spare GPU host | Former Core/STT/OCR host; units left in place as spare |

The Gateway must be reachable from the Core; the Core needs no inbound access at all. That
asymmetry is deliberate — see [ADR 0002](adr/0002-jsonrpc-over-websocket.md).

Durable Core state lives on the Core host at `$DATA_DIR`. Whisper (`gpu-transcriber`,
large-v3-turbo int8_float16 on the Core GPU — CTranslate2 4.8.1 built for CUDA 11.4 / sm_72) stays on
loopback (`:17493`). Handwriting OCR runs on `<ocr-host>`: `llama-server` serves OvisOCR2 on
loopback `:8081`, and `handwriting-ocr` publishes the job API on `:17494`. YouTube still
downloads through the VPS: one file at a time, then the file is pulled onto
`$DATA_DIR/youtube`.

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

## Core (macOS)

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
build (re-run after every `cursor-agent` upgrade, before restarting Core):

```bash
cd ~/agent-core-src/agent-core && python -m tools.acp_probe --all
```

It exits non-zero and names the affected code path if a capability the Core depends on has changed.
`plan-mcp` must stay green: Telegram chat sessions run in Cursor `plan` mode and still need MCP.

## YouTube on the gateway VPS

YouTube is blocked from the Core/GPU LAN, so downloads run on `<gateway-host>` over SSH and are
pulled back. The VPS is transit only — never `/tmp`, never a library.

Use a dedicated system user **`ytdl`**, not `root` and not `assistant` (assistant holds the bot
token). Layout on the VPS:

| Path | Purpose |
| --- | --- |
| `/var/lib/telegram-gateway/youtube/work` | Scratch job dirs (one at a time) |
| `/var/lib/telegram-gateway/youtube/cookies.txt` | Netscape cookies, mode `600`, owner `ytdl` |
| `/var/lib/telegram-gateway/youtube/venv` | `yt-dlp==2026.08.19` (pin; bump with config) |

On the Core, copy `agent-core/youtube.config.toml.example` to `DATA_DIR/youtube/config.toml`,
point `ssh_key` at a key whose public half is in `ytdl`'s `authorized_keys`, and pin the host
key:

```bash
ssh-keyscan -t ed25519 <gateway-host> >> DATA_DIR/youtube/known_hosts
```

Core refuses `remote = "root@…"` and checks `yt-dlp --version` against `download.ytdlp_version`
before the first fetch.

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

## GPU host (optional, Linux with an NVIDIA card)

Skip this unless transcription on the Core CPU is too slow to live with. The service is
`gpu-transcriber`; what it is and why it replaced an SSH pipeline is in
[ADR 0008](adr/0008-transcription-service-on-the-gpu-host.md).

```bash
git clone <repo> ~/gpu-transcriber/src
python3 -m venv ~/.assistant/venv-whisper
~/.assistant/venv-whisper/bin/pip install -r ~/gpu-transcriber/src/gpu-transcriber/requirements.txt
```

The service is never installed into the virtualenv: the unit puts the checkout's
`gpu-transcriber/` directory on `PYTHONPATH`, so an upgrade is a `git pull` and a restart. If the
checkout lives somewhere other than `~/gpu-transcriber/src`, that one line in the unit changes.

```bash
install -d -m 700 ~/.config/gpu-transcriber
install -m 600 ~/gpu-transcriber/src/gpu-transcriber/service.env.example \
        ~/.config/gpu-transcriber/service.env
openssl rand -base64 48        # GPU_STT_TOKEN here and in the Core's STT_GPU_TOKEN
"${EDITOR:-vi}" ~/.config/gpu-transcriber/service.env
```

It runs as a **user unit**, because there is no passwordless `sudo` on this machine and none is
needed. Lingering must be on, or the service will stop when the session ends:

```bash
loginctl enable-linger "$USER"          # needs sudo once; check with `loginctl show-user $USER -p Linger`
cp ~/gpu-transcriber/src/deploy/systemd/gpu-transcriber.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gpu-transcriber
journalctl --user -u gpu-transcriber -f
```

`LD_LIBRARY_PATH` in the unit is not optional. On the spare x86 GPU host `ctranslate2` loads
cuBLAS from the `nvidia` wheels inside the virtualenv; on the Core host it must point at
`~/.local/opt/ctranslate2/lib` (the CUDA 11.4 build) and `/usr/local/cuda/lib64`.

PyPI's aarch64 `ctranslate2` wheel has no CUDA. On the Core host the library is built from
CTranslate2 4.8.1 inside `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` (nvcc 11.4, sm_72,
flash-attention off). `src/ops/mean_gpu.cu` is patched: CUDA 11.4 cannot do `bfloat16 /= float`.
The recipe and the patch live in `deploy/xavier/`. Rebuild:

```bash
# on <core-user>@<core-host>; stop ollama / gpu-transcriber first so make has RAM
<repo>/deploy/xavier/build-ctranslate2-cuda.sh
```

The Core host cannot clone GitHub. If `$HOME/.local/src/CTranslate2` is missing, rsync a
`v4.8.1` tree (submodules included) onto that path, then run the script. It installs the
`.so` under `~/.local/opt/ctranslate2` and replaces the CPU wheel in
`~/.assistant/venv-whisper`. Then `GPU_STT_DEVICE=cuda` and `GPU_STT_COMPUTE_TYPE=float16`
in `~/.config/gpu-transcriber/service.env`, and the Core-host unit from
`deploy/systemd/gpu-transcriber-xavier.service`.

Verify from the Core, which is the only client that matters:

```bash
curl -sf http://<gpu-host>:17493/health
```

`model_loaded: false` right after a start, or after ten minutes of quiet, is correct: Whisper
drops `large-v3` (and OCR drops OvisOCR2) so the two can share one card. The next job reloads
the weights; `/health` still answers and jobs queue rather than fail. Then set `STT_BACKEND=gpu`,
`STT_GPU_URL` and `STT_GPU_TOKEN` in the Core's `.env` and restart it.
The Core logs `transcription service ready at ...` at startup; a `transcription service unreachable`
there means the token, the address or the firewall.

## Handwriting OCR (`handwriting-ocr`)

Optional fourth unit on `<ocr-host>` (llama.cpp). Agent Core on the Core host (`<core-host>`) reaches
this service over the LAN. A dedicated `llama-server` serves OvisOCR2 on localhost `:8081`;
the existing text 7B on `:8080` is left alone. Only the OCR job API is published on the LAN.

```bash
# on OCR host <ocr-host>
# VL weights: ~/models/OvisOCR2.Q8_0.gguf + OvisOCR2.mmproj-q8_0.gguf

install -d -m 700 ~/.config/handwriting-ocr ~/.handwriting-ocr/tmp
install -m 600 ~/handwriting-ocr/service.env.example ~/.config/handwriting-ocr/service.env
"${EDITOR:-vi}" ~/.config/handwriting-ocr/service.env   # set OCR_TOKEN

sudo cp deploy/systemd/llama-server-ocr.service /etc/systemd/system/
sudo cp deploy/systemd/handwriting-ocr-llamacpp.service /etc/systemd/system/handwriting-ocr.service
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server-ocr handwriting-ocr

# from Core host <core-host>
python3 -c 'import urllib.request; print(urllib.request.urlopen("http://<ocr-host>:17494/health").read())'
```

Then on Core (`<core-host>`) set `OCR_ENABLED=true`, `OCR_SERVICE_URL=http://<ocr-host>:17494` and
`OCR_SERVICE_TOKEN` (same value as `OCR_TOKEN`) in `.env` and restart. There is no local OCR
fallback: if the service is down, the photo job fails with `ocr_unavailable`. The VL
`llama-server` sleeps after `--sleep-idle-seconds` (3600; one hour). `POST /v1/model/unload` only clears
the OCR worker's ready flag; llama-server keeps the weights.

Deployment on 2026-09-05 uses `<ocr-release>/app` and its sibling `venv`
via `/etc/systemd/system/handwriting-ocr.service.d/90-ocr-release.conf`. The previous app and
venv remain in place. Root-only configuration backups are in
`<ocr-release>/backup` (manifest maps numbered files to
original paths; an absent file was absent before deployment). Do not copy the backed-up
service.env into logs: it contains the API token. The release passed all 53 tests on the OCR host,
then a real API upload returned all eight control lines in one pass. Active settings:
`OCR_MAX_PASSES=2`, `OCR_IDLE_UNLOAD_SECONDS=3600`, `OCR_OLLAMA_KEEP_ALIVE=1h`.

## Web search (`web-search`)

Optional unit on the Core host itself (`<core-host>`). Pure standard library — no venv, no CUDA,
no weights — so the system interpreter is enough. It binds loopback, which is the point: the Brave
key stays on this machine and is never handed to the model or sent to the Gateway.

Pick one backend. `searxng` is the default: no key, no card, no quota, and lookups stay on the
LAN's filtered DNS. `brave` is the hosted alternative.

### SearXNG (default)

Runs as a container on the Core host, bound to loopback, from `deploy/searxng/`:

```bash
# on Core host <core-host>
cd <repo>/deploy/searxng
cp .env.example .env
sed -i "s|^SEARXNG_SECRET=.*|SEARXNG_SECRET=$(openssl rand -hex 32)|" .env
docker compose up -d

curl -s "http://127.0.0.1:8888/search?q=test&format=json" | head -c 200
```

`settings.yml` lists `json` under `search.formats`; without it `/search?format=json` answers 403
and every query fails. The rate limiter is off because the only client is `web-search` on the same
host and it would otherwise throttle our own batch. Google is disabled as an upstream engine —
aggregating it through SearXNG earns captchas instead of results.

Upstream lookups go through Pi-hole at `<core-host>`, set both in the compose file's `dns:` and
daemon-wide in `deploy/docker/daemon.json`. The daemon-wide copy is insurance: Docker falls back to
Google's `8.8.8.8` when the host's `/etc/resolv.conf` holds only loopback addresses, which this
host's does (`127.0.0.53`, the systemd-resolved stub). Docker currently reads the real upstreams
from `/run/systemd/resolve/resolv.conf` and lands on Pi-hole anyway, so nothing is broken today —
but a `resolved` reconfiguration would move every container's lookups off the filtered path without
a word. **Merge** that file rather than replacing it: the existing `runtimes` block is what lets GPU
containers start. Restarting `dockerd` restarts Pi-hole and takes LAN DNS down for a few seconds,
so pick the moment.

Blocked domains come back as `0.0.0.0` in Pi-hole's NULL mode, which `guard_url` refuses as a
non-public address — so a tracker domain cannot be fetched even if a page links to it.

### Brave

Register at [api-dashboard.search.brave.com](https://api-dashboard.search.brave.com/), subscribe to
the **Search** plan (a card is required as an anti-fraud check), then *API Keys → Add API Key*.
There is no free plan for new accounts any more: Search is $5 per 1000 requests with $5 of credit
renewed monthly, roughly 1000 queries. Set the dashboard's monthly credit limit to $5 and nothing
is ever charged. Brave asks for an attribution somewhere public in exchange for the free credit.
That credit runs at **one request per second**, so set `SEARCH_PARALLEL=1`; a batch of four would
just collect 429s.

```bash
# on Core host <core-host>
install -d -m 700 ~/.config/web-search ~/.web-search/tmp
install -m 600 <repo>/web-search/service.env.example ~/.config/web-search/service.env
"${EDITOR:-vi}" ~/.config/web-search/service.env   # set SEARCH_TOKEN, and the backend if not SearXNG

# The unit has no dependencies, but this host's system python is 3.8 and it needs 3.12+, so it
# runs on a bare venv built from the same uv-managed interpreter the Core uses.
"$(sed -n 's/^command = \([^ ]*\).*/\1/p' ~/agent-core/.venv/pyvenv.cfg)" \
    -m venv ~/.assistant/venv-web-search

sudo cp deploy/systemd/web-search.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now web-search

curl -s localhost:17495/health | python3 -m json.tool
```

Then set `SEARCH_ENABLED=true`, `SEARCH_SERVICE_URL=http://127.0.0.1:17495` and
`SEARCH_SERVICE_TOKEN` (same value as `SEARCH_TOKEN`) in the Core's `.env` and restart it. The Core
logs `search service ready at ...`; a `web search warmup failed` there means the token or the
address, and is deliberately not fatal — the tools stay registered and report the failure per call.

With `SEARCH_ENABLED=false` the `web_search` and `web_fetch` MCP tools are not registered at all,
and the assistant has no network reach through MCP. That is the default. Turning it on is what gives
the model a way to pull outside content into a session, so the results carry a standing reminder
that they are content and not instructions, and anything inferred from them still needs
confirmation before it is written.

`SEARCH_FETCH_ENABLED=false` narrows it further: snippets only, no way to open a page.

Chat sessions have carried both tools since they were registered, but nothing told the agent to
prefer them over its own built-in search, so `SEARCH_ENABLED=true` now also adds a paragraph to the
operating instructions naming `web_search` and `web_fetch` and telling it not to use the built-in
one. Those instructions only travel with the **first** message of a session, so an existing
conversation keeps the old ones until the user sends `/new`.

Turning search on also gives the **YouTube factcheck** its search. That pass runs on a scoped MCP
token limited to `web_search` and `web_fetch`, released the moment the pass ends: it reads a
transcript nobody vouched for, so it must not be one prompt away from the calendar, and the second
pass must not search at all. To check the whole pipeline without sending a Telegram message:

```bash
# on the Core host
DATA_DIR=~/factcheck-run/data ~/agent-core/.venv/bin/python \
    <repo>/scripts/factcheck_video.py https://youtu.be/VIDEO_ID
```

It caches the audio and the transcript under `~/factcheck-run`, so a rerun costs only the agent
calls. An hour of audio takes about four minutes to transcribe on the Core host.

Some sources are simply unreachable from this network — `fbi.gov` does not answer at all, on either
address family. `web_fetch` reports that as a `502` and the pass falls back to the search snippet,
which is the right outcome but worth knowing when a citation looks thinner than expected.

`/health` reports `backend_reachable`. For SearXNG it probes `/config`; for Brave it reports the
outcome of the last real query, because a probe would spend quota.

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

# GPU host, if used
cd ~/gpu-transcriber/src && git pull
systemctl --user restart gpu-transcriber

cd ~/handwriting-ocr/src && git pull
systemctl --user restart handwriting-ocr

# Core host, if search is enabled
cd <repo> && git pull
sudo systemctl restart web-search
```

Restarting the transcription service loses a job in flight; the Core notices, says so in Telegram
and finishes the recording on its CPU. Restarting it while nothing is being transcribed costs
nothing but the model load.

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
trades accuracy for speed, against the stated priority. The other way out is the GPU host.

**"GPU недоступен — расшифровываю на CPU"** — the transcription service refused, timed out or
failed the job, and the recording went to the Core CPU. The reason is in the Core's log
(`transcription service ... failed`) and, if the service was up enough to log at all, in
`journalctl --user -u gpu-transcriber`. Common causes: the service was restarting, the model was
still loading, or `LD_LIBRARY_PATH` in the unit no longer matches the virtualenv's Python version,
in which case the log shows a cuBLAS load failure and nothing is ever transcribed there.

**The percentage does not move** — the Core polls the service every two seconds and only edits the
Telegram message when the number changes. A number that is frozen for fifteen minutes makes the
Core give up on the job and fall back to the CPU rather than wait forever.

**Reminders fire late** — check that the Core was running. The scheduler catches up on startup and
fires anything overdue, so a late reminder after a reboot is the designed behaviour rather than a
lost one.

## What lives where

| Data | Machine | Lost if the disk dies |
| --- | --- | --- |
| Reminders, tasks, notes, memory, contacts, calendar, sessions, YouTube library | Core (`$DATA_DIR` on `<core-host>`) | Everything the assistant knows |
| Pending requests, pending uploads, delivery state, callback tokens | Gateway | A few in-flight messages |
| Audio being transcribed, OCR spool | the Core host (same machine as Core) | The job that was running |

Back up `$DATA_DIR/core.sqlite3`. The Gateway's database is not worth backing up, which
is the point of keeping it purely transport state.

Audio is never kept. The Gateway deletes its copy once the Core acknowledges the upload, the Core
deletes its copy once transcription finishes — success or failure — and the GPU service deletes its
spooled copy when the Core collects the result, or on a TTL sweep if nobody ever does. The
transcript is the useful artefact; the recording is the sensitive one.

## Проверяемые резервные копии (Core host, сентябрь 2026)

`assistant-backup.timer` запускает локальную копию ежедневно около 04:00 по времени платы.
Инструменты: `/usr/local/lib/assistant-backup/`. Архивы: `/var/lib/assistant-backups/`,
принадлежат root; на хосте Core отдельная группа assistant-backup может только читать готовый архив.
Сохраняются два последних автоматически созданных архива.
Включены база, пользовательские каталоги и YouTube-библиотека; исключён временный `tmp/`.
SQLite снимается через online backup API, поэтому учитываются подтверждённые изменения в WAL.
Файлы копируются с проверкой неизменности размера/времени, но единой транзакции между файлами
и базой нет: для строго согласованного общего снимка остановите записывающие сервисы.

Проверка восстановления не меняет рабочие данные:

```sh
sudo python3 /usr/local/lib/assistant-backup/assistant_backup.py restore \
  --archive /var/lib/assistant-backups/ARCHIVE.tar \
  --target /var/lib/assistant-backups/NEW-RESTORE-DIRECTORY
```

Каталог назначения обязан отсутствовать. После распаковки проверяются SHA-256 каждого файла,
целостность SQLite и внешние ключи. Ошибка удаляет только созданный проверочный каталог.
Файлы восстанавливаются с закрытыми правами владельца; его executable-бит сохраняется.
Символические ссылки и специальные файлы не поддерживаются и вызывают ошибку вместо
незаметного пропуска. Фактическую замену рабочей базы инструмент не выполняет.

Локальная копия не защищает от потери платы. Хост OCR забирает архив по SSH каждый день около 05:00
через assistant-backup-pull.timer, проверяет свежесть снимка SQLite, восстанавливает в отдельный
каталог, сверяет SHA-256 и SQLite, затем сохраняет архив и удаляет проверочную распаковку.
На хосте OCR хранятся семь подтверждённых архивов; ротация выполняется только после успешной проверки.

Проверенные архивы забирает rsync **этим хостом** (`<gpu-host>`,
`<backup-storage>/assistant-backups/`) около 06:00 через user timer
`assistant-backup-sync-pull.timer`. Хост OCR не может открыть входящий SSH на эту
станцию (Wi‑Fi не принимает новые соединения с LAN), поэтому копирование идёт
в обратную сторону: станция подключается к `assistant-backup@<ocr-host>`.
Копируются только уже проверенные `.tar` и `last-success.json`. После передачи
сверяются SHA-256 через `rsync --checksum`. На хранилище остаётся тот же набор
архивов, что на хосте OCR (семь). Публичный Gateway VPS для этих архивов не используется:
на нём нет места под снимок ~9 ГБ.

На хосте OCR учётная запись `assistant-backup` читает `/var/lib/assistant-backups`
через `rrsync -ro`, `restrict` и `from="<lan-gateway>,<gpu-host>"` (станция за NAT роутера
видна хосту OCR как `<lan-gateway>`). Ключ станции:
`~/.config/assistant-backup/id_ed25519`. Shell на хосте OCR этим ключом недоступен.
Юнит push на хосте OCR (`assistant-backup-sync.timer`) не включается, пока плата
не сможет достучаться до хранилища.

## Обновление Core с проверкой и откатом

На текущем хосте Core и GPU transcriber — системные сервисы:
`systemctl status agent-core gpu-transcriber` и `journalctl -u agent-core`.
Команды с `--user` в старых инструкциях относятся к прежнему развёртыванию.
Python Core находится в `~/.venv/ (Core)`, но пакет импортируется из
`<repo>/agent-core/` (editable install).
На хосте OCR системные юниты — `llama-server-ocr` и `handwriting-ocr`.

`scripts/deploy_core.py --stage DIRECTORY` выполняется на хосте Core от root.
Пакет содержит изменяемые Python-файлы по относительным путям и `manifest.json`:
`files -> relative_path -> {before: SHA256, after: SHA256}`.
Установщик проверяет исходные и новые файлы, синтаксис, отсутствие выполняемых и ожидающих
задач; сохраняет исходники в `<core-releases>/`, останавливает Core,
заменяет перечисленные файлы и проверяет новый запуск и handshake с Gateway.
При неуспехе возвращает исходники и проверяет подключение старой версии.
Новые зависимости, секреты, миграции и юниты этим инструментом не обновляются.
Перед вызовом выполняйте проверку конфигурации через `get_settings().validate_runtime()`
из рабочего окружения Core. Между проверкой очереди и остановкой остаётся короткое окно
приёма нового задания: обновляйте во время отсутствия активности пользователя.

## Диагностический отчёт

```sh
sudo python3 /usr/local/lib/assistant-backup/assistant_health.py \
  --database $DATA_DIR/core.sqlite3 \
  --service agent-core --service gpu-transcriber \
  --endpoint stt=http://127.0.0.1:17493/health \
  --backup-directory /var/lib/assistant-backups
```

JSON содержит счётчики и возраст очередей, время последнего завершённого задания,
состояние и память сервисов, ответы health и возраст локальной копии. Пользовательские тексты
не читаются и не выводятся. Код возврата 1 означает провал проверки или превышение порогов:
ожидание задания >15 минут, выполнение >2 часов, доставка >5 минут, копия >36 часов.
Длительная задача может правомерно превышать порог — это сигнал для проверки, не команда убить её.
`off_host_verified=false` означает, что этот каталог не содержит подтверждения внешней копии.
`remote_synced=false` означает, что последний архив ещё не подтверждён на хранилище rsync.
На хосте OCR добавьте `--require-verified-backup`: отсутствие успешного восстановления последнего
архива будет ошибкой проверки. После настройки третьей копии добавьте `--require-remote-sync`.
Это проверка по запросу, а не автоматически настроенное оповещение.

После временного отказа GPU Core использует CPU и на следующем запросе спустя 60 секунд
пробует GPU снова. Перезапуск Core для выхода из CPU fallback больше не нужен.

## Права на архивы и межхостовый ключ

Core и OCR работают как `<core-user>`. На обеих платах этот пользователь не имеет доступа
к /var/lib/assistant-backups. На хосте OCR ключ находится в /var/lib/assistant-backup-client
(root, 0700; ключ 0600), инструменты — в /usr/local/lib/assistant-backup (root).
На хосте Core учётная запись assistant-backup имеет только чтение архива через принудительную
SSH-команду cat /var/lib/assistant-backups/latest.tar, с ограничением адреса <ocr-host>
и опцией restrict. Ключ не авторизован для `<core-user>` или root на хосте Core.
Ключ станции `~/.config/assistant-backup/id_ed25519` читает только
/var/lib/assistant-backups на хосте OCR через rrsync -ro; shell там нет.

Эти меры отделяют резервные копии от процессов бота. Они не заменяют разграничение
пользователей внутри общей базы Core: оно обеспечивается проверками приложения
и пользовательскими каталогами, покрытыми тестами. Администратор root видит общий архив.
Проверить внешнюю копию:

```sh
sudo python3 /usr/local/lib/assistant-backup/assistant_health.py \
  --backup-directory /var/lib/assistant-backups --require-verified-backup --require-remote-sync
```


## On-demand OCR and background transcription (2026-09-08)

On <ocr-host> set OCR_MANAGE_LLAMA_SERVICE=true and use the same local lock file
<gpu-lock> for OCR_GPU_LOCK_PATH and GPU_STT_GPU_LOCK_PATH.
Neither worker preloads. Both hold the lock until their model is unloaded.
OCR starts user llama-server-ocr.service for a job and stops it afterwards, including errors.
Install deploy/systemd/llama-server-ocr-user.service under
~/.config/systemd/user/llama-server-ocr.service without enabling it. Disable the old
system llama-server-ocr unit and remove it from handwriting-ocr Wants/After.
The Core service-user manager must remain available (lingering is already used for Whisper).
Do not manually start either model outside the shared lock.

Audio uploads and YouTube work use a separate per-user transcription queue. Text requests
continue during STT and recording analysis. Final turns in the same assistant conversation
are serialized with a lock to keep agent and MCP contexts consistent.
