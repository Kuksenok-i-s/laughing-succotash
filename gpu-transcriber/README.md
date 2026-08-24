# GPU transcriber

Transcription service for the machine with the GPU. Holds one faster-whisper model in memory and
answers a small HTTP API, so the Core can hand over a recording, watch a percentage move, and
collect a transcript.

It exists because the alternative did not work: transcription used to be a set of one-shot SSH
commands per recording, with liveness guessed from `pgrep` and progress copied back as a file. See
[ADR 0008](../docs/adr/0008-transcription-service-on-the-gpu-host.md).

This process is not a general-purpose API. It has no TLS, no rate limiting and no notion of users —
one bearer token, one LAN, one client.

## Layout

```
gpu_transcriber/
├── main.py            composition: HTTP server, GPU worker, sweeper
├── config.py          settings from the environment
├── server.py          five endpoints on ThreadingHTTPServer
├── jobs.py            job registry, audio spool, queue, TTL sweep
├── worker.py          the single thread that owns the GPU
└── engine.py          faster-whisper, loaded once
```

Standard library only, apart from `faster-whisper` itself. The virtualenv on the GPU host runs a
fresh CPython, and five endpoints do not justify putting more wheels into it.

## API

Every endpoint except `/health` requires `Authorization: Bearer <GPU_STT_TOKEN>`, compared in
constant time. Job ids come from the Core (a ULID) and must be `[A-Za-z0-9_-]{1,64}`: the id becomes
a directory name.

| Request | Answer |
| --- | --- |
| `PUT /v1/jobs/{id}?language=&beam_size=&filename=` | `202` and the job snapshot. Body is the raw audio. |
| `GET /v1/jobs/{id}` | `200` with `status`, `percent`, `position_sec`, `duration_sec`, `segments`, `elapsed_sec`, `error`. |
| `GET /v1/jobs/{id}/result` | `200` with `text`, `language`, `duration`, `segments[]`. `409` and a snapshot while it is still running. |
| `DELETE /v1/jobs/{id}` | `200`. Removes the record and the audio. |
| `GET /health` | `200` with `model`, `model_loaded`, `queued`. No token. |

`status` is `queued`, `running`, `done` or `failed`. `language=auto` or an empty value means detect
it; anything else is passed to whisper as given.

`PUT` is idempotent: an id already known returns the existing snapshot instead of starting a second
transcription. Audio is streamed to disk rather than read into memory, and a body over
`GPU_STT_MAX_UPLOAD_MB` is refused before anything is written.

```bash
curl -sf localhost:17493/health
curl -sf -X PUT --data-binary @voice.ogg \
     -H "Authorization: Bearer $GPU_STT_TOKEN" \
     "localhost:17493/v1/jobs/manual01?language=auto&filename=voice.ogg"
curl -sf -H "Authorization: Bearer $GPU_STT_TOKEN" localhost:17493/v1/jobs/manual01
```

## Running

```bash
pip install -r requirements.txt        # into the venv that has CUDA-capable ctranslate2
cp service.env.example ~/.config/gpu-transcriber/service.env   # then chmod 600 and set the token
set -a && . ~/.config/gpu-transcriber/service.env && set +a
python -m gpu_transcriber.main
pytest
```

Deployment as a user systemd unit is in [`../docs/operations.md`](../docs/operations.md).

## Notes worth knowing

**The model loads after the port opens.** On an RTX 4080 that took 85 seconds reading the weights
off disk and 2 seconds on a restart while they were still in the page cache. Until the model is in
memory `/health` answers `model_loaded: false` and jobs sit in the queue. Refusing connections
instead would send the Core to its CPU fallback for as long as that process lives, which is a far
worse trade.

**The registry is in memory.** A restart loses jobs in flight; the Core sees a `404`, raises
`SttError` and transcribes on the CPU. A durable queue would instead replay an hour of GPU work that
nobody is waiting for any more.

**One job at a time.** Two large-v3 runs on one card are slower together than one after the other,
and the memory spike risks the process.

**Audio is the only thing that grows.** It is deleted when the Core collects the result, and swept
after `GPU_STT_JOB_TTL_SECONDS` otherwise — including spool directories left behind by a restart,
which no other mechanism would ever remove.

**Failures are refused connections or `failed` jobs, never silence.** That is what the Core needs to
decide to use the CPU, and what the user sees as a fallback marker in Telegram.
