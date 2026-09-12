# GPU transcriber

Transcription service for the machine with the GPU. Loads one faster-whisper model, answers a
small HTTP API, and drops the weights after ten idle minutes so OCR can share the card.

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
├── worker.py          the single thread that dispatches GPU jobs
├── chunks.py          split long audio, stitch transcripts
├── process_engine.py  spawned model process, progress and result transport
└── engine.py          faster-whisper inside the model process
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

**The model loads after the port opens, and unloads after ten idle minutes.** Until the weights
are in memory `/health` answers `model_loaded: false` and jobs sit in the queue. Refusing
connections instead would send the Core to its CPU fallback for as long as that process lives.
`GPU_STT_IDLE_UNLOAD_SECONDS=600` (0 disables) drops the model so OCR can use the same card; the
next job reloads it. On a host without OCR set it to 0: a cold load costs tens of seconds per
voice note.

**Decoding is batched.** Silero VAD cuts the file into speech windows and `GPU_STT_BATCH_SIZE`
(default 8) of them go through the model together via `BatchedInferencePipeline`; windows are
decoded independently (`condition_on_previous_text=False`), so a hallucination cannot seed the
next window. `GPU_STT_BATCH_SIZE=0` or `GPU_STT_VAD_FILTER=false` falls back to one window at a
time. Default beam is 2: the decoder is the bottleneck and turbo barely moves between 2 and 5.

Whisper runs in a separate spawned process. Idle unloading terminates and joins that process,
releasing its CUDA context and native allocator pools; the HTTP server and job registry stay
alive. A failed model process fails the current job, and the next job starts a fresh process.
The child inherits the service's systemd memory cgroup. Clearing a Python reference alone does
not guarantee that native memory is returned to the system.

**The registry is in memory.** A restart loses jobs in flight; the Core sees a `404`, raises
`SttError` and transcribes on the CPU. A durable queue would instead replay an hour of GPU work that
nobody is waiting for any more.

**One job at a time on this card.** Two large-v3 runs on one GPU are slower together than one after
the other. A recording longer than ``GPU_STT_CHUNK_SECONDS`` (five minutes) is split and the slices
run in sequence on this worker; the Core may send alternate slices to a second Jetson when OCR
is idle there.

**Audio is prepared before Whisper.** One ffmpeg pass applies `highpass=f=80`, EBU loudnorm
at -16 LUFS, and 16 kHz mono PCM. Quiet voice notes otherwise starve Silero VAD; rumble
below 80 Hz is not speech. If ffmpeg fails the original file still goes to the model. An
upload with `?prepared=1` (a DualHost slice the Core already filtered) skips this pass.

**Windows are cut while the GPU decodes.** A long file is split into 300 s windows by two
ffmpeg threads running ahead of the decoder, so filter time hides behind decode time instead
of adding to it. Each window is deleted as soon as its transcript is in.

**On a shared card the weights stay warm between jobs.** With `GPU_STT_GPU_LOCK_PATH` set the
slot lock is held only for the duration of a job; afterwards the model stays loaded for
`GPU_STT_IDLE_UNLOAD_SECONDS` so the next slice of the same recording does not pay a cold
load, and is dropped at once if another process (OCR) takes the slot.

**Audio is the only thing that grows.** It is deleted when the Core collects the result, and swept
after `GPU_STT_JOB_TTL_SECONDS` otherwise — including spool directories left behind by a restart,
which no other mechanism would ever remove.

**Failures are refused connections or `failed` jobs, never silence.** That is what the Core needs to
decide to use the CPU, and what the user sees as a fallback marker in Telegram.
