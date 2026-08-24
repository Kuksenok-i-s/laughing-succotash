# 8. A transcription service on the GPU host, instead of SSH per job

Status: accepted

Supersedes the remote half of [ADR 0004](0004-local-whisper.md); local CPU whisper stays as the
fallback described there.

## Context

`STT_BACKEND=gpu` was implemented as a set of one-shot SSH commands per recording: copy a runner
script to the GPU host, start it under `nohup`, decide whether it was still alive with `pgrep`, and
pull `progress.json` back over `scp` every thirty seconds. The Mini also re-encoded every recording
to mp3 first.

In one evening that path produced four distinct failures, none of which shared a cause:

- `pgrep -f <path>/run.py` matched the SSH command that was running the check, so a job that had
  already exited looked alive forever.
- `file -b` was used to validate the uploaded audio and reported success for a path that did not
  exist, because the path itself contained a word it recognised.
- The progress callback was invoked from the SSH polling thread. Callers schedule coroutines from
  that hook, which raised "there is no current event loop" on the first tick — so every GPU job
  failed instantly and silently ran on the CPU instead.
- The launch command blocked. `bash` groups `A && B` into a subshell that inherits the SSH channel's
  stdout, so `nohup` on `B` did not detach it, and the polling loop never started.

Each one was a small bug. What they had in common is the design: liveness inferred from process
tables, progress transported as a file, and a model reloaded from scratch for every recording —
which cost more than most of the transcriptions themselves.

## Decision

A long-running service on the GPU host, `gpu-transcriber`, with a small HTTP API. The model is
loaded once at startup and stays in memory. A job is a `PUT` with the audio as the body; progress is
a field in a JSON response; a finished job is deleted by the client and, failing that, swept on a
TTL.

The Core talks to it over the LAN with a bearer token and no TLS. The service is written against the
standard library only — `ThreadingHTTPServer` plus `json` — because it lives inside the virtualenv
that already holds faster-whisper, and adding wheels to a fresh CPython for five endpoints is risk
without return.

`FallbackSTT` is unchanged: a transport error or a `failed` job raises `SttError` and the recording
goes to local CPU whisper, now with a visible marker in Telegram.

## Consequences

Progress reaches Telegram every two seconds instead of every thirty, and no longer depends on
copying a file across the network. The percentage now moves during a transcription, which is what
distinguishes a long job from a hung one.

Language detection works. The SSH runner took the language as a positional argument and always
received a hardcoded `ru`; the service treats `auto` as "detect" and passes `None` to whisper.

Nothing accumulates on disk. The mp3 re-encode on the Mini is gone because the service accepts the
original file, and the spool directory is swept on a TTL rather than by hand.

The cost is a third machine in the deployment, and a service that must be restarted after an
upgrade. Two known windows exist: for up to a minute and a half after a cold start the model is
still loading and jobs wait in the queue, and a restart mid-job loses that job. The first only delays a
recording; the second ends with the Core falling back to CPU, which is visible to the user rather
than silent.

The API is deliberately narrow — five endpoints, no batching, one job at a time. Two large-v3 runs
on one card are slower together than one after the other, so concurrency here would buy nothing.
