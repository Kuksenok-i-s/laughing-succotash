# 4. Local faster-whisper large-v3, accuracy over speed

Status: accepted

## Context

Voice is a primary input. Recordings range from a three-second command to a multi-hour meeting.
The content is personal and confidential. Cursor's ACP build reports
`promptCapabilities.audio == false`, so audio cannot reach the agent as audio under any
arrangement.

## Decision

Transcription runs locally on the Mac mini with `faster-whisper`, default model `large-v3`. The
explicit priority is **accuracy over speed**; a long recording taking a long time is acceptable.

The model is loaded once and reused. Inference is CPU-bound, so it runs in a worker thread and
never on the asyncio event loop. `STT_MAX_CONCURRENT=1` by default, appropriate for an Intel Mac
mini where two concurrent large-v3 transcriptions would thrash rather than parallelise.

The rest of the system sees only the `SpeechToText` protocol and plain dataclasses
(`TranscriptionResult`, `TranscriptSegment`). No `faster_whisper` object escapes the adapter.

## Consequences

No audio leaves the machine, and there is no per-minute transcription cost.

Transcription is slow enough that it must be a job with progress reporting, not a blocking call —
which the async job model already provides.

Because the adapter boundary is a narrow protocol, swapping to `whisper.cpp` or a Metal build
later touches one file. Segment timestamps are preserved because transcript ordering must never be
lost, and they are what makes hierarchical chunking of long recordings possible.
