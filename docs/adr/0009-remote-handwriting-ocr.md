# 9. Remote handwriting OCR via Qwen3-VL on the GPU host

Status: accepted

## Context

Handwritten notes arrive as Telegram photos. The Gateway and Core must not run vision models
locally: Cursor remains the conversational agent (ADR 0003), and Whisper already has its own
remote GPU path (ADR 0008). OCR needs a separate, unloadable model lifecycle because Qwen3-VL and
Whisper compete for the same VRAM.

## Decision

A fourth deploy unit, `handwriting-ocr`, on the GPU host `10.0.7.49` (where Ollama already runs).
Agent Core on `10.0.7.46` is the only LAN client:

1. Telegram photo / `image/*` document is accepted automatically (and again via `/ocr` on reply).
2. Bytes travel Gateway → Core over the existing binary framing, under `image.begin` /
   `image.commit` / `image.abort`.
3. Core uploads the image to `handwriting-ocr` over LAN HTTP with a bearer token.
4. The service runs **exactly two** Ollama `/api/chat` calls against Qwen3-VL:
   - pass 1: original image → literal `raw_text`;
   - pass 2: original image + that `raw_text` → visual check, correction, clean Markdown.
5. Core analyses the Markdown as `Provenance.UNTRUSTED_CONTENT`. Memory and contacts are available
   as read tools on the final turn; writes still require confirmation (ADR 0007).
6. There is **no local OCR fallback**. An unreachable service fails the job with
   `ocr_unavailable`.
7. After `OCR_IDLE_UNLOAD_SECONDS` (default 600) of no jobs, the worker unloads (`keep_alive: 0`)
   so Whisper can use the same card. `POST /v1/model/load` and `/v1/model/unload` remain for
   explicit control. `OCR_OLLAMA_KEEP_ALIVE` defaults to `10m` so Ollama does not drop the weights
   before the worker does.

PaddleOCR-VL is deliberately not used: the pipeline is Qwen3-VL only, via Ollama on localhost of
the GPU host. Only the narrow OCR HTTP API is published on the LAN.

## Consequences

- Photos become a first-class intake path alongside voice.
- OCR and Whisper must not be scheduled onto the same card at once without an unload step.
- Existing installs keep OCR off until `OCR_ENABLED=true` and `OCR_SERVICE_TOKEN` are set.
