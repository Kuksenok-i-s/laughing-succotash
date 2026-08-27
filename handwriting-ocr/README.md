# Handwriting OCR / photo triage

Qwen3-VL on the GPU host `10.0.7.49` (Ollama). Agent Core on `10.0.7.46` uploads a photo and
collects either a scene description or clean Markdown.

```
original image
       │
       ▼
 Qwen3-VL pass 1 (triage, colour)
       ├─ kind=other  → description (1 pass total)
       └─ kind=text   → grayscale+contrast
                            │
                            ▼
                      pass 2: correct raw text
                            │
                            ▼
                      pass 3: clean Markdown
```

This process is not a general-purpose API. It has no TLS, no rate limiting and no notion of users —
one bearer token, one LAN, one client. Ollama itself stays on localhost.

## Layout

```
handwriting_ocr/
├── main.py            composition: HTTP server, OCR worker, sweeper
├── config.py          settings from the environment
├── server.py          job + model lifecycle endpoints
├── jobs.py            job registry, image spool, queue, TTL sweep
├── worker.py          the single thread that owns the GPU slot
├── preprocess.py      grayscale + contrast for OCR passes
└── engine.py          triage + optional two OCR passes via Ollama
```

## API

Every endpoint except `/health` requires `Authorization: Bearer <OCR_TOKEN>`.

| Request | Answer |
| --- | --- |
| `PUT /v1/jobs/{id}?filename=&content_type=` | `202` and the job snapshot. Body is the raw image. |
| `GET /v1/jobs/{id}` | `200` with `status`, `percent`, `stage`, `elapsed_sec`, `error`. |
| `GET /v1/jobs/{id}/result` | `200` with `kind`, `raw_text`, `markdown`, `description`, `model`, `elapsed_seconds`, `passes`. |
| `DELETE /v1/jobs/{id}` | `200`. Removes the record and the image. |
| `POST /v1/model/load` | Load Qwen3-VL into Ollama VRAM. |
| `POST /v1/model/unload` | Unload immediately (`keep_alive: 0`). Also happens after `OCR_IDLE_UNLOAD_SECONDS` of quiet. |
| `GET /health` | `200` with `model`, `model_loaded`, `ollama_reachable`, `queued`. |

`status` is `queued`, `running`, `done` or `failed`. Stages while running are `recognizing` then
`structuring`. After `OCR_IDLE_UNLOAD_SECONDS` (default 600) of no jobs the worker unloads Qwen3-VL
(`keep_alive: 0`); the next photo loads it again. Set `OCR_OLLAMA_KEEP_ALIVE` at least as long as
that idle window so Ollama does not drop the weights first.

## Running

```bash
cp service.env.example ~/.config/handwriting-ocr/service.env   # chmod 600, set the token
# pull the vision model once: ollama pull qwen3-vl
set -a && . ~/.config/handwriting-ocr/service.env && set +a
PYTHONPATH=. python -m handwriting_ocr.main
pytest
```

Deployment as a user systemd unit is in [`../docs/operations.md`](../docs/operations.md).
