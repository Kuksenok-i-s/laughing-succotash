# Handwriting OCR / photo triage

Qwen3-VL on `10.0.7.98` (llama.cpp `llama-server`) or on a host with Ollama. Agent Core uploads
a photo and collects either a scene description or clean Markdown.

Each image gets EXIF orientation correction and conservative document detection with OpenCV.
A large, closed, bright quadrilateral on a darker background is cropped and perspective-corrected
at source resolution before applying `OCR_IMAGE_MAX_EDGE`. Ambiguous images and pages touching
the frame keep their full extent. The resolution cap is unchanged.

One page pass produces the transcription (`OCR_PIPELINE=ocr`) or text/scene classification
(`triage`). With `OCR_MAX_PASSES=2` (default), at most one small unreadable fragment is retried.
The first response must locate it using `[?bbox:x1,y1,x2,y2|…?]`, with coordinates in 0–1000.
The crop comes from the rectified source, not the resized page, and cannot exceed 25% of its area.
For light paper, the retry removes blank margins with padding, normalizes contrast and enlarges
small text up to 3× within the same image cap. Dark or nearly uniform fragments retain their
original appearance. Thresholding is used for margin detection only. Only that placeholder is replaced. Missing/invalid coordinates, uncertain or failed retries keep
`[?…?]`; ordinary readable pages need one request. Legacy values above 2 also allow only one retry.
`OCR_MAX_PASSES=1` disables retries. There are no repeated full-page or Markdown-only model passes.
Model localization is best effort and needs validation on real photos from the deployed OCR model.

This process is not a general-purpose API. It has no TLS, no rate limiting and no notion of users —
one bearer token, one LAN, one client. The infer backend (`OCR_BACKEND=llamacpp` or `ollama`)
stays on localhost.

## Layout

```
handwriting_ocr/
├── main.py            composition: HTTP server, OCR worker, sweeper
├── config.py          settings from the environment
├── server.py          job + model lifecycle endpoints
├── jobs.py            job registry, image spool, queue, TTL sweep
├── worker.py          the single thread that owns the GPU slot
├── preprocess.py      document rectification and source-resolution crops
└── engine.py          one page pass + optional fragment retry
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

`status` is `queued`, `running`, `done` or `failed`. Recognition uses the `recognizing` stage.
After `OCR_IDLE_UNLOAD_SECONDS` (default 3600) of no jobs the worker unloads Qwen3-VL
(`keep_alive: 0`); the next photo loads it again. Set `OCR_OLLAMA_KEEP_ALIVE` at least as long as
that idle window so Ollama does not drop the weights first (default `1h`).
For llama.cpp, the systemd unit uses `--sleep-idle-seconds 3600`; the worker unload endpoint
only clears its readiness flag. Update existing service.env overrides and restart the services
to apply these settings. A longer warm window retains model memory for longer.

## Running

```bash
cp service.env.example ~/.config/handwriting-ocr/service.env   # chmod 600, set the token
# pull the vision model once: ollama pull qwen3-vl
set -a && . ~/.config/handwriting-ocr/service.env && set +a
PYTHONPATH=. python -m handwriting_ocr.main
pytest
```

Deployment as a user systemd unit is in [`../docs/operations.md`](../docs/operations.md).
