#!/usr/bin/env python3
"""Sequential 2B VL bake-off on one llama-server port. Never load two models at once."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path.home() / "ocr-bake"
MODELS_DIR = Path.home() / "models"
LLAMA = Path.home() / "llama.cpp/build/bin/llama-server"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

sys.path.insert(0, str(Path.home() / "handwriting-ocr"))
from handwriting_ocr.engine import LlamaCppEngine  # noqa: E402

MODELS = [
    {
        "id": "qwen3-vl-2b",
        "label": "Qwen3-VL-2B",
        "gguf": "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        "mmproj": "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
    },
    {
        "id": "qwen2-vl-2b",
        "label": "Qwen2-VL-2B",
        "gguf": "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
        "mmproj": "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf",
    },
    {
        "id": "internvl3-2b",
        "label": "InternVL3-2B",
        "gguf": "InternVL3-2B-Instruct-Q4_K_M.gguf",
        "mmproj": "mmproj-InternVL3-2B-Instruct-Q8_0.gguf",
    },
    {
        "id": "smolvlm2-2.2b",
        "label": "SmolVLM2-2.2B",
        "gguf": "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        "mmproj": "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
    },
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = BOLD if bold and BOLD.exists() else FONT
    return ImageFont.truetype(str(path), size)


def _paper(size: tuple[int, int]) -> Image.Image:
    img = Image.new("RGB", size, (236, 228, 208))
    return img


def write_cases() -> list[dict]:
    images = ROOT / "images"
    images.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []

    img = _paper((900, 520))
    draw = ImageDraw.Draw(img)
    draw.text((48, 36), "Список", font=_font(42, True), fill=(30, 30, 30))
    for i, line in enumerate(["молоко", "хлеб", "яйца 10 шт", "кофе"]):
        draw.text((64, 120 + i * 80), f"— {line}", font=_font(36), fill=(40, 40, 40))
    path = images / "ru_list.png"
    img.save(path)
    cases.append(
        {
            "id": "ru_list",
            "path": path,
            "want_kind": "text",
            "need": ["молоко", "хлеб"],
        }
    )

    img = _paper((1000, 420))
    draw = ImageDraw.Draw(img)
    draw.text((40, 40), "Вт 19:00  зал", font=_font(34, True), fill=(25, 25, 25))
    draw.text((40, 120), "присед 5x5  100 кг", font=_font(32), fill=(35, 35, 35))
    draw.text((40, 190), "жим лёжа  4x6  80", font=_font(32), fill=(35, 35, 35))
    draw.text((40, 260), "подтягивания  3x8", font=_font(32), fill=(35, 35, 35))
    path = images / "ru_workout.png"
    img.save(path)
    cases.append(
        {
            "id": "ru_workout",
            "path": path,
            "want_kind": "text",
            "need": ["присед", "жим"],
        }
    )

    img = _paper((880, 300))
    draw = ImageDraw.Draw(img)
    draw.text((40, 80), "BUY MILK", font=_font(56, True), fill=(20, 20, 20))
    draw.text((40, 180), "and bread", font=_font(40), fill=(40, 40, 40))
    path = images / "en_note.png"
    img.save(path)
    cases.append(
        {
            "id": "en_note",
            "path": path,
            "want_kind": "text",
            "need": ["milk"],
        }
    )

    img = Image.new("RGB", (720, 480), (40, 70, 110))
    draw = ImageDraw.Draw(img)
    draw.ellipse((220, 90, 500, 370), fill=(180, 40, 40), outline=(20, 20, 20), width=6)
    draw.rectangle((330, 360, 390, 450), fill=(90, 40, 30))
    path = images / "scene_mug.png"
    img.save(path)
    cases.append(
        {
            "id": "scene_mug",
            "path": path,
            "want_kind": "other",
            "need": [],
        }
    )
    return cases


def wait_models(timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8081/v1/models", timeout=2) as resp:
                last = resp.read().decode()
            if last:
                return
        except Exception as exc:
            last = str(exc)
        time.sleep(1.5)
    raise RuntimeError(f"llama-server did not answer /v1/models: {last}")


def start_server(model: dict) -> subprocess.Popen:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        f"{Path.home() / 'llama.cpp/build/bin'}:"
        "/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    cmd = [
        str(LLAMA),
        "-m",
        str(MODELS_DIR / model["gguf"]),
        "--mmproj",
        str(MODELS_DIR / model["mmproj"]),
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
        "-ngl",
        "99",
        "-c",
        "16384",
        "-np",
        "1",
        "--alias",
        model["id"],
    ]
    log = (ROOT / "logs").joinpath(f"{model['id']}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w")
    proc = subprocess.Popen(
        cmd,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    return proc


def stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def contains(hay: str, needle: str) -> bool:
    return needle.casefold() in (hay or "").casefold()


def score(case: dict, result: dict) -> dict:
    blob = " ".join(
        [
            result.get("raw_text") or "",
            result.get("markdown") or "",
            result.get("description") or "",
        ]
    )
    kind_ok = result.get("kind") == case["want_kind"]
    hits = [word for word in case["need"] if contains(blob, word)]
    miss = [word for word in case["need"] if not contains(blob, word)]
    text_ok = not case["need"] or (len(hits) == len(case["need"]))
    return {
        "kind_ok": kind_ok,
        "text_ok": text_ok,
        "hits": hits,
        "miss": miss,
        "ok": kind_ok and text_ok,
    }


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    cases = write_cases()
    report: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "models": []}

    for model in MODELS:
        gguf = MODELS_DIR / model["gguf"]
        mmproj = MODELS_DIR / model["mmproj"]
        if not gguf.exists() or not mmproj.exists():
            report["models"].append(
                {
                    "id": model["id"],
                    "label": model["label"],
                    "error": f"missing files {gguf.name} / {mmproj.name}",
                    "cases": [],
                }
            )
            continue

        print(f"=== {model['label']} ===", flush=True)
        proc = None
        row: dict = {"id": model["id"], "label": model["label"], "cases": []}
        try:
            load_started = time.monotonic()
            proc = start_server(model)
            wait_models()
            row["load_sec"] = round(time.monotonic() - load_started, 2)
            engine = LlamaCppEngine(
                llama_url="http://127.0.0.1:8081",
                model=model["id"],
                request_timeout=180.0,
            )
            engine.load()
            for case in cases:
                print(f"  {case['id']} ...", flush=True)
                started = time.monotonic()
                try:
                    result = engine.recognize(case["path"])
                    err = None
                except Exception as exc:
                    result = {}
                    err = f"{type(exc).__name__}: {exc}"
                judged = score(case, result) if not err else {"ok": False, "kind_ok": False, "text_ok": False, "hits": [], "miss": case["need"]}
                row["cases"].append(
                    {
                        "id": case["id"],
                        "want_kind": case["want_kind"],
                        "elapsed_sec": round(time.monotonic() - started, 2),
                        "kind": result.get("kind"),
                        "passes": result.get("passes"),
                        "raw_text": (result.get("raw_text") or "")[:400],
                        "markdown": (result.get("markdown") or "")[:400],
                        "description": (result.get("description") or "")[:240],
                        "error": err,
                        **judged,
                    }
                )
                print(
                    f"    kind={result.get('kind')} ok={judged.get('ok')} "
                    f"{row['cases'][-1]['elapsed_sec']}s",
                    flush=True,
                )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL {row['error']}", flush=True)
        finally:
            stop_proc(proc)
            time.sleep(2)
        wins = sum(1 for item in row["cases"] if item.get("ok"))
        row["wins"] = wins
        row["n"] = len(cases)
        report["models"].append(row)

    out = ROOT / "results.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
