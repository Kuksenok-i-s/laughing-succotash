"""Own CUDA in a spawned process, so idle shutdown releases native allocations."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

from .engine import ProgressHook, WhisperEngine


def _serve(connection, factory, options):
    try:
        engine = factory(**options)
        engine.load()
        connection.send(("ready", None))
        while True:
            request = connection.recv()
            if request is None:
                return
            path, language, beam_size = request
            result = engine.transcribe(
                Path(path), language=language, beam_size=beam_size,
                on_progress=lambda *args: connection.send(("progress", args)),
            )
            connection.send(("result", result))
    except EOFError:
        pass
    except Exception as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class ProcessWhisperEngine:
    """One worker thread owns requests; health checks only read readiness."""

    def __init__(self, *, model: str, factory=WhisperEngine, **options: Any):
        self._options = {"model": model, **options}
        self._factory = factory
        # Never fork a threaded HTTP server or an initialized CUDA runtime.
        self._context = multiprocessing.get_context("spawn")
        self._process = None
        self._connection = None
        self._loaded = False

    @property
    def model_name(self) -> str:
        return self._options["model"]

    @property
    def ready(self) -> bool:
        process = self._process
        return bool(self._loaded and process and process.is_alive())

    def _receive(self, timeout=None):
        if timeout is not None and not self._connection.poll(timeout):
            raise RuntimeError("Whisper process startup timed out")
        try:
            kind, value = self._connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError("Whisper process exited unexpectedly") from exc
        if kind == "error":
            raise RuntimeError(value)
        return kind, value

    def load(self) -> None:
        if self.ready:
            return
        self.unload()
        parent, child = self._context.Pipe()
        self._connection = parent
        self._process = self._context.Process(
            target=_serve, args=(child, self._factory, self._options), daemon=True,
        )
        try:
            self._process.start()
            child.close()
            kind, _ = self._receive(timeout=180)
            if kind != "ready":
                raise RuntimeError("Invalid Whisper startup response")
            self._loaded = True
        except BaseException:
            child.close()
            self.unload()
            raise

    def unload(self) -> None:
        self._loaded = False
        process = self._process
        if process is not None and process.pid is not None:
            # Process exit releases CUDA pools, libc arenas and model state together.
            if process.is_alive():
                try:
                    self._connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                process.join(timeout=5)
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            if process.is_alive():
                raise RuntimeError("Whisper process did not stop")
        if self._connection is not None:
            self._connection.close()
        self._process = None
        self._connection = None

    def transcribe(
        self, audio_path: Path, *, language: str | None,
        beam_size: int | None, on_progress: ProgressHook | None = None,
    ) -> dict[str, Any]:
        self.load()
        try:
            self._connection.send((str(audio_path), language, beam_size))
            while True:
                kind, value = self._receive()
                if kind == "result":
                    return value
                if kind != "progress":
                    raise RuntimeError("Invalid Whisper worker response")
                if on_progress is not None:
                    on_progress(*value)
        except BaseException:
            self.unload()
            raise
