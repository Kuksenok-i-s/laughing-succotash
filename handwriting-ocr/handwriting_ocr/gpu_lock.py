"""Exclusive GPU ownership shared by OCR and Whisper processes on one host."""
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def gpu_slot(path: str):
    if not path:
        yield
        return
    import fcntl
    lock = Path(path).expanduser()
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
