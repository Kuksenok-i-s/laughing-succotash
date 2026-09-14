"""Exclusive GPU ownership shared by OCR and Whisper processes on one host."""
from contextlib import contextmanager
from pathlib import Path
import fcntl


@contextmanager
def gpu_slot(path: str):
    if not path:
        yield
        return
    lock = Path(path).expanduser()
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def gpu_claimed_elsewhere(path: str) -> bool:
    """True when another process holds the slot right now.

    A non-blocking probe: the caller is not taking the GPU, only asking whether the weights it
    keeps warm should be dropped so the holder gets the memory.
    """
    if not path:
        return False
    lock = Path(path).expanduser()
    if not lock.exists():
        return False
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False
