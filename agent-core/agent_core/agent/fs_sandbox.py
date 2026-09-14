"""Per-session filesystem containment for Cursor ACP.

Built-in ACP file tools do not go through the permission callback. Containment is therefore:
the session's ``cwd`` is the user's directory, and every ``fs/read_text_file`` /
``fs/write_text_file`` path is checked against that root (default deny).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# Telegram (and other) numeric ids only — rejects path traversal in the directory name.
_USER_DIR_ID = re.compile(r"^[0-9]+$")


def telegram_dir_id(user_id: str) -> str:
    """Map a namespaced user id (``tg:123``) to the filesystem suffix ``123``.

    Raises ``ValueError`` if the id is not a safe numeric Telegram id.
    """
    text = (user_id or "").strip()
    if ":" in text:
        _, _, rest = text.partition(":")
        text = rest.strip()
    if not text or not _USER_DIR_ID.fullmatch(text):
        raise ValueError(f"refusing unsafe user workspace id from {user_id!r}")
    return text


def path_inside(root: Path, candidate: Path) -> bool:
    """True when ``candidate`` resolves to a path at or under ``root``.

    Symlinks are followed via ``resolve()``, so a link planted inside the root that points
    outside is still denied.
    """
    try:
        root_resolved = root.resolve()
        resolved = candidate.resolve()
    except OSError:
        return False
    if resolved == root_resolved:
        return True
    try:
        resolved.relative_to(root_resolved)
        return True
    except ValueError:
        return False


def resolve_under_root(root: Path, raw_path: str | Path) -> Path | None:
    """Resolve ``raw_path`` relative to ``root`` when it is not absolute.

    Returns the absolute path only if it lies inside ``root``; otherwise ``None``.
    """
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    if not path_inside(root, path):
        return None
    return path.resolve()
