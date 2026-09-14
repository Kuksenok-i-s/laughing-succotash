"""Per-user ACP filesystem sandbox."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.agent.acp_client import AcpClient
from agent_core.agent.fs_sandbox import path_inside, resolve_under_root, telegram_dir_id
from agent_core.config import Settings


def test_telegram_dir_id_strips_namespace() -> None:
    assert telegram_dir_id("tg:8375266535") == "8375266535"
    assert telegram_dir_id("tg:1") == "1"


@pytest.mark.parametrize(
    "bad",
    ["", "tg:", "tg:abc", "tg:12/34", "tg:..", "tg:1..2", "../etc", "tg:1:2"],
)
def test_telegram_dir_id_rejects_unsafe(bad: str) -> None:
    with pytest.raises(ValueError):
        telegram_dir_id(bad)


def test_user_workspace_creates_per_user_dir(settings: Settings) -> None:
    a = settings.user_workspace("tg:111")
    b = settings.user_workspace("tg:222")
    assert a.name == "user_111"
    assert b.name == "user_222"
    assert a.is_dir() and b.is_dir()
    assert a != b
    assert a.parent == settings.resolved_data_dir


def test_path_inside_and_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "user_1"
    root.mkdir()
    (root / "ok.txt").write_text("hi", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    link = root / "escape"
    link.symlink_to(outside)

    assert path_inside(root, root / "ok.txt")
    assert not path_inside(root, outside)
    assert not path_inside(root, link)
    assert resolve_under_root(root, "ok.txt") == (root / "ok.txt").resolve()
    assert resolve_under_root(root, str(outside)) is None
    assert resolve_under_root(root, "escape") is None


def test_acp_fs_allows_only_bound_session_root(tmp_path: Path) -> None:
    user_a = tmp_path / "user_1"
    user_b = tmp_path / "user_2"
    user_a.mkdir()
    user_b.mkdir()
    (user_a / "note.txt").write_text("alpha", encoding="utf-8")
    (user_b / "note.txt").write_text("beta", encoding="utf-8")

    client = AcpClient(cwd=tmp_path)
    client.bind_session_root("sess-a", user_a)
    client.bind_session_root("sess-b", user_b)

    assert client._read_file(
        {"sessionId": "sess-a", "path": str(user_a / "note.txt")}
    ) == {"content": "alpha"}
    assert client._read_file(
        {"sessionId": "sess-a", "path": str(user_b / "note.txt")}
    ) == {"content": ""}
    assert client._read_file({"sessionId": "sess-a", "path": "note.txt"}) == {
        "content": "alpha"
    }
    assert client._read_file({"path": str(user_a / "note.txt")}) == {"content": ""}

    client._write_file(
        {
            "sessionId": "sess-a",
            "path": str(user_a / "pic.txt"),
            "content": "saved",
        }
    )
    assert (user_a / "pic.txt").read_text(encoding="utf-8") == "saved"

    client._write_file(
        {
            "sessionId": "sess-a",
            "path": str(user_b / "stolen.txt"),
            "content": "nope",
        }
    )
    assert not (user_b / "stolen.txt").exists()

    repo = tmp_path / "tg_bot_kirpich" / "prompts.py"
    repo.parent.mkdir()
    repo.write_text("original", encoding="utf-8")
    client._write_file(
        {"sessionId": "sess-a", "path": str(repo), "content": "hacked"}
    )
    assert repo.read_text(encoding="utf-8") == "original"
