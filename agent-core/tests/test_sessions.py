"""SessionManager forces plan mode on Telegram chat Cursor sessions."""

from __future__ import annotations

from agent_core.assistant.sessions import SessionManager
from agent_core.mcp.server import ContextRegistry


async def test_ensure_session_sets_plan_mode(repos, settings, backend) -> None:
    user = await repos.conversations.ensure_user("tg:1")
    conversation = await repos.conversations.create_conversation(user.user_id)
    sessions = SessionManager(
        repos.conversations,
        backend,
        ContextRegistry(),
        None,
        user_workspace=settings.user_workspace,
    )

    record, created = await sessions.ensure_session(
        conversation.conversation_id, user_id=user.user_id
    )
    assert created is True
    assert backend.modes == [(record.external_id, "plan")]

    backend.modes.clear()
    again, created_again = await sessions.ensure_session(
        conversation.conversation_id, user_id=user.user_id
    )
    assert created_again is False
    assert again.external_id == record.external_id
    assert backend.modes == [(record.external_id, "plan")]

    skill = settings.user_workspace(user.user_id) / ".cursor/skills/trainer-journal/SKILL.md"
    assert skill.is_file()
    assert "training_log_save" in skill.read_text(encoding="utf-8")
