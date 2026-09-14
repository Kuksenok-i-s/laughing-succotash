"""SessionManager forces plan mode on Telegram chat Cursor sessions."""

from __future__ import annotations

import pytest

from agent_core.agent.base import AgentError
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


@pytest.mark.parametrize("resumed", [False, True])
async def test_chat_is_refused_if_protected_mode_fails(repos, settings, backend, resumed):
    user = await repos.conversations.ensure_user("tg:1")
    conversation = await repos.conversations.create_conversation(user.user_id)
    sessions = SessionManager(repos.conversations, backend, ContextRegistry(), None,
                              user_workspace=settings.user_workspace)
    if resumed:
        await sessions.ensure_session(conversation.conversation_id, user_id=user.user_id)

    async def fail(*args):
        raise AgentError("mode unavailable")

    working_set_mode = backend.set_mode
    backend.set_mode = fail
    with pytest.raises(AgentError, match="protected chat mode"):
        await sessions.ensure_session(conversation.conversation_id, user_id=user.user_id)
    backend.set_mode = working_set_mode
    _, created = await sessions.ensure_session(conversation.conversation_id, user_id=user.user_id)
    assert created is True  # initial instructions must be sent after recovery


async def test_chat_requires_backend_mode_support(repos, settings, backend):
    user = await repos.conversations.ensure_user("tg:1")
    conversation = await repos.conversations.create_conversation(user.user_id)
    sessions = SessionManager(repos.conversations, backend, ContextRegistry(), None,
                              user_workspace=settings.user_workspace)
    backend.set_mode = None
    with pytest.raises(AgentError, match="protected chat mode"):
        await sessions.ensure_session(conversation.conversation_id, user_id=user.user_id)


async def test_switching_provider_creates_new_session_without_losing_conversation(repos, settings, backend):
    user = await repos.conversations.ensure_user('tg:1')
    conversation = await repos.conversations.create_conversation(user.user_id)
    old = await repos.conversations.create_session(
        conversation.conversation_id, backend='acp',
        workspace=str(settings.user_workspace(user.user_id)), external_id='old-cursor-id',
    )
    sessions = SessionManager(repos.conversations, backend, ContextRegistry(), None,
                              user_workspace=settings.user_workspace)
    record, created = await sessions.ensure_session(conversation.conversation_id, user_id=user.user_id)
    assert created is True
    assert record.backend == backend.name
    assert record.session_id != old.session_id
    assert record.conversation_id == conversation.conversation_id
