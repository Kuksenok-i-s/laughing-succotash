"""The permission model is the security boundary; these are its contract tests."""

from __future__ import annotations

import pytest

from agent_core.agent.base import Provenance
from agent_core.mcp import permissions
from agent_core.mcp.permissions import Decision, Tier, ToolContext, decide, tier_of


def ctx(provenance: Provenance) -> ToolContext:
    return ToolContext(user_id="tg:1", conversation_id="conv", provenance=provenance)


DIRECT = ctx(Provenance.DIRECT_COMMAND)
TRANSCRIPT = ctx(Provenance.UNTRUSTED_CONTENT)


@pytest.mark.parametrize(
    "tool",
    ["calendar_list", "task_list", "note_search", "memory_search", "contact_search",
     "system_status", "web_search", "calendar_find_free_slots"],
)
def test_read_tools_run_without_asking(tool: str) -> None:
    assert decide(tool, DIRECT) is Decision.ALLOW
    # Reading stays automatic even while analysing a recording: it has no side effects.
    assert decide(tool, TRANSCRIPT) is Decision.ALLOW


@pytest.mark.parametrize(
    "tool", ["reminder_create", "task_create", "note_create", "calendar_create"]
)
def test_safe_writes_depend_on_who_asked(tool: str) -> None:
    assert decide(tool, DIRECT) is Decision.ALLOW
    # The same call inferred from a recording becomes a proposal.
    assert decide(tool, TRANSCRIPT) is Decision.CONFIRM


@pytest.mark.parametrize(
    "tool", ["calendar_delete", "task_delete", "note_delete", "memory_forget"]
)
def test_dangerous_tools_always_confirm(tool: str) -> None:
    assert decide(tool, DIRECT) is Decision.CONFIRM
    assert decide(tool, TRANSCRIPT) is Decision.CONFIRM


def test_unknown_tool_fails_closed() -> None:
    assert tier_of("shell") is Tier.DANGEROUS
    assert decide("some_new_tool", DIRECT) is Decision.CONFIRM


def test_every_registered_tool_has_a_tier() -> None:
    """A capability added without a tier would silently become DANGEROUS at runtime."""
    from agent_core.mcp.server import ToolRegistry
    from agent_core.mcp.tools import register_tools

    registry = ToolRegistry()
    register_tools(registry, _NullRepos(), calendar_provider=None, search_provider=_NullSearch())

    unclassified = [name for name in registry.names() if name not in permissions.TIERS]
    assert unclassified == []


def test_confirmation_prompt_shows_the_actual_arguments() -> None:
    text = permissions.describe_action(
        "calendar_create",
        {"title": "Созвон", "starts_at": "2026-08-24T15:00:00", "ends_at": "2026-08-24T16:00:00"},
    )
    assert "Созвон" in text
    assert "24.08 15:00" in text
    assert "24.08 16:00" in text


def test_confirmation_prompt_for_unknown_tool_is_still_specific() -> None:
    assert "weird_tool" in permissions.describe_action("weird_tool", {})


class _NullSearch:
    async def search(self, query, limit):
        return []

    async def fetch(self, url):
        return {}


class _NullRepos:
    """register_tools only stores references; nothing is called during registration."""

    def __getattr__(self, _name):
        return None
