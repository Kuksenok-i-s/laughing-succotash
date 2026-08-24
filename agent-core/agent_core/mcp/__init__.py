"""The personal-assistant MCP server and its permission model."""

from .permissions import Decision, Tier, ToolContext, decide, tier_of
from .server import ContextRegistry, McpServer, ToolRegistry
from .tools import register_tools

__all__ = [
    "ContextRegistry",
    "Decision",
    "McpServer",
    "Tier",
    "ToolContext",
    "ToolRegistry",
    "decide",
    "register_tools",
    "tier_of",
]
