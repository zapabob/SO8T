"""MCP protocol data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolSpec:
    """Tool specification."""

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """Tool call request."""

    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Tool call result."""

    name: str
    output: Dict[str, Any]
    success: bool = True
