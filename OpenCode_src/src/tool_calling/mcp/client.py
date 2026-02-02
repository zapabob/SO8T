"""Minimal MCP client stub for registering and calling tools."""
from __future__ import annotations

from typing import Callable, Dict

from .protocol import ToolCall, ToolResult, ToolSpec


class MCPClient:
    """In-process MCP client stub."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable] = {}
        self._specs: Dict[str, ToolSpec] = {}

    def register_tool(self, spec: ToolSpec, handler: Callable) -> None:
        self._specs[spec.name] = spec
        self._tools[spec.name] = handler

    def list_tools(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def call_tool(self, call: ToolCall) -> ToolResult:
        if call.name not in self._tools:
            return ToolResult(name=call.name, output={"error": "tool_not_found"}, success=False)
        handler = self._tools[call.name]
        result = handler(**call.arguments)
        return ToolResult(name=call.name, output={"result": result}, success=True)
