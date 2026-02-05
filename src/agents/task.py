"""Task models and routing artifacts for subagents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Task:
    description: str
    routing_strategy: str = "single_best"
    required_capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubagentMatch:
    subagent_name: str
    score: float
    capabilities: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubagentAssignment:
    subagent_name: str
    task_portion: float
    capabilities: List[str]
    configuration: Dict[str, Any]


@dataclass
class RoutingDecision:
    strategy: str
    assignments: List[SubagentAssignment]
    reasoning: str

    @staticmethod
    def fallback_routing(task: Task) -> "RoutingDecision":
        return RoutingDecision(
            strategy="fallback",
            assignments=[],
            reasoning=f"No matching subagents found for task: {task.description}",
        )
