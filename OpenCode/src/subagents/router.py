# Dynamic task routing for subagents.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import re

from .schema import SubagentDefinition
from .registry import SubagentRegistry


@dataclass
class SubagentAssignment:
    subagent_name: str
    task_portion: float
    score: float
    capabilities: List[str]


@dataclass
class RoutingDecision:
    strategy: str
    assignments: List[SubagentAssignment]
    reasoning: str


class DynamicTaskRouter:
    def __init__(self, registry: SubagentRegistry):
        self.registry = registry

    def _score_subagent(self, task: str, subagent: SubagentDefinition) -> float:
        score = 0.0
        for trigger in subagent.triggers:
            if not trigger.pattern:
                continue
            if re.search(trigger.pattern, task, re.IGNORECASE):
                score = max(score, float(trigger.confidence_threshold))
        return score

    def route_task(self, task: str, strategy: str = "auto") -> RoutingDecision:
        candidates: List[SubagentAssignment] = []
        for subagent in self.registry.subagents.values():
            score = self._score_subagent(task, subagent)
            if score > 0:
                capabilities = [cap.name for cap in subagent.capabilities]
                candidates.append(SubagentAssignment(
                    subagent_name=subagent.name,
                    task_portion=0.0,
                    score=score,
                    capabilities=capabilities,
                ))

        if not candidates:
            return RoutingDecision(strategy="fallback", assignments=[], reasoning="No matching subagent triggers")

        candidates.sort(key=lambda item: item.score, reverse=True)
        if strategy in ("single", "single_best"):
            best = candidates[0]
            best.task_portion = 1.0
            return RoutingDecision(strategy="single", assignments=[best], reasoning=f"Selected {best.subagent_name}")

        if strategy in ("parallel", "auto"):
            top = candidates[:3]
            total = sum(item.score for item in top) or 1.0
            for item in top:
                item.task_portion = item.score / total
            return RoutingDecision(strategy="parallel", assignments=top, reasoning="Parallel routing to top candidates")

        # default fallback to single
        best = candidates[0]
        best.task_portion = 1.0
        return RoutingDecision(strategy="single", assignments=[best], reasoning=f"Selected {best.subagent_name}")
