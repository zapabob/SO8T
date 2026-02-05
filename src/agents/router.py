"""Dynamic routing for subagent tasks."""
from __future__ import annotations

from datetime import datetime
from typing import List

from .task import RoutingDecision, SubagentAssignment, Task
from .registry import SubagentRegistry


class DynamicTaskRouter:
    def __init__(self, registry: SubagentRegistry) -> None:
        self.registry = registry
        self.routing_history = []

    def route_task(self, task: Task) -> RoutingDecision:
        candidates = self.registry.find_subagents_for_task(task)
        if not candidates:
            return RoutingDecision.fallback_routing(task)

        if task.routing_strategy == "single_best":
            decision = self.route_to_single_best(candidates)
        elif task.routing_strategy == "parallel":
            decision = self.route_parallel(candidates)
        elif task.routing_strategy == "sequential":
            decision = self.route_sequential(candidates)
        else:
            decision = self.route_to_single_best(candidates)

        self.routing_history.append(
            {
                "task": task,
                "candidates": candidates,
                "decision": decision,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return decision

    def route_to_single_best(self, candidates) -> RoutingDecision:
        best_match = candidates[0]
        return RoutingDecision(
            strategy="single",
            assignments=[
                SubagentAssignment(
                    subagent_name=best_match.subagent_name,
                    task_portion=1.0,
                    capabilities=best_match.capabilities,
                    configuration=best_match.configuration,
                )
            ],
            reasoning=f"Selected {best_match.subagent_name} with score {best_match.score:.2f}",
        )

    def route_parallel(self, candidates) -> RoutingDecision:
        parallel_candidates = candidates[:3] if len(candidates) >= 2 else candidates
        if len(parallel_candidates) < 2:
            return self.route_to_single_best(candidates)

        total_score = sum(candidate.score for candidate in parallel_candidates)
        assignments: List[SubagentAssignment] = []
        for candidate in parallel_candidates:
            portion = candidate.score / total_score if total_score else 1 / len(parallel_candidates)
            assignments.append(
                SubagentAssignment(
                    subagent_name=candidate.subagent_name,
                    task_portion=portion,
                    capabilities=candidate.capabilities,
                    configuration=candidate.configuration,
                )
            )

        return RoutingDecision(
            strategy="parallel",
            assignments=assignments,
            reasoning=f"Parallel execution across {len(assignments)} subagents",
        )

    def route_sequential(self, candidates) -> RoutingDecision:
        assignments: List[SubagentAssignment] = []
        for candidate in candidates:
            assignments.append(
                SubagentAssignment(
                    subagent_name=candidate.subagent_name,
                    task_portion=0.0,
                    capabilities=candidate.capabilities,
                    configuration=candidate.configuration,
                )
            )

        return RoutingDecision(
            strategy="sequential",
            assignments=assignments,
            reasoning="Sequential execution for all matched subagents",
        )
