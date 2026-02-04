#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic skill dispatcher (Codex integration).
Uses subagent registry to route tasks based on capabilities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from subagents import SubagentManager, Task


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_manager() -> SubagentManager:
    manager = SubagentManager(
        definitions_dir=PROJECT_ROOT / "subagents" / "definitions",
        config_path=PROJECT_ROOT / "config" / "subagents.yaml",
    )
    manager.load()
    return manager


def dispatch_task(description: str, routing_strategy: str, tags: List[str], capabilities: List[str]) -> Dict[str, Any]:
    manager = load_manager()
    task = Task(
        description=description,
        routing_strategy=routing_strategy,
        required_capabilities=capabilities,
        tags=tags,
    )
    decision = manager.route(task)
    return {
        "description": description,
        "routing": {
            "strategy": decision.strategy,
            "assignments": [
                {
                    "subagent_name": a.subagent_name,
                    "task_portion": a.task_portion,
                    "capabilities": a.capabilities,
                    "configuration": a.configuration,
                }
                for a in decision.assignments
            ],
            "reasoning": decision.reasoning,
        },
    }


def execute_workflow(tasks_path: Path) -> List[Dict[str, Any]]:
    manager = load_manager()
    tasks_payload = json.loads(tasks_path.read_text(encoding="utf-8")) if tasks_path.suffix == ".json" else None
    schedule: List[Dict[str, Any]] = []

    if tasks_payload is None:
        import yaml
        tasks_payload = yaml.safe_load(tasks_path.read_text(encoding="utf-8")) or {}

    for task_entry in tasks_payload.get("tasks", []):
        task = Task(
            description=task_entry.get("description", ""),
            routing_strategy=task_entry.get("routing_strategy", "single_best"),
            required_capabilities=task_entry.get("required_capabilities", []) or [],
            tags=task_entry.get("tags", []) or [],
        )
        decision = manager.route(task)
        schedule.append(
            {
                "id": task_entry.get("id"),
                "description": task.description,
                "routing": {
                    "strategy": decision.strategy,
                    "assignments": [
                        {
                            "subagent_name": a.subagent_name,
                            "task_portion": a.task_portion,
                            "capabilities": a.capabilities,
                            "configuration": a.configuration,
                        }
                        for a in decision.assignments
                    ],
                    "reasoning": decision.reasoning,
                },
            }
        )
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic skill dispatcher")
    subparsers = parser.add_subparsers(dest="command")

    dispatch_parser = subparsers.add_parser("dispatch-task")
    dispatch_parser.add_argument("--task", required=True)
    dispatch_parser.add_argument("--routing-strategy", default="single_best", choices=["single_best", "parallel", "sequential"])
    dispatch_parser.add_argument("--tags", nargs="*", default=[])
    dispatch_parser.add_argument("--capabilities", nargs="*", default=[])

    workflow_parser = subparsers.add_parser("execute-workflow")
    workflow_parser.add_argument("--workflow", default="config/subagent_tasks.yaml")

    args = parser.parse_args()

    if args.command == "dispatch-task":
        result = dispatch_task(args.task, args.routing_strategy, args.tags, args.capabilities)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "execute-workflow":
        schedule = execute_workflow(Path(args.workflow))
        output_path = PROJECT_ROOT / "results" / "dynamic_dispatch_schedule.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(schedule, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(output_path))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
