#!/usr/bin/env python3
"""CLI for managing SO8T subagents."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from subagents.definitions import Capability, Personality, SubagentDefinition, Trigger
from subagents.manager import SubagentManager
from subagents.task import Task
from subagents.validator import SubagentValidator


DEFAULT_DEFINITIONS_DIR = Path("subagents/definitions")
DEFAULT_CONFIG_PATH = Path("config/subagents.yaml")
DEFAULT_TASKS_PATH = Path("config/subagent_tasks.yaml")


def _load_manager(definitions_dir: Path, config_path: Path) -> SubagentManager:
    manager = SubagentManager(definitions_dir=definitions_dir, config_path=config_path)
    manager.load()
    return manager


def cmd_list(args: argparse.Namespace) -> None:
    manager = _load_manager(args.definitions_dir, args.config_path)
    print("Available subagents:")
    for name in manager.registry.subagents:
        print(f"- {name}")


def cmd_validate(args: argparse.Namespace) -> None:
    manager = _load_manager(args.definitions_dir, args.config_path)
    validator = SubagentValidator()
    all_valid = True
    for subagent in manager.registry.subagents.values():
        result = validator.validate_subagent_definition(subagent)
        status = "OK" if result.is_valid else "ISSUES"
        print(f"[{status}] {subagent.name}")
        if not result.is_valid:
            all_valid = False
            for issue in result.issues:
                print(f"  - {issue}")
            for rec in result.recommendations:
                print(f"  * {rec}")
    if not all_valid:
        raise SystemExit(1)


def _parse_capabilities(values: List[str]) -> List[Capability]:
    capabilities: List[Capability] = []
    for value in values:
        parts = value.split(":", 1)
        name = parts[0].strip()
        description = parts[1].strip() if len(parts) > 1 else ""
        capabilities.append(Capability(name=name, description=description))
    return capabilities


def cmd_create(args: argparse.Namespace) -> None:
    definitions_dir = args.definitions_dir
    definitions_dir.mkdir(parents=True, exist_ok=True)
    target_path = definitions_dir / f"{args.name}.yaml"
    if target_path.exists() and not args.force:
        raise SystemExit(f"Definition already exists: {target_path}")

    personality = Personality(
        role=args.role,
        expertise=[item.strip() for item in args.expertise.split(",") if item.strip()],
        communication_style=args.communication_style,
        risk_tolerance=args.risk_tolerance,
    )
    capabilities = _parse_capabilities(args.capability or [])
    triggers = [
        Trigger(pattern=pattern, confidence_threshold=args.confidence_threshold, priority=args.priority)
        for pattern in (args.trigger_pattern or [])
    ]

    definition = SubagentDefinition(
        name=args.name,
        version=args.version,
        personality=personality,
        capabilities=capabilities,
        triggers=triggers,
    )

    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(definition.to_dict(), handle, sort_keys=False, allow_unicode=True)

    print(f"Created {target_path}")


def cmd_config(args: argparse.Namespace) -> None:
    config_path = args.config_path
    config: Dict = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    if args.context:
        config["context"] = args.context
    if args.worktree:
        config["worktree"] = args.worktree
    if args.enable_subagents:
        config["enable_subagents"] = [item.strip() for item in args.enable_subagents.split(",") if item.strip()]

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Updated {config_path}")


def cmd_env_config(args: argparse.Namespace) -> None:
    config_path = args.config_path
    config: Dict = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    env_overrides = config.get("environment_overrides", {})
    env_overrides.setdefault(args.environment, {})

    if args.restrict_tools is not None:
        env_overrides[args.environment]["restrict_tools"] = [
            item.strip() for item in args.restrict_tools.split(",") if item.strip()
        ]
    if args.response_style:
        env_overrides[args.environment]["response_style"] = args.response_style

    config["environment_overrides"] = env_overrides
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Updated {config_path}")


def cmd_delegate(args: argparse.Namespace) -> None:
    manager = _load_manager(args.definitions_dir, args.config_path)

    required_capabilities = [item.strip() for item in (args.required_capabilities or "").split(",") if item.strip()]
    tags = [item.strip() for item in (args.tags or "").split(",") if item.strip()]

    task = Task(
        description=args.task,
        routing_strategy=args.routing_strategy,
        required_capabilities=required_capabilities,
        tags=tags,
    )

    decision = manager.route(task)
    payload = {
        "strategy": decision.strategy,
        "reasoning": decision.reasoning,
        "assignments": [
            {
                "subagent_name": assignment.subagent_name,
                "task_portion": assignment.task_portion,
                "capabilities": assignment.capabilities,
                "configuration": assignment.configuration,
            }
            for assignment in decision.assignments
        ],
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_schedule(args: argparse.Namespace) -> None:
    manager = _load_manager(args.definitions_dir, args.config_path)
    tasks_path = args.tasks_path or DEFAULT_TASKS_PATH
    if not tasks_path.exists():
        raise SystemExit(f"Tasks file not found: {tasks_path}")

    tasks_payload = yaml.safe_load(tasks_path.read_text(encoding="utf-8")) or {}
    schedule = []
    for task_entry in tasks_payload.get("tasks", []):
        description = task_entry.get("description", "")
        if not description:
            continue
        task = Task(
            description=description,
            routing_strategy=task_entry.get("routing_strategy", "single_best"),
            required_capabilities=task_entry.get("required_capabilities", []) or [],
            tags=task_entry.get("tags", []) or [],
        )
        decision = manager.route(task)
        schedule.append(
            {
                "id": task_entry.get("id"),
                "description": description,
                "routing": {
                    "strategy": decision.strategy,
                    "reasoning": decision.reasoning,
                    "assignments": [
                        {
                            "subagent_name": assignment.subagent_name,
                            "task_portion": assignment.task_portion,
                            "capabilities": assignment.capabilities,
                            "configuration": assignment.configuration,
                        }
                        for assignment in decision.assignments
                    ],
                },
            }
        )

    output_path = args.output or Path("results/subagent_schedule.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schedule, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Schedule written: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SO8T Subagent CLI")
    parser.add_argument("--definitions-dir", type=Path, default=DEFAULT_DEFINITIONS_DIR)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List subagents")
    list_parser.set_defaults(func=cmd_list)

    validate_parser = subparsers.add_parser("validate", help="Validate subagent definitions")
    validate_parser.set_defaults(func=cmd_validate)

    create_parser = subparsers.add_parser("create", help="Create a subagent definition")
    create_parser.add_argument("name")
    create_parser.add_argument("--role", required=True)
    create_parser.add_argument("--expertise", default="")
    create_parser.add_argument("--communication-style", default="neutral")
    create_parser.add_argument("--risk-tolerance", default="medium")
    create_parser.add_argument("--version", default="1.0.0")
    create_parser.add_argument("--capability", action="append", help="Format: name:description")
    create_parser.add_argument("--trigger-pattern", action="append")
    create_parser.add_argument("--confidence-threshold", type=float, default=0.8)
    create_parser.add_argument("--priority", default="medium")
    create_parser.add_argument("--force", action="store_true")
    create_parser.set_defaults(func=cmd_create)

    config_parser = subparsers.add_parser("config", help="Update project config")
    config_parser.add_argument("--context")
    config_parser.add_argument("--worktree")
    config_parser.add_argument("--enable-subagents", help="Comma-separated list")
    config_parser.set_defaults(func=cmd_config)

    env_parser = subparsers.add_parser("env-config", help="Update environment config")
    env_parser.add_argument("environment")
    env_parser.add_argument("--restrict-tools")
    env_parser.add_argument("--response-style")
    env_parser.set_defaults(func=cmd_env_config)

    delegate_parser = subparsers.add_parser("delegate", help="Route a task")
    delegate_parser.add_argument("task")
    delegate_parser.add_argument("--routing-strategy", default="single_best")
    delegate_parser.add_argument("--required-capabilities")
    delegate_parser.add_argument("--tags")
    delegate_parser.set_defaults(func=cmd_delegate)

    schedule_parser = subparsers.add_parser("schedule", help="Generate schedule from task file")
    schedule_parser.add_argument("--tasks-path", type=Path, default=DEFAULT_TASKS_PATH)
    schedule_parser.add_argument("--output", type=Path, help="Output JSON path")
    schedule_parser.set_defaults(func=cmd_schedule)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
