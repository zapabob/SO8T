#!/usr/bin/env python3
"""Delegate a task to configured subagents."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.subagents.registry import load_registry
from src.subagents.router import DynamicTaskRouter


def main():
    parser = argparse.ArgumentParser(description="Delegate task to subagents")
    parser.add_argument("task", help="Task description to route")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "config" / "subagents" / "registry.yaml"))
    parser.add_argument("--strategy", default="auto", choices=["auto","single","single_best","parallel"])
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    registry = load_registry(Path(args.registry))
    router = DynamicTaskRouter(registry)
    decision = router.route_task(args.task, strategy=args.strategy)
    payload = {
        "strategy": decision.strategy,
        "reasoning": decision.reasoning,
        "assignments": [
            {
                "subagent": a.subagent_name,
                "task_portion": a.task_portion,
                "score": a.score,
                "capabilities": a.capabilities,
            }
            for a in decision.assignments
        ],
    }

    logs_dir = PROJECT_ROOT / "logs" / "subagents"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"delegate_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.json"
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("Strategy:", payload["strategy"])
        print("Reasoning:", payload["reasoning"])
        for entry in payload["assignments"]:
            cap_str = ", ".join(entry.get("capabilities") or [])
            print(
                f"- {entry.get('subagent')} ({entry.get('task_portion'):.2f}) :: {cap_str}"
            )
        print("Log:", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
