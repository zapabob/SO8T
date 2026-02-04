"""Subagent manager orchestrating registry and routing."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .registry import SubagentRegistry
from .router import DynamicTaskRouter
from .task import RoutingDecision, Task


class SubagentManager:
    def __init__(self, definitions_dir: Path, config_path: Optional[Path] = None) -> None:
        self.definitions_dir = definitions_dir
        self.config_path = config_path
        self.registry = SubagentRegistry()
        self.router = DynamicTaskRouter(self.registry)
        self.project_config: Dict[str, Any] = {}

    def load(self) -> None:
        self.registry.load_from_directory(self.definitions_dir)
        if self.config_path and self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as handle:
                self.project_config = yaml.safe_load(handle) or {}

    def get_active_subagents(self) -> List[str]:
        enabled = self.project_config.get("enable_subagents")
        if enabled:
            return enabled
        return list(self.registry.subagents.keys())

    def route(self, task: Task) -> RoutingDecision:
        return self.router.route_task(task)

    def resolve_environment_config(self, env: str) -> Dict[str, Any]:
        env_overrides = self.project_config.get("environment_overrides", {})
        return env_overrides.get(env, {})
