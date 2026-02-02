# Subagent registry and loader.
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .schema import (
    SubagentCapability,
    SubagentConfiguration,
    SubagentDefinition,
    SubagentPersonality,
    SubagentTriggers,
)


class SubagentRegistry:
    def __init__(self):
        self.subagents: Dict[str, SubagentDefinition] = {}
        self.capability_index: Dict[str, List[str]] = {}

    def register_subagent(self, subagent_def: SubagentDefinition) -> bool:
        if subagent_def.name in self.subagents:
            return False
        self.subagents[subagent_def.name] = subagent_def
        for capability in subagent_def.capabilities:
            self.capability_index.setdefault(capability.name, []).append(subagent_def.name)
        return True

    def list_subagents(self) -> List[str]:
        return sorted(self.subagents.keys())

    def get(self, name: str) -> Optional[SubagentDefinition]:
        return self.subagents.get(name)

    def find_by_capability(self, capability: str) -> List[str]:
        return self.capability_index.get(capability, [])


def _parse_personality(data: Dict) -> SubagentPersonality:
    return SubagentPersonality(
        role=data.get("role", ""),
        expertise=data.get("expertise", []) or [],
        communication_style=data.get("communication_style", ""),
        risk_tolerance=data.get("risk_tolerance", ""),
    )


def _parse_capabilities(data: List[Dict]) -> List[SubagentCapability]:
    capabilities = []
    for item in data or []:
        capabilities.append(
            SubagentCapability(
                name=item.get("name", ""),
                description=item.get("description", ""),
                tools=item.get("tools", []) or [],
                permissions=item.get("permissions", []) or [],
            )
        )
    return capabilities


def _parse_configuration(data: Dict, environment: Optional[str]) -> SubagentConfiguration:
    return SubagentConfiguration(
        project_overrides=data.get("project_overrides", {}) or {},
        environment_overrides=data.get("environment_overrides", {}) or {},
        user_overrides=data.get("user_overrides", {}) or {},
        environment=environment,
    )


def _parse_triggers(data: List[Dict]) -> List[SubagentTriggers]:
    triggers = []
    for item in data or []:
        triggers.append(
            SubagentTriggers(
                pattern=item.get("pattern", ""),
                confidence_threshold=float(item.get("confidence_threshold", 0.0)),
                priority=item.get("priority", ""),
            )
        )
    return triggers


def load_subagent_definition(path: Path, environment: Optional[str] = None) -> SubagentDefinition:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not payload or "subagent" not in payload:
        raise ValueError(f"Invalid subagent definition: {path}")
    data = payload["subagent"]
    personality = _parse_personality(data.get("personality", {}))
    capabilities = _parse_capabilities(data.get("capabilities", []))
    configuration = _parse_configuration(data.get("configuration", {}), environment)
    triggers = _parse_triggers(data.get("triggers", []))
    return SubagentDefinition(
        name=data.get("name", ""),
        version=data.get("version", ""),
        personality=personality,
        capabilities=capabilities,
        configuration=configuration,
        triggers=triggers,
    )


def load_registry(registry_path: Path) -> SubagentRegistry:
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not payload or "subagents" not in payload:
        raise ValueError(f"Invalid registry file: {registry_path}")
    environment = payload.get("defaults", {}).get("environment")
    registry = SubagentRegistry()
    base_dir = registry_path.parent
    for entry in payload.get("subagents", []):
        rel_path = entry.get("file")
        if not rel_path:
            continue
        agent_path = (base_dir / rel_path).resolve()
        subagent_def = load_subagent_definition(agent_path, environment=environment)
        registry.register_subagent(subagent_def)
    return registry


def registry_to_dict(registry: SubagentRegistry) -> Dict[str, Dict]:
    return {name: asdict(defn) for name, defn in registry.subagents.items()}
