"""Subagent definitions and data models for SO8T."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Personality:
    role: str
    expertise: List[str]
    communication_style: str
    risk_tolerance: str


@dataclass
class Capability:
    name: str
    description: str
    tools: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)


@dataclass
class SubagentConfiguration:
    project_overrides: Dict[str, Any] = field(default_factory=dict)
    environment_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Trigger:
    pattern: Optional[str] = None
    file_pattern: Optional[str] = None
    action: Optional[str] = None
    confidence_threshold: float = 0.0
    priority: str = "medium"


@dataclass
class SubagentDefinition:
    name: str
    version: str = "1.0.0"
    personality: Personality = None
    capabilities: List[Capability] = field(default_factory=list)
    configuration: SubagentConfiguration = field(default_factory=SubagentConfiguration)
    triggers: List[Trigger] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubagentDefinition":
        payload = data.get("subagent", data)

        personality = payload.get("personality") or {}
        personality_obj = Personality(
            role=personality.get("role", ""),
            expertise=personality.get("expertise", []) or [],
            communication_style=personality.get("communication_style", ""),
            risk_tolerance=personality.get("risk_tolerance", "medium"),
        )

        capabilities = []
        for item in payload.get("capabilities", []) or []:
            capabilities.append(
                Capability(
                    name=item.get("name", ""),
                    description=item.get("description", ""),
                    tools=item.get("tools", []) or [],
                    permissions=item.get("permissions", []) or [],
                )
            )

        configuration = payload.get("configuration") or {}
        config_obj = SubagentConfiguration(
            project_overrides=configuration.get("project_overrides", {}) or {},
            environment_overrides=configuration.get("environment_overrides", {}) or {},
        )

        triggers = []
        for item in payload.get("triggers", []) or []:
            triggers.append(
                Trigger(
                    pattern=item.get("pattern"),
                    file_pattern=item.get("file_pattern"),
                    action=item.get("action"),
                    confidence_threshold=float(item.get("confidence_threshold", 0.0) or 0.0),
                    priority=item.get("priority", "medium"),
                )
            )

        return cls(
            name=payload.get("name", ""),
            version=payload.get("version", "1.0.0"),
            personality=personality_obj,
            capabilities=capabilities,
            configuration=config_obj,
            triggers=triggers,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subagent": {
                "name": self.name,
                "version": self.version,
                "personality": {
                    "role": self.personality.role,
                    "expertise": self.personality.expertise,
                    "communication_style": self.personality.communication_style,
                    "risk_tolerance": self.personality.risk_tolerance,
                },
                "capabilities": [
                    {
                        "name": capability.name,
                        "description": capability.description,
                        "tools": capability.tools,
                        "permissions": capability.permissions,
                    }
                    for capability in self.capabilities
                ],
                "configuration": {
                    "project_overrides": self.configuration.project_overrides,
                    "environment_overrides": self.configuration.environment_overrides,
                },
                "triggers": [
                    {
                        "pattern": trigger.pattern,
                        "file_pattern": trigger.file_pattern,
                        "action": trigger.action,
                        "confidence_threshold": trigger.confidence_threshold,
                        "priority": trigger.priority,
                    }
                    for trigger in self.triggers
                ],
            }
        }
