" Dataclasses for subagent definitions.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SubagentPersonality:
    role: str
    expertise: List[str]
    communication_style: str
    risk_tolerance: str


@dataclass
class SubagentCapability:
    name: str
    description: str
    tools: List[str]
    permissions: List[str]


@dataclass
class SubagentConfiguration:
    project_overrides: Dict[str, object] = field(default_factory=dict)
    environment_overrides: Dict[str, Dict[str, object]] = field(default_factory=dict)
    user_overrides: Dict[str, object] = field(default_factory=dict)
    environment: Optional[str] = None


@dataclass
class SubagentTriggers:
    pattern: str
    confidence_threshold: float
    priority: str


@dataclass
class SubagentDefinition:
    name: str
    version: str
    personality: SubagentPersonality
    capabilities: List[SubagentCapability]
    configuration: SubagentConfiguration
    triggers: List[SubagentTriggers]
