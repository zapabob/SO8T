"""SO8T subagent system."""
from .definitions import Capability, Personality, SubagentConfiguration, SubagentDefinition, Trigger
from .manager import SubagentManager
from .registry import SubagentRegistry
from .router import DynamicTaskRouter
from .task import RoutingDecision, SubagentAssignment, SubagentMatch, Task
from .validator import SubagentValidator, ValidationResult

__all__ = [
    "Capability",
    "Personality",
    "SubagentConfiguration",
    "SubagentDefinition",
    "Trigger",
    "SubagentManager",
    "SubagentRegistry",
    "DynamicTaskRouter",
    "RoutingDecision",
    "SubagentAssignment",
    "SubagentMatch",
    "Task",
    "SubagentValidator",
    "ValidationResult",
]
