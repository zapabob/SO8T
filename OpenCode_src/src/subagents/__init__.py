" Subagent system registry and routing utilities.
from .schema import (
    SubagentDefinition,
    SubagentCapability,
    SubagentPersonality,
    SubagentConfiguration,
)
from .registry import SubagentRegistry
from .router import DynamicTaskRouter
from .permissions import collect_permissions, satisfies_permissions
