# Subagent permission helpers.
from __future__ import annotations

from typing import Iterable, Set

from .schema import SubagentDefinition


def collect_permissions(subagent: SubagentDefinition) -> Set[str]:
    permissions: Set[str] = set()
    for capability in subagent.capabilities:
        permissions.update(capability.permissions)
    return permissions


def satisfies_permissions(subagent: SubagentDefinition, required: Iterable[str]) -> bool:
    if not required:
        return True
    permissions = collect_permissions(subagent)
    return all(req in permissions for req in required)
