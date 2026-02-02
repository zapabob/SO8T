# Subagent permission helpers.
from __future__ import annotations

from typing import Iterable, Set, Dict, Any, Optional
from pathlib import Path
import os

import yaml

from .schema import SubagentDefinition


_POLICY_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_policy(project_root: Optional[Path] = None) -> Dict[str, Any]:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    policy_env = os.environ.get("SUBAGENT_POLICY_PATH")
    if policy_env:
        policy_path = Path(policy_env)
    else:
        policy_path = project_root / "config" / "subagents" / "policy.yaml"
    cache_key = str(policy_path)
    if cache_key in _POLICY_CACHE:
        return _POLICY_CACHE[cache_key]
    if not policy_path.exists():
        _POLICY_CACHE[cache_key] = {}
        return {}
    payload = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    _POLICY_CACHE[cache_key] = payload
    return payload


def collect_permissions(subagent: SubagentDefinition) -> Set[str]:
    permissions: Set[str] = set()
    for capability in subagent.capabilities:
        permissions.update(capability.permissions)
    return permissions


def effective_permissions(subagent: SubagentDefinition) -> Set[str]:
    permissions = collect_permissions(subagent)
    policy = _load_policy()
    policy_cfg = policy.get("policy", {})

    def apply_scope(perms: Set[str], scope: Dict[str, Any]) -> Set[str]:
        allow = scope.get("allow")
        deny = scope.get("deny")
        if allow and "*" not in allow:
            perms = perms.intersection(set(allow))
        if deny:
            perms = perms.difference(set(deny))
        return perms

    defaults = policy_cfg.get("defaults", {})
    permissions = apply_scope(permissions, defaults)

    env = subagent.configuration.environment
    env_cfg = policy_cfg.get("environments", {}).get(env or "", {})
    if env_cfg:
        permissions = apply_scope(permissions, env_cfg)

    agent_cfg = policy_cfg.get("subagents", {}).get(subagent.name, {})
    if agent_cfg:
        permissions = apply_scope(permissions, agent_cfg)

    return permissions


def satisfies_permissions(subagent: SubagentDefinition, required: Iterable[str]) -> bool:
    if not required:
        return True
    permissions = effective_permissions(subagent)
    return all(req in permissions for req in required)
