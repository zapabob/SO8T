"""Validation utilities for subagent definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .definitions import SubagentDefinition


@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[str]
    recommendations: List[str]


class SubagentValidator:
    def validate_subagent_definition(self, definition: SubagentDefinition) -> ValidationResult:
        issues: List[str] = []

        if not definition.name:
            issues.append("Missing subagent name")
        if not definition.personality:
            issues.append("Missing personality definition")
        if not definition.capabilities:
            issues.append("No capabilities defined")

        for capability in definition.capabilities:
            if not capability.name:
                issues.append("Capability missing name")
            if not capability.description:
                issues.append(f"Capability missing description: {capability.name or 'unknown'}")

        config_issues = self._validate_configuration_consistency(definition)
        issues.extend(config_issues)

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            recommendations=self._generate_recommendations(issues),
        )

    def _validate_configuration_consistency(self, definition: SubagentDefinition) -> List[str]:
        issues: List[str] = []
        project_overrides = definition.configuration.project_overrides
        env_overrides = definition.configuration.environment_overrides
        for env in env_overrides:
            if env in project_overrides:
                issues.append(f"Conflicting configuration for environment: {env}")
        return issues

    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        recommendations: List[str] = []
        if "Missing personality definition" in issues:
            recommendations.append("Add personality role/expertise to improve routing")
        if "No capabilities defined" in issues:
            recommendations.append("Define at least one capability for this subagent")
        return recommendations
