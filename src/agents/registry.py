"""Subagent registry and matching logic."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .definitions import SubagentDefinition
from .task import SubagentMatch, Task


class SubagentRegistry:
    def __init__(self) -> None:
        self.subagents: Dict[str, SubagentDefinition] = {}
        self.capability_index: Dict[str, List[str]] = {}

    def register_subagent(self, subagent_def: SubagentDefinition) -> bool:
        if not subagent_def.name:
            return False
        if subagent_def.name in self.subagents:
            return False

        self.subagents[subagent_def.name] = subagent_def

        for capability in subagent_def.capabilities:
            if capability.name not in self.capability_index:
                self.capability_index[capability.name] = []
            self.capability_index[capability.name].append(subagent_def.name)

        return True

    def load_from_file(self, path: Path) -> Optional[SubagentDefinition]:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        subagent_def = SubagentDefinition.from_dict(data)
        if self.register_subagent(subagent_def):
            return subagent_def
        return None

    def load_from_directory(self, directory: Path) -> List[SubagentDefinition]:
        loaded = []
        if not directory.exists():
            return loaded
        for file_path in directory.glob("*.y*ml"):
            subagent_def = self.load_from_file(file_path)
            if subagent_def:
                loaded.append(subagent_def)
        return loaded

    def find_subagents_for_task(self, task: Task) -> List[SubagentMatch]:
        matches: List[SubagentMatch] = []
        for subagent_name, subagent in self.subagents.items():
            score = self._calculate_match_score(task, subagent)
            if score > 0:
                matches.append(
                    SubagentMatch(
                        subagent_name=subagent_name,
                        score=score,
                        capabilities=[cap.name for cap in subagent.capabilities],
                        configuration=self._get_relevant_config(task, subagent),
                    )
                )
        return sorted(matches, key=lambda match: match.score, reverse=True)

    def _calculate_match_score(self, task: Task, subagent: SubagentDefinition) -> float:
        score = 0.0
        description = task.description.lower()

        if task.required_capabilities:
            for capability in task.required_capabilities:
                if capability in self.capability_index and subagent.name in self.capability_index[capability]:
                    score += 2.0

        if subagent.personality and subagent.personality.expertise:
            for expertise in subagent.personality.expertise:
                if expertise.lower() in description:
                    score += 1.0

        for trigger in subagent.triggers:
            if trigger.pattern:
                try:
                    if re.search(trigger.pattern, task.description, flags=re.IGNORECASE):
                        score += 1.5
                except re.error:
                    if trigger.pattern.lower() in description:
                        score += 1.0

        if task.tags:
            for tag in task.tags:
                if subagent.personality and any(tag.lower() in exp.lower() for exp in subagent.personality.expertise):
                    score += 0.5

        return score

    def _get_relevant_config(self, task: Task, subagent: SubagentDefinition) -> Dict:
        config = dict(subagent.configuration.project_overrides)
        config.update({"task_tags": task.tags})
        return config
