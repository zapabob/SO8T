#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quadrality reward strategy (Phase 5).
Assigns reward labels based on correctness + tool usage.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "reasoning_mastery": 2.0,
    "skill_usage": 1.0,
    "precision_failure": -1.0,
    "tool_hallucination": -2.0,
}


@dataclass
class RewardResult:
    label: str
    score: float
    reason: str


def load_weights(config_path: Optional[Path]) -> Dict[str, float]:
    if not config_path or not config_path.exists():
        return DEFAULT_WEIGHTS.copy()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return {**DEFAULT_WEIGHTS, **data.get("reward_weights", {})}


def evaluate_reward(entry: Dict[str, object], weights: Dict[str, float]) -> RewardResult:
    meta = entry.get("metadata", {}) or {}
    correct = meta.get("answer_correct")
    tool_used = meta.get("tool_used") or meta.get("tool_name") is not None
    tool_success = meta.get("tool_success", True)

    if correct is None:
        return RewardResult("unscored", 0.0, "missing answer_correct")

    if correct and not tool_used:
        return RewardResult("reasoning_mastery", weights["reasoning_mastery"], "correct without tool")
    if correct and tool_used:
        return RewardResult("skill_usage", weights["skill_usage"], "correct with tool")
    if not correct and tool_used and not tool_success:
        return RewardResult("tool_hallucination", weights["tool_hallucination"], "tool used but incorrect")
    if not correct and tool_used:
        return RewardResult("tool_hallucination", weights["tool_hallucination"], "incorrect with tool")
    return RewardResult("precision_failure", weights["precision_failure"], "incorrect without tool")


def load_jsonl(paths: Iterable[Path]) -> Iterable[Dict[str, object]]:
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def annotate_dataset(inputs: List[Path], output_path: Path, weights: Dict[str, float]) -> Dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in load_jsonl(inputs):
            result = evaluate_reward(entry, weights)
            entry.setdefault("metadata", {})
            entry["metadata"].update({
                "reward_label": result.label,
                "reward_score": result.score,
                "reward_reason": result.reason,
            })
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            counts[result.label] = counts.get(result.label, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Quadrality reward strategy annotator")
    parser.add_argument("--input", nargs="*", required=True, help="Input JSONL files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--config", default="config/reward_strategy.yaml", help="Reward config YAML")
    args = parser.parse_args()

    weights = load_weights(Path(args.config))
    counts = annotate_dataset([Path(p) for p in args.input], Path(args.output), weights)
    logger.info("Reward labels: %s", counts)


if __name__ == "__main__":
    main()
