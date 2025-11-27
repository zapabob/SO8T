#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline dataset inventory & quality audit.

Collects sample counts, thinking-format coverage, label quality, and diversity
metrics for the datasets used in the SO8T pipeline, then caches the results to
D:/webdataset/quality_cache and emits a Markdown summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

THINKING_TAGS = (
    "<think-task>",
    "<think-safety>",
    "<think-policy>",
    "<think-final>",
    "<final>",
)

FOUR_CLASS_LABELS = {"ALLOW", "ESCALATION", "DENY", "REFUSE"}


@dataclass
class DatasetSpec:
    name: str
    path: Path
    kind: str  # deep_research | pairwise | sft
    min_samples: int


DATASETS: Tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="deep_research_raw",
        path=Path(r"D:/webdataset/aegis_v2.0/deep_research_thinking_dataset.jsonl"),
        kind="deep_research",
        min_samples=5000,
    ),
    DatasetSpec(
        name="deep_research_cleansed",
        path=Path(r"D:/webdataset/aegis_v2.0/deep_research_thinking_dataset_cleansed.jsonl"),
        kind="deep_research",
        min_samples=5000,
    ),
    DatasetSpec(
        name="pairwise_dataset",
        path=Path(r"D:/webdataset/aegis_v2.0/pairwise_dataset.jsonl"),
        kind="pairwise",
        min_samples=50000,
    ),
    DatasetSpec(
        name="thinking_sft_dataset",
        path=Path(r"D:/webdataset/processed/thinking_sft/thinking_sft_dataset.jsonl"),
        kind="sft",
        min_samples=50000,
    ),
)


def compute_logic_consistency(text: str) -> bool:
    """Check if all required thinking tags exist in the text."""
    if not text:
        return False
    text_lower = text.lower()
    return all(tag.lower() in text_lower for tag in THINKING_TAGS[:-1]) and THINKING_TAGS[-1] in text_lower


def choose_text_fields(sample: Dict[str, Any], spec: DatasetSpec) -> Tuple[str, str]:
    """Return (prompt_like, response_like) strings per dataset kind."""
    if spec.kind == "pairwise":
        return sample.get("prompt", ""), sample.get("chosen", "")
    if spec.kind == "sft":
        # Instruction/input behave like prompts; output is the response.
        prompt = sample.get("instruction", "") + ("\n" + sample.get("input", "") if sample.get("input") else "")
        return prompt, sample.get("output", "")
    # Deep-research datasets usually have prompt/response style keys.
    prompt = sample.get("prompt", "") or sample.get("instruction", "")
    response = sample.get("response", "") or sample.get("output", "")
    return prompt, response


def detect_nsfw_fields(sample: Dict[str, Any]) -> bool:
    """Return True if any NSFW/safety related annotation exists."""
    candidates = ("nsfw_flag", "nsfw", "nsfw_score", "safety_label", "safety_score")
    return any(key in sample for key in candidates)


def finalize_metrics(metrics: Dict[str, Any], spec: DatasetSpec) -> Dict[str, Any]:
    """Normalize metric types and derive aggregate stats."""
    sample_count = metrics["sample_count"] or 1
    metrics["meets_sample_target"] = metrics["sample_count"] >= spec.min_samples
    metrics["logic_consistency_pct"] = round(metrics["logic_consistency_hits"] / sample_count * 100, 2)
    metrics["nsfw_field_coverage_pct"] = round(metrics["nsfw_field_hits"] / sample_count * 100, 2)
    metrics["diversity_ratio"] = round(
        (len(metrics["unique_prompt_hashes"]) if isinstance(metrics["unique_prompt_hashes"], set) else metrics["unique_prompt_hashes"])
        / sample_count,
        4,
    )
    metrics["avg_prompt_chars"] = round(metrics["prompt_chars"] / sample_count, 2)
    metrics["avg_response_chars"] = round(metrics["response_chars"] / sample_count, 2)
    metrics["four_class_distribution"] = dict(metrics["four_class_distribution"])
    metrics["domain_distribution"] = dict(metrics["domain_distribution"])

    if metrics["quality_scores"]:
        metrics["quality_score_avg"] = round(statistics.mean(metrics["quality_scores"]), 4)
        metrics["quality_score_min"] = min(metrics["quality_scores"])
        metrics["quality_score_max"] = max(metrics["quality_scores"])
    else:
        metrics["quality_score_avg"] = None

    if not metrics["meets_sample_target"]:
        metrics["issues"].append("below_target_samples")
    if metrics["logic_consistency_pct"] < 95.0:
        metrics["issues"].append("logic_consistency_drop")
    if metrics["nsfw_field_coverage_pct"] < 90.0:
        metrics["issues"].append("nsfw_coverage_low")
    if metrics["label_issues"] > 0:
        metrics["issues"].append("label_inconsistencies")

    metrics["issues"] = sorted(set(metrics["issues"]))
    metrics["unique_prompt_hashes"] = (
        len(metrics["unique_prompt_hashes"]) if isinstance(metrics["unique_prompt_hashes"], set) else metrics["unique_prompt_hashes"]
    )
    metrics["quality_scores"] = []  # avoid bloating cache
    return metrics


def evaluate_dataset(spec: DatasetSpec) -> Dict[str, Any]:
    """Compute quality metrics for a dataset."""
    metrics: Dict[str, Any] = {
        "name": spec.name,
        "path": str(spec.path),
        "kind": spec.kind,
        "exists": spec.path.exists(),
        "file_size_bytes": spec.path.stat().st_size if spec.path.exists() else 0,
        "sample_count": 0,
        "meets_sample_target": False,
        "logic_consistency_hits": 0,
        "label_issues": 0,
        "four_class_distribution": Counter(),
        "domain_distribution": Counter(),
        "nsfw_field_hits": 0,
        "prompt_chars": 0,
        "response_chars": 0,
        "unique_prompt_hashes": set(),
        "quality_scores": [],
        "issues": [],
    }

    if not spec.path.exists():
        metrics["issues"].append("missing_file")
        return finalize_metrics(metrics, spec)

    if metrics["file_size_bytes"] == 0:
        metrics["issues"].append("empty_file")
        return finalize_metrics(metrics, spec)

    with spec.path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc=f"[{spec.name}]", unit="lines"):
            line = line.strip()
            if not line:
                continue
            metrics["sample_count"] += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                metrics["issues"].append("json_decode_error")
                continue

            prompt_text, response_text = choose_text_fields(sample, spec)
            metrics["prompt_chars"] += len(prompt_text)
            metrics["response_chars"] += len(response_text)

            if response_text and compute_logic_consistency(response_text):
                metrics["logic_consistency_hits"] += 1

            if detect_nsfw_fields(sample):
                metrics["nsfw_field_hits"] += 1

            prompt_hash = hash(prompt_text.strip())
            metrics["unique_prompt_hashes"].add(prompt_hash)

            if spec.kind == "pairwise":
                label = sample.get("four_class_label", "").upper()
                if label in FOUR_CLASS_LABELS:
                    metrics["four_class_distribution"][label] += 1
                else:
                    metrics["label_issues"] += 1
                try:
                    score = float(sample.get("quality_score", 0.0))
                    metrics["quality_scores"].append(score)
                except (TypeError, ValueError):
                    metrics["label_issues"] += 1
            elif spec.kind == "sft":
                label = sample.get("four_class_label", "").upper()
                if label and label in FOUR_CLASS_LABELS:
                    metrics["four_class_distribution"][label] += 1
                elif label:
                    metrics["label_issues"] += 1
                domain = sample.get("domain_label", "unknown")
                metrics["domain_distribution"][domain] += 1
                safety_label = sample.get("safety_label", "").upper()
                if safety_label and safety_label not in ("ALLOW", "ESCALATION", "DENY", "REFUSE"):
                    metrics["label_issues"] += 1

    return finalize_metrics(metrics, spec)


def detect_worktree_name() -> str:
    """Return git worktree name or 'main'."""
    try:
        git_dir = subprocess.check_output(["git", "rev-parse", "--git-dir"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "main"

    path = Path(git_dir)
    if "worktrees" in path.parts:
        parts = list(path.parts)
        idx = parts.index("worktrees")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "main"


def build_markdown_report(results: List[Dict[str, Any]], report_path: Path) -> None:
    """Generate markdown summary with metrics per dataset."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Dataset Quality Audit ({now})",
        "",
        "| Dataset | Samples | Target OK | Logic % | NSFW % | Diversity | Avg Prompt | Avg Response | Issues |",
        "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in results:
        lines.append(
            "| {name} | {sample_count:,} | {target} | {logic:.2f}% | {nsfw:.2f}% | {diversity:.4f} | "
            "{prompt:.1f} | {response:.1f} | {issues} |".format(
                name=entry["name"],
                sample_count=entry["sample_count"],
                target="[OK]" if entry["meets_sample_target"] else "[NG]",
                logic=entry["logic_consistency_pct"],
                nsfw=entry["nsfw_field_coverage_pct"],
                diversity=entry["diversity_ratio"],
                prompt=entry["avg_prompt_chars"],
                response=entry["avg_response_chars"],
                issues=", ".join(entry["issues"]) if entry["issues"] else "-",
            )
        )

    lines.append("")
    lines.append("## Detailed Notes")
    for entry in results:
        lines.append(f"### {entry['name']}")
        lines.append(f"- Path: `{entry['path']}`")
        lines.append(f"- File size: {entry['file_size_bytes']:,} bytes")
        lines.append(f"- Unique prompts ratio: {entry['diversity_ratio']:.4f}")
        if entry["four_class_distribution"]:
            lines.append(f"- Four-class distribution: {entry['four_class_distribution']}")
        if entry["domain_distribution"]:
            lines.append(f"- Domain distribution (top 5): {dict(list(entry['domain_distribution'].items())[:5])}")
        if entry.get("quality_score_avg") is not None:
            lines.append(
                f"- Quality score avg: {entry['quality_score_avg']:.3f} "
                f"[{entry['quality_score_min']:.2f}, {entry['quality_score_max']:.2f}]"
            )
        lines.append(f"- Issues: {entry['issues'] or 'None'}")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SO8T pipeline dataset audit")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(r"D:/webdataset/quality_cache"),
        help="Directory to store cached JSON metrics",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="_docs markdown path; inferred when omitted",
    )
    args = parser.parse_args()

    results = [evaluate_dataset(spec) for spec in DATASETS]

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_path = args.cache_dir / f"dataset_quality_{timestamp}.json"
    cache_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    worktree_name = detect_worktree_name()
    if args.report_path is None:
        report_path = Path("_docs") / f"{datetime.now().strftime('%Y-%m-%d')}_{worktree_name}_dataset_quality_audit.md"
    else:
        report_path = args.report_path

    build_markdown_report(results, report_path)
    print(f"[OK] Cached metrics -> {cache_path}")
    print(f"[OK] Markdown report -> {report_path}")


if __name__ == "__main__":
    main()

