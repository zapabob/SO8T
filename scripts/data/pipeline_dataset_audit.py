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
import os
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from huggingface_hub import HfApi, HfFileSystem
import pandas as pd

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
    parser.add_argument(
        "--huggingface-audit",
        action="store_true",
        help="Audit cloned HuggingFace datasets instead of SO8T pipeline datasets",
    )
    args = parser.parse_args()

    if args.huggingface_audit:
        # Audit HuggingFace datasets
        worktree_name = detect_worktree_name()
        if args.report_path is None:
            report_path = Path("_docs") / f"{datetime.now().strftime('%Y-%m-%d')}_{worktree_name}_huggingface_dataset_audit.md"
        else:
            report_path = args.report_path

        # Get datasets dynamically
        hf_datasets = get_huggingface_datasets()
        results = audit_huggingface_datasets(hf_datasets, report_path)
        print(f"[OK] HuggingFace dataset audit completed -> {report_path}")
    else:
        # Audit SO8T pipeline datasets
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


@dataclass
class HuggingFaceDatasetSpec:
    repo_id: str
    local_path: Path
    expected_license: str
    primary_use: str  # coding, nsfw, reasoning, etc.
    modality: str  # text, multimodal, etc.
    language: str  # en, ja, multi


def get_huggingface_datasets() -> Tuple[HuggingFaceDatasetSpec, ...]:
    """Get HuggingFace dataset specs with paths relative to the script location."""
    # Determine base path - try to find datasets relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # scripts/data -> scripts -> project_root

    # Possible dataset locations
    possible_bases = [
        project_root / ".." / ".." / "webdataset" / "datasets",  # Relative to project
        Path(r"D:\webdataset\datasets"),  # Absolute D: drive
        Path("./datasets"),  # Relative to cwd
        script_dir.parent.parent / "datasets",  # Direct relative
    ]

    datasets_base = None
    for base in possible_bases:
        if (base / "ehartford_wizard_vicuna_70k_unfiltered").exists():
            datasets_base = base
            break

    if datasets_base is None:
        # Fallback to the original absolute path
        datasets_base = Path(r"D:\webdataset\datasets")

    return (
        HuggingFaceDatasetSpec(
            repo_id="FreedomIntelligence/alpaca-gpt4-japanese",
            local_path=datasets_base / "FreedomIntelligence_alpaca_gpt4_japanese",
            expected_license="Apache-2.0",
            primary_use="reasoning",
            modality="text",
            language="ja"
        ),
        HuggingFaceDatasetSpec(
            repo_id="FreedomIntelligence/sharegpt-japanese",
            local_path=datasets_base / "FreedomIntelligence_sharegpt_japanese",
            expected_license="Apache-2.0",
            primary_use="reasoning",
            modality="text",
            language="ja"
        ),
        HuggingFaceDatasetSpec(
            repo_id="FreedomIntelligence/MMLU_Japanese",
            local_path=datasets_base / "FreedomIntelligence_MMLU_Japanese",
            expected_license="MIT",
            primary_use="reasoning",
            modality="text",
            language="ja"
        ),
        HuggingFaceDatasetSpec(
            repo_id="shi3z/anthropic_hh_rlhf_japanese",
            local_path=datasets_base / "shi3z_anthropic_hh_rlhf_japanese",
            expected_license="MIT",
            primary_use="safety",
            modality="text",
            language="ja"
        ),
        HuggingFaceDatasetSpec(
            repo_id="fujiki/japanese_hh-rlhf-49k",
            local_path=datasets_base / "fujiki_japanese_hh-rlhf_49k",
            expected_license="MIT",
            primary_use="safety",
            modality="text",
            language="ja"
        ),
        HuggingFaceDatasetSpec(
            repo_id="nomic-ai/gpt4all-j-prompt-generations",
            local_path=datasets_base / "nomic-ai_gpt4all-j-prompt-generations",
            expected_license="Apache-2.0",
            primary_use="coding",
            modality="text",
            language="en"
        ),
        HuggingFaceDatasetSpec(
            repo_id="teknium/GPTeacher-General-Instruct",
            local_path=datasets_base / "teknium_GPTeacher-General-Instruct",
            expected_license="MIT",
            primary_use="coding",
            modality="text",
            language="en"
        ),
        HuggingFaceDatasetSpec(
            repo_id="ehartford/wizard_vicuna_70k_unfiltered",
            local_path=datasets_base / "ehartford_wizard_vicuna_70k_unfiltered",
            expected_license="Apache-2.0",
            primary_use="reasoning",
            modality="text",
            language="en"
        ),
        HuggingFaceDatasetSpec(
            repo_id="OpenAssistant/oasst2",
            local_path=datasets_base / "OpenAssistant_oasst2",
            expected_license="Apache-2.0",
            primary_use="coding",
            modality="text",
            language="multi"
        ),
        HuggingFaceDatasetSpec(
            repo_id="open-orca/OpenOrca",
            local_path=datasets_base / "open-orca_OpenOrca",
            expected_license="MIT",
            primary_use="reasoning",
            modality="text",
            language="en"
        ),
        HuggingFaceDatasetSpec(
            repo_id="Elizezen/japanese-nsfw-syosetsu-dataset",
            local_path=datasets_base / "Elizezen_japanese-nsfw-syosetsu-dataset",
            expected_license="Apache-2.0",
            primary_use="nsfw",
            modality="text",
            language="ja"
        ),
        HuggingFaceDatasetSpec(
            repo_id="eliasalbouzidi/NSFW-Safe-Dataset",
            local_path=datasets_base / "eliasalbouzidi_NSFW-Safe-Dataset",
            expected_license="Apache-2.0",
            primary_use="nsfw",
            modality="text",
            language="en"
        ),
        HuggingFaceDatasetSpec(
            repo_id="RyokoExtra/JapaneseGoblin",
            local_path=datasets_base / "RyokoExtra_JapaneseGoblin",
            expected_license="Apache-2.0",
            primary_use="reasoning",
            modality="text",
            language="ja"
        ),
    )


def evaluate_huggingface_dataset(spec: HuggingFaceDatasetSpec) -> Dict[str, Any]:
    """Evaluate a cloned HuggingFace dataset for content analysis."""
    # Resolve to absolute path and check existence
    resolved_path = spec.local_path.resolve()

    metrics = {
        "repo_id": spec.repo_id,
        "local_path": str(resolved_path),
        "expected_license": spec.expected_license,
        "primary_use": spec.primary_use,
        "modality": spec.modality,
        "language": spec.language,
        "exists": resolved_path.exists(),
        "sample_count": 0,
        "nsfw_count": 0,
        "coding_count": 0,
        "math_science_count": 0,
        "business_mcp_count": 0,
        "total_chars": 0,
        "issues": []
    }

    if not resolved_path.exists():
        metrics["issues"].append("dataset_not_found")
        return metrics

    # Use resolved path for all operations
    spec.local_path = resolved_path

    # Define keyword patterns for content analysis
    nsfw_keywords = [
        "nsfw", "adult", "erotic", "porn", "sex", "nude", "naked", "hentai",
        "18+", "xxx", "fetish", "bdsm", "incest", "rape", "violence", "gore"
    ]

    coding_keywords = [
        "python", "javascript", "java", "cpp", "c++", "rust", "go", "typescript",
        "html", "css", "sql", "git", "api", "function", "class", "import",
        "def ", "const ", "let ", "var ", "function ", "class ", "interface",
        "docker", "kubernetes", "aws", "azure", "gcp", "linux", "windows"
    ]

    math_science_keywords = [
        "theorem", "proof", "equation", "formula", "algorithm", "hypothesis",
        "quantum", "physics", "chemistry", "biology", "mathematics", "calculus",
        "integral", "derivative", "matrix", "vector", "probability", "statistics"
    ]

    business_mcp_keywords = [
        "business", "strategy", "management", "finance", "marketing", "sales",
        "project", "planning", "budget", "report", "meeting", "presentation",
        "email", "communication", "leadership", "team", "client", "customer",
        "mcp", "tool", "api", "integration", "workflow", "automation"
    ]

    # Scan all text files in the dataset
    text_files = []
    if spec.local_path.joinpath("README.md").exists():
        text_files.append(spec.local_path / "README.md")

    # Find data files (JSON, JSONL, TXT, etc.)
    # Prioritize README.md first, then data files
    readme_path = spec.local_path / "README.md"
    if readme_path.exists():
        text_files.append(readme_path)

    for ext in [".json", ".jsonl", ".txt", ".csv"]:
        text_files.extend(spec.local_path.glob(f"**/*{ext}"))

    for file_path in text_files:
        try:
            if file_path.suffix in [".json", ".jsonl"]:
                # Try to read as JSONL first (line-delimited JSON)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                item = json.loads(line)
                                _analyze_sample_content(item, metrics, nsfw_keywords, coding_keywords,
                                                      math_science_keywords, business_mcp_keywords)
                            except json.JSONDecodeError:
                                # If line-by-line parsing fails, try reading whole file as single JSON
                                break
                        else:
                            # Successfully processed all lines as JSONL
                            continue

                    # If we broke out of the loop, try reading as single JSON object
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                _analyze_sample_content(item, metrics, nsfw_keywords, coding_keywords,
                                                      math_science_keywords, business_mcp_keywords)
                        elif isinstance(data, dict):
                            _analyze_sample_content(data, metrics, nsfw_keywords, coding_keywords,
                                                  math_science_keywords, business_mcp_keywords)

                except json.JSONDecodeError as e:
                    metrics["issues"].append(f"json_parse_error_{file_path.name}: {str(e)}")
                except UnicodeDecodeError as e:
                    metrics["issues"].append(f"unicode_error_{file_path.name}: {str(e)}")

            elif file_path.suffix in [".txt", ".md"]:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    _analyze_text_content(content, metrics, nsfw_keywords, coding_keywords,
                                        math_science_keywords, business_mcp_keywords)
            else:
                # Skip other file types for now
                continue

        except Exception as e:
            metrics["issues"].append(f"error_reading_{file_path.name}: {str(e)}")

    return metrics


def _analyze_sample_content(sample: Dict[str, Any], metrics: Dict[str, Any],
                          nsfw_keywords: List[str], coding_keywords: List[str],
                          math_science_keywords: List[str], business_mcp_keywords: List[str]):
    """Analyze a single data sample for content categories."""
    if not isinstance(sample, dict):
        return

    metrics["sample_count"] += 1
    text_content = ""

    # Extract text from common fields
    for field in ["text", "content", "instruction", "input", "output", "response", "prompt"]:
        if field in sample and isinstance(sample[field], str):
            text_content += " " + sample[field]

    _analyze_text_content(text_content, metrics, nsfw_keywords, coding_keywords,
                         math_science_keywords, business_mcp_keywords)


def _analyze_text_content(text: str, metrics: Dict[str, Any],
                        nsfw_keywords: List[str], coding_keywords: List[str],
                        math_science_keywords: List[str], business_mcp_keywords: List[str]):
    """Analyze text content for keyword matches."""
    if not text:
        return

    text_lower = text.lower()
    metrics["total_chars"] += len(text)

    # Count keyword matches
    metrics["nsfw_count"] += sum(1 for kw in nsfw_keywords if kw in text_lower)
    metrics["coding_count"] += sum(1 for kw in coding_keywords if kw in text_lower)
    metrics["math_science_count"] += sum(1 for kw in math_science_keywords if kw in text_lower)
    metrics["business_mcp_count"] += sum(1 for kw in business_mcp_keywords if kw in text_lower)


def audit_huggingface_datasets(datasets: Tuple[HuggingFaceDatasetSpec, ...], output_path: Path) -> List[Dict[str, Any]]:
    """Audit all HuggingFace datasets and generate report."""
    print("[INFO] Auditing HuggingFace datasets...")

    results = []
    for spec in tqdm(datasets, desc="Auditing datasets"):
        result = evaluate_huggingface_dataset(spec)
        results.append(result)

    # Generate markdown report
    report_lines = [
        "# HuggingFace Dataset Audit Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Total datasets: {len(results)}",
        f"- Existing datasets: {sum(1 for r in results if r['exists'])}",
        f"- Missing datasets: {sum(1 for r in results if not r['exists'])}",
        "",
        "## Dataset Details",
        "",
        "| Dataset | Exists | Samples | NSFW | Coding | Math/Sci | Business | Total Chars | Issues |",
        "|---------|--------|---------|-------|--------|----------|----------|-------------|--------|"
    ]

    for result in results:
        exists_mark = "✅" if result["exists"] else "❌"
        issues_str = ", ".join(result["issues"]) if result["issues"] else "None"
        report_lines.append(
            f"| {result['repo_id']} | {exists_mark} | {result['sample_count']} | "
            f"{result['nsfw_count']} | {result['coding_count']} | {result['math_science_count']} | "
            f"{result['business_mcp_count']} | {result['total_chars']} | {issues_str} |"
        )

    report_lines.extend([
        "",
        "## Content Analysis Summary",
        "",
        "### Primary Use Distribution",
    ])

    # Group by primary use
    use_counts = {}
    for result in results:
        if result["exists"]:
            use = result["primary_use"]
            if use not in use_counts:
                use_counts[use] = {"count": 0, "nsfw": 0, "coding": 0, "math_sci": 0, "business": 0}
            use_counts[use]["count"] += 1
            use_counts[use]["nsfw"] += result["nsfw_count"]
            use_counts[use]["coding"] += result["coding_count"]
            use_counts[use]["math_sci"] += result["math_science_count"]
            use_counts[use]["business"] += result["business_mcp_count"]

    for use, stats in use_counts.items():
        report_lines.extend([
            f"#### {use.title()} Datasets ({stats['count']} datasets)",
            f"- NSFW keywords: {stats['nsfw']}",
            f"- Coding keywords: {stats['coding']}",
            f"- Math/Science keywords: {stats['math_sci']}",
            f"- Business/MCP keywords: {stats['business']}",
            ""
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[OK] HuggingFace dataset audit report -> {output_path}")
    return results


if __name__ == "__main__":
    main()

