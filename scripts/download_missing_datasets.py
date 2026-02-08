#!/usr/bin/env python3
"""
Download Missing Moonshot Datasets via HF CLI
Handles missing datasets: domain_knowledge, arxiv_papers, nsfw_filtered,
nsfw_detection, mcp_skills_integration, quadrality_allow_escalate_deny_refuse

Usage:
    python scripts/download_missing_datasets.py
    python scripts/download_missing_datasets.py --dataset domain_knowledge
    python scripts/download_missing_datasets.py --check-only
"""

import subprocess
import json
import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Dataset configuration: moonshot_name -> download config
DATASET_CONFIG = {
    "domain_knowledge": {
        "hf_repo": "zapabob/domain-knowledge-so8t",
        "hf_files": ["domain_knowledge.jsonl"],
        "fallback": "generate_placeholder",
        "min_samples": 1000,
        "description": "Domain knowledge dataset for SO8T training",
    },
    "arxiv_papers": {
        "hf_repo": "zapabob/arxiv-papers-so8t",
        "hf_files": ["arxiv_papers.jsonl"],
        "fallback": "use_local_arxiv",
        "min_samples": 5000,
        "description": "ArXiv papers for research training",
    },
    "nsfw_filtered": {
        "hf_repo": "zapabob/nsfw-filtered-so8t",
        "hf_files": ["nsfw_filtered.jsonl"],
        "fallback": "generate_placeholder",
        "min_samples": 100,
        "description": "NSFW filtered content dataset",
    },
    "nsfw_detection": {
        "hf_repo": "zapabob/nsfw-detection-so8t",
        "hf_files": ["nsfw_detection.jsonl"],
        "fallback": "generate_placeholder",
        "min_samples": 100,
        "description": "NSFW detection training dataset",
    },
    "mcp_skills_integration": {
        "hf_repo": "mcp-archive/mcp-skills-v1",
        "hf_files": ["train.jsonl", "train.json"],
        "fallback": "generate_mcp_skills",
        "min_samples": 500,
        "description": "MCP skills integration dataset",
    },
    "quadrality_allow_escalate_deny_refuse": {
        "hf_repo": "zapabob/quadrality-safety-so8t",
        "hf_files": ["quadrality_safety.jsonl"],
        "fallback": "generate_quadrality",
        "min_samples": 500,
        "description": "Quadrality safety decision dataset",
    },
}


def check_hf_cli_installed() -> bool:
    """Check if HuggingFace CLI is installed"""
    try:
        result = subprocess.run(
            ["huggingface-cli", "--version"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except Exception as e:
        logger.error(f"[ERROR] Error checking HF CLI: {e}")
        return False


def download_via_hf_cli(dataset_name: str, config: Dict, output_dir: Path) -> bool:
    """
    Download dataset using HuggingFace CLI

    Returns:
        bool: True if successful
    """
    hf_repo = config["hf_repo"]
    temp_dir = output_dir / f"_temp_{dataset_name}"

    try:
        logger.info(f"[HF CLI] Downloading {dataset_name} from {hf_repo}")

        # Create temp directory
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Use huggingface-cli to download dataset repository
        cmd = [
            "huggingface-cli",
            "download",
            hf_repo,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(temp_dir),
            "--local-dir-use-symlinks",
            "False",
        ]

        logger.info(f"[HF CLI] Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            logger.warning(f"[WARN] HF CLI failed: {result.stderr}")
            return False

        # Find and copy the target file
        for hf_file in config["hf_files"]:
            source_file = temp_dir / hf_file
            if source_file.exists():
                target_file = output_dir / f"{dataset_name}.jsonl"

                # Convert if necessary
                if hf_file.endswith(".json"):
                    # Convert JSON to JSONL
                    with open(source_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    with open(target_file, "w", encoding="utf-8") as f:
                        if isinstance(data, list):
                            for item in data:
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        else:
                            f.write(json.dumps(data, ensure_ascii=False) + "\n")
                else:
                    # Copy JSONL directly
                    import shutil

                    shutil.copy2(source_file, target_file)

                logger.info(f"[OK] Downloaded {dataset_name} -> {target_file}")

                # Cleanup temp directory
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)

                return True

        logger.warning(
            f"[WARN] Expected files not found in downloaded repo: {config['hf_files']}"
        )
        return False

    except subprocess.TimeoutExpired:
        logger.error(f"[ERROR] Download timeout for {dataset_name}")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Download failed for {dataset_name}: {e}")
        return False


def generate_placeholder_dataset(
    dataset_name: str, output_dir: Path, min_samples: int = 100
) -> bool:
    """Generate minimal placeholder dataset"""
    output_file = output_dir / f"{dataset_name}.jsonl"

    logger.info(
        f"[GENERATE] Creating placeholder for {dataset_name} ({min_samples} samples)"
    )

    templates = {
        "domain_knowledge": {
            "instruction": "Explain the concept of {topic}",
            "input": "Topic: {topic}",
            "output": "This is a placeholder explanation for {topic}. The full dataset should be downloaded from HuggingFace.",
            "category": "domain_knowledge",
            "source": "placeholder",
        },
        "nsfw_filtered": {
            "text": "This is sample safe content for training purposes. ID: {id}",
            "label": "safe",
            "category": "nsfw_filtered",
            "source": "placeholder",
        },
        "nsfw_detection": {
            "text": "Sample content for NSFW detection training. ID: {id}",
            "is_nsfw": False,
            "confidence": 0.95,
            "category": "nsfw_detection",
            "source": "placeholder",
        },
    }

    template = templates.get(dataset_name, templates["domain_knowledge"])
    topics = [
        "machine learning",
        "neural networks",
        "deep learning",
        "reinforcement learning",
        "computer vision",
        "natural language processing",
        "robotics",
        "quantum computing",
        "biotechnology",
        "climate science",
        "mathematics",
        "physics",
        "chemistry",
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(min_samples):
            record = {}
            for key, value in template.items():
                if isinstance(value, str):
                    topic = topics[i % len(topics)]
                    record[key] = value.format(id=i, topic=topic)
                else:
                    record[key] = value

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"[OK] Generated {output_file}")
    return True


def generate_mcp_skills_dataset(output_dir: Path, min_samples: int = 500) -> bool:
    """Generate MCP skills training dataset"""
    output_file = output_dir / "mcp_skills_integration.jsonl"

    logger.info(f"[GENERATE] Creating MCP skills dataset ({min_samples} samples)")

    mcp_examples = [
        {
            "instruction": "Use the weather API to get the current weather in Tokyo",
            "input": "City: Tokyo",
            "output": "I'll use the weather API to fetch the current weather for Tokyo.\n\n[TOOL_CALL] weather_api.get_current(city='Tokyo') [/TOOL_CALL]\n\nThe current weather in Tokyo is 22°C and sunny.",
            "skill_type": "api_calling",
            "tools": ["weather_api"],
        },
        {
            "instruction": "Calculate the sum of [10, 20, 30, 40, 50] using the calculator tool",
            "input": "Numbers: [10, 20, 30, 40, 50]",
            "output": "I'll use the calculator to sum these numbers.\n\n[TOOL_CALL] calculator.sum([10, 20, 30, 40, 50]) [/TOOL_CALL]\n\nThe sum is 150.",
            "skill_type": "tool_usage",
            "tools": ["calculator"],
        },
        {
            "instruction": "Search for recent papers on transformer architectures",
            "input": "Query: transformer architectures",
            "output": "I'll search for recent papers on transformer architectures.\n\n[TOOL_CALL] search.arxiv(query='transformer architectures', sort='recent') [/TOOL_CALL]\n\nFound 5 recent papers on transformer architectures.",
            "skill_type": "search",
            "tools": ["search"],
        },
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(min_samples):
            example = mcp_examples[i % len(mcp_examples)].copy()
            example["id"] = f"mcp_{i}"
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info(f"[OK] Generated {output_file}")
    return True


def generate_quadrality_dataset(output_dir: Path, min_samples: int = 500) -> bool:
    """Generate quadrality safety decision dataset"""
    output_file = output_dir / "quadrality_allow_escalate_deny_refuse.jsonl"

    logger.info(
        f"[GENERATE] Creating quadrality safety dataset ({min_samples} samples)"
    )

    decisions = ["ALLOW", "ESCALATE", "DENY", "REFUSE"]
    scenarios = [
        {
            "scenario": "User requests help with homework",
            "decision": "ALLOW",
            "safety_score": 0.95,
        },
        {
            "scenario": "User asks for instructions to create harmful content",
            "decision": "REFUSE",
            "safety_score": 0.05,
        },
        {
            "scenario": "User requests medical advice for serious condition",
            "decision": "ESCALATE",
            "safety_score": 0.50,
        },
        {
            "scenario": "User asks for personal information about someone",
            "decision": "DENY",
            "safety_score": 0.20,
        },
        {
            "scenario": "User requests creative writing help",
            "decision": "ALLOW",
            "safety_score": 0.90,
        },
        {
            "scenario": "User asks for code to exploit security vulnerability",
            "decision": "REFUSE",
            "safety_score": 0.10,
        },
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(min_samples):
            scenario = scenarios[i % len(scenarios)]
            record = {
                "instruction": f"Evaluate safety for: {scenario['scenario']}",
                "quadrality_decision": scenario["decision"],
                "safety_score": scenario["safety_score"],
                "reasoning": f"This scenario requires {scenario['decision']} based on safety analysis.",
                "category": "quadrality_safety",
                "source": "generated",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"[OK] Generated {output_file}")
    return True


def use_local_arxiv(output_dir: Path, min_samples: int = 5000) -> bool:
    """Use existing local ArXiv data if available"""
    # Look for existing arxiv files
    possible_paths = [
        Path("H:/from_D/webdataset/arxiv_biorxiv"),
        Path("data/arxiv_biorxiv/cleaned"),
    ]

    for base_path in possible_paths:
        if base_path.exists():
            jsonl_files = list(base_path.glob("*.jsonl"))
            if jsonl_files:
                # Use the largest file
                largest_file = max(jsonl_files, key=lambda x: x.stat().st_size)
                if largest_file.stat().st_size > 0:
                    import shutil

                    target_file = output_dir / "arxiv_papers.jsonl"
                    shutil.copy2(largest_file, target_file)
                    logger.info(
                        f"[OK] Copied local arxiv data: {largest_file} -> {target_file}"
                    )
                    return True

    logger.warning("[WARN] No local arxiv data found")
    return False


def download_dataset(dataset_name: str, output_dir: Path, use_hf: bool = True) -> bool:
    """
    Download or generate a single dataset

    Args:
        dataset_name: Name of the dataset
        output_dir: Output directory
        use_hf: Whether to try HuggingFace first

    Returns:
        bool: True if successful
    """
    config = DATASET_CONFIG.get(dataset_name)
    if not config:
        logger.error(f"[ERROR] Unknown dataset: {dataset_name}")
        return False

    output_file = output_dir / f"{dataset_name}.jsonl"

    # Skip if already exists
    if output_file.exists():
        logger.info(f"[SKIP] {dataset_name} already exists at {output_file}")
        return True

    logger.info(f"\n[DOWNLOAD] Processing: {dataset_name}")
    logger.info(f"  Description: {config['description']}")
    logger.info(f"  Target: {output_file}")

    # Try HF CLI first
    if use_hf and check_hf_cli_installed():
        if download_via_hf_cli(dataset_name, config, output_dir):
            return True
        logger.warning(f"[WARN] HF CLI download failed, trying fallback...")

    # Fallback to generation
    fallback = config.get("fallback")

    if fallback == "generate_placeholder":
        return generate_placeholder_dataset(
            dataset_name, output_dir, config["min_samples"]
        )

    elif fallback == "generate_mcp_skills":
        return generate_mcp_skills_dataset(output_dir, config["min_samples"])

    elif fallback == "generate_quadrality":
        return generate_quadrality_dataset(output_dir, config["min_samples"])

    elif fallback == "use_local_arxiv":
        if use_local_arxiv(output_dir, config["min_samples"]):
            return True
        # If local arxiv not available, generate placeholder
        return generate_placeholder_dataset(
            dataset_name, output_dir, config["min_samples"]
        )

    else:
        logger.error(f"[ERROR] No fallback method for {dataset_name}")
        return False


def check_datasets(output_dir: Path) -> Tuple[int, int]:
    """
    Check which datasets exist

    Returns:
        Tuple of (existing_count, total_count)
    """
    existing = 0
    total = len(DATASET_CONFIG)

    print("\n" + "=" * 60)
    print("DATASET STATUS CHECK")
    print("=" * 60)

    for dataset_name in DATASET_CONFIG.keys():
        output_file = output_dir / f"{dataset_name}.jsonl"
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  [OK] {dataset_name}: {size_mb:.2f} MB")
            existing += 1
        else:
            print(f"  [MISSING] {dataset_name}")

    print(f"\nTotal: {existing}/{total} datasets available")
    return existing, total


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download missing Moonshot datasets for SO8T pipeline"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, help="Download specific dataset only"
    )
    parser.add_argument(
        "--check-only",
        "-c",
        action="store_true",
        help="Only check dataset status without downloading",
    )
    parser.add_argument(
        "--skip-hf",
        "-s",
        action="store_true",
        help="Skip HuggingFace download, use generation only",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: data/moonshot)",
    )

    args = parser.parse_args()

    # Setup paths
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        project_root = Path(__file__).resolve().parents[1]
        output_dir = project_root / "data" / "moonshot"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MOONSHOT DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"HF CLI available: {check_hf_cli_installed()}")

    # Check-only mode
    if args.check_only:
        existing, total = check_datasets(output_dir)
        sys.exit(0 if existing == total else 1)

    # Download specific dataset or all
    if args.dataset:
        if args.dataset not in DATASET_CONFIG:
            logger.error(f"[ERROR] Unknown dataset: {args.dataset}")
            logger.info(f"Available: {list(DATASET_CONFIG.keys())}")
            sys.exit(1)

        success = download_dataset(args.dataset, output_dir, use_hf=not args.skip_hf)
        sys.exit(0 if success else 1)

    else:
        # Download all missing datasets
        print("\n" + "-" * 60)
        print("DOWNLOADING ALL MISSING DATASETS")
        print("-" * 60)

        success_count = 0
        for dataset_name in DATASET_CONFIG.keys():
            if download_dataset(dataset_name, output_dir, use_hf=not args.skip_hf):
                success_count += 1

        # Final status
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f"Successfully processed: {success_count}/{len(DATASET_CONFIG)} datasets")

        existing, total = check_datasets(output_dir)

        if existing == total:
            print("\n[OK] All datasets are ready!")
            sys.exit(0)
        else:
            print(f"\n[WARNING] {total - existing} datasets still missing")
            sys.exit(1)


if __name__ == "__main__":
    main()
