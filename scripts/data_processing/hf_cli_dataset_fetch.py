#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF CLI Dataset Fetcher for Moonshot v3.0

- Uses `hf download <dataset_id> --repo-type dataset`
- Stores each dataset under base_dir/<dataset_id_safe>
- Writes a manifest JSON with download results
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hf_cli_dataset_fetch.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


HF_DATASETS: List[Dict[str, str]] = [
    {"id": "armanc/scientific_papers", "category": "science_papers"},
    {"id": "irds/cord19", "category": "science_covid"},
    {"id": "pritamdeka/cord-19-fulltext", "category": "science_covid_fulltext"},
    {"id": "ccdv/arxiv-summarization", "category": "science"},
    {"id": "sentence-transformers/s2orc", "category": "science"},
    {"id": "ncbi/pubmed", "category": "science"},
    {"id": "WINGNUS/ACL-OCL", "category": "science_nlp"},
    {"id": "dwb2023/gdelt-event-2025-v3", "category": "osint_events"},
    {"id": "wikimedia/wikipedia", "category": "osint_knowledge"},
    {"id": "Alphonse7/Wikidata5M-KG", "category": "osint_knowledge_graph"},
    {"id": "andreas-helgesson/gdelt-big", "category": "osint_events"},
    {"id": "togethercomputer/RedPajama-Data-1T", "category": "openweb"},
    {"id": "togethercomputer/RedPajama-Data-1T-Sample", "category": "openweb_sample"},
    {"id": "Geralt-Targaryen/openwebtext2", "category": "openweb"},
    {"id": "oscar-corpus/OSCAR-2201", "category": "openweb_multilingual"},
    {"id": "openai/gsm8k", "category": "math_cot"},
    {"id": "nvidia/OpenMath-GSM8K-masked", "category": "math_cot"},
    {"id": "fever/fever", "category": "fact_check"},
    {"id": "GioApc/promptoxicity", "category": "nsfw_toxicity"},
    {"id": "google/jigsaw_toxicity_pred", "category": "nsfw_toxicity"},
    {"id": "ibm-research/Toucan", "category": "mcp_tool_calling"},
    {"id": "AymanTarig/function-calling-v0.2-with-r1-cot", "category": "cot_tool_calling"},
    {"id": "deepseek-ai/DeepSeek-V3", "category": "think_reference", "optional": True},
    {"id": "SidhiPanda/peS2o", "category": "science_large", "optional": True},
]


def safe_dirname(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def hf_cli_available() -> bool:
    return shutil.which("hf") is not None


def download_dataset(dataset_id: str, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["hf", "download", dataset_id, "--repo-type", "dataset", "--local-dir", str(out_dir)]
    logger.info(f"[HF] Download: {dataset_id} -> {out_dir}")
    result = {"dataset_id": dataset_id, "local_dir": str(out_dir), "status": "ok"}
    try:
        subprocess.run(cmd, check=True)
        result["size_bytes"] = dir_size_bytes(out_dir)
    except subprocess.CalledProcessError as e:
        result["status"] = "failed"
        result["error"] = str(e)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="HF CLI dataset fetcher")
    parser.add_argument("--base-dir", type=str, required=True, help="Base directory to store datasets")
    parser.add_argument("--manifest", type=str, required=True, help="Output manifest JSON path")
    parser.add_argument("--max-datasets", type=int, default=0, help="Limit number of datasets (0 = all)")
    args = parser.parse_args()

    if not hf_cli_available():
        logger.error("hf CLI not found. Please install huggingface_hub CLI (hf).")
        return 1

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    include_large = os.getenv("SO8T_HF_INCLUDE_LARGE") == "1"
    filtered = [d for d in HF_DATASETS if include_large or not d.get("optional")]
    datasets = filtered if args.max_datasets <= 0 else filtered[: args.max_datasets]
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "base_dir": str(base_dir),
        "datasets": [],
    }

    for item in datasets:
        ds_id = item["id"]
        ds_cat = item["category"]
        out_dir = base_dir / safe_dirname(ds_id)
        result = download_dataset(ds_id, out_dir)
        result["category"] = ds_cat
        manifest["datasets"].append(result)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"[HF] Manifest saved: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
