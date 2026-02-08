import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Specialized datasets for safety/policy alignment
SAFETY_DATASETS = [
    {"source": "HF", "id": "NSFW-Image-Recognition/nsfw-labels", "category": "nsfw_visual"},
    {"source": "HF", "id": "drug-abuse-detection/clinical-labeled", "category": "drug_text"},
    {"source": "GH", "id": "NSFW-Detection/awesome-nsfw-lists", "category": "nsfw_agg"},
    {"source": "HF", "id": "Axcxept/toxic-japanese-2025", "category": "toxicity_jp"}
]

class NSFWDrugAggregator:
    """
    Aggregates safety and policy-related datasets for SO8ViT thinking alignment.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def aggregate(self) -> List[Dict[str, Any]]:
        logger.info("Aggregating safety data manifests...")
        results = []
        for ds in tqdm(SAFETY_DATASETS, desc="Safety Datasets"):
            results.append({
                "source": ds["source"],
                "id": ds["id"],
                "category": ds["category"],
                "description": f"Safety alignment data for {ds['category']} detection and reasoning.",
                "importance": "high"
            })
        return results

    def run(self):
        manifest = self.aggregate()
        output_path = self.output_dir / "safety_alignment_v4_manifest.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"datasets": manifest}, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved safety manifest to {output_path}")

if __name__ == "__main__":
    aggregator = NSFWDrugAggregator(Path("data/collected_2025_2026/safety_v4"))
    aggregator.run()
