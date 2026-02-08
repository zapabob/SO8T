import os
import json
import logging
import subprocess
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class IntegratedMoonshotV4:
    """
    Improved Moonshot Pipeline v4 for SO8ViT.
    Orchestrates Fetch -> Process -> SFT -> RL -> SO8ViT flow.
    """
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[3]
        self.data_dir = self.project_root / "data" / "collected_2025_2026"
        self.scripts_dir = self.project_root / "src" / "data" / "collection"
        self.output_manifest = self.data_dir / "moonshot_v4_manifest.json"

    def run_fetchers(self):
        logger.info("--- PHASE 1: Specialized Data Collection (V4) ---")
        fetchers = [
            "wikipedia_specialized.py",
            "youtube_metadata_fetcher.py",
            "gov_whitepaper_fetcher.py",
            "nsfw_drug_aggregator.py"
        ]
        
        for script in tqdm(fetchers, desc="V4 Fetchers"):
            script_path = self.scripts_dir / script
            logger.info(f"Running fetcher: {script}")
            try:
                subprocess.run(["py", "-3", str(script_path)], check=True, cwd=self.project_root)
            except subprocess.CalledProcessError as e:
                logger.error(f"Fetcher {script} failed: {e}")

    def run_hf_sync(self):
        logger.info("--- PHASE 2: HF CLI Synchronization ---")
        huggingface_script = self.project_root / "src" / "data" / "processing" / "hf_cli_dataset_fetch.py"
        base_dir = self.project_root / "data" / "hf_datasets"
        manifest_path = base_dir / "hf_manifest_v4.json"
        
        logger.info(f"Syncing with HuggingFace datasets to {base_dir}...")
        try:
            subprocess.run([
                "py", "-3", str(huggingface_script),
                "--base-dir", str(base_dir),
                "--manifest", str(manifest_path)
            ], check=True, cwd=self.project_root)
        except subprocess.CalledProcessError as e:
            logger.error(f"HF Sync failed: {e}")

    def run_so8vit_training(self):
        logger.info("--- PHASE 3: SO8ViT Multimodal Thinking Training ---")
        trainer_script = self.project_root / "src" / "training" / "train_unsloth_so8t.py"
        logger.info("Launching Unsloth SO8ViT Training loop...")
        try:
            # We use an environment variable to signal SO8ViT mode if necessary, 
            # though the script itself might be the primary entry point.
            env = os.environ.copy()
            env["SO8T_USE_UNSLOTH"] = "1"
            subprocess.run(["py", "-3", str(trainer_script)], check=True, cwd=self.project_root, env=env)
        except subprocess.CalledProcessError as e:
            logger.error(f"SO8ViT Training failed: {e}")

    def execute_all(self):
        logger.info(f"Starting Improved Moonshot Pipeline v4: {datetime.now()}")
        self.run_fetchers()
        self.run_hf_sync()
        self.run_so8vit_training()
        logger.info("Moonshot Pipeline v4 Complete.")

if __name__ == "__main__":
    pipeline = IntegratedMoonshotV4()
    pipeline.execute_all()
