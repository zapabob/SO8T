#!/usr/bin/env python3
"""
GB-Scale Dataset Acquisition for SO8T Training
Downloads large-scale datasets to address data shortage issue
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import logging
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GBScaleDatasetDownloader:
    """Download large-scale datasets for SO8T training"""

    def __init__(self, output_dir: Path, max_size_gb: float = 50.0):
        self.output_dir = output_dir
        self.max_size_gb = max_size_gb
        self.downloaded_size = 0.0

    def download_slimpajama_sample(self) -> bool:
        """Download SlimPajama-627B sample (high-quality curated data)"""
        dataset_name = "cerebras/SlimPajama-627B"
        dataset_dir = self.output_dir / dataset_name.replace('/', '_')

        logger.info(f"Downloading SlimPajama sample to {dataset_dir}")

        try:
            # Use huggingface_hub to download a subset
            cmd = [
                sys.executable, "-c",
                f"""
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='{dataset_name}',
    repo_type='dataset',
    local_dir='{dataset_dir}',
    max_workers=4,
    resume_download=True
)
"""
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.output_dir))
            if result.returncode == 0:
                size_gb = self._get_dir_size_gb(dataset_dir)
                self.downloaded_size += size_gb
                logger.info(f"✅ SlimPajama downloaded: {size_gb:.2f} GB")
                return True
            else:
                logger.error(f"❌ SlimPajama download failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ SlimPajama download error: {e}")
            return False

    def download_starcoder_sample(self) -> bool:
        """Download StarCoder sample (code dataset)"""
        dataset_name = "bigcode/starcoderdata"
        dataset_dir = self.output_dir / dataset_name.replace('/', '_')

        logger.info(f"Downloading StarCoder sample to {dataset_dir}")

        try:
            # Download just a few files as sample
            cmd = [
                sys.executable, "-c",
                f"""
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
from huggingface_hub import hf_hub_download
import json

# Download dataset info first
try:
    info_file = hf_hub_download(
        repo_id='{dataset_name}',
        filename='data/train-00000-of-00001.parquet',
        repo_type='dataset',
        local_dir='{dataset_dir}',
        local_dir_use_symlinks=False
    )
    print(f"Downloaded StarCoder sample")
except Exception as e:
    print(f"StarCoder download failed: {{e}}")
"""
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.output_dir))
            if result.returncode == 0:
                size_gb = self._get_dir_size_gb(dataset_dir)
                self.downloaded_size += size_gb
                logger.info(f"✅ StarCoder sample downloaded: {size_gb:.2f} GB")
                return True
            else:
                logger.error(f"❌ StarCoder download failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ StarCoder download error: {e}")
            return False

    def download_fineweb_edu_sample(self) -> bool:
        """Download FineWeb-Edu sample (educational content)"""
        dataset_name = "HuggingFaceFW/fineweb-edu"
        dataset_dir = self.output_dir / dataset_name.replace('/', '_')

        logger.info(f"Downloading FineWeb-Edu sample to {dataset_dir}")

        try:
            cmd = [
                sys.executable, "-c",
                f"""
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
from huggingface_hub import hf_hub_download

# Download a small sample
try:
    sample_file = hf_hub_download(
        repo_id='{dataset_name}',
        filename='sample/000_00000.parquet',
        repo_type='dataset',
        local_dir='{dataset_dir}',
        local_dir_use_symlinks=False
    )
    print(f"Downloaded FineWeb-Edu sample")
except Exception as e:
    print(f"FineWeb-Edu download failed: {{e}}")
"""
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.output_dir))
            if result.returncode == 0:
                size_gb = self._get_dir_size_gb(dataset_dir)
                self.downloaded_size += size_gb
                logger.info(f"✅ FineWeb-Edu sample downloaded: {size_gb:.2f} GB")
                return True
            else:
                logger.error(f"❌ FineWeb-Edu download failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ FineWeb-Edu download error: {e}")
            return False

    def _get_dir_size_gb(self, path: Path) -> float:
        """Get directory size in GB"""
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 ** 3)

    def run_download_pipeline(self, datasets: List[str]) -> bool:
        """Run the download pipeline for specified datasets"""
        logger.info(f"=== STARTING GB-SCALE DATASET ACQUISITION ===")
        logger.info(f"Target directory: {self.output_dir}")
        logger.info(f"Max size limit: {self.max_size_gb} GB")
        logger.info(f"Datasets to download: {datasets}")
        logger.info("")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0

        for dataset in datasets:
            if self.downloaded_size >= self.max_size_gb:
                logger.info(f"Reached size limit: {self.downloaded_size:.2f} GB >= {self.max_size_gb} GB")
                break

            if dataset == "slimpajama":
                if self.download_slimpajama_sample():
                    success_count += 1
            elif dataset == "starcoder":
                if self.download_starcoder_sample():
                    success_count += 1
            elif dataset == "fineweb-edu":
                if self.download_fineweb_edu_sample():
                    success_count += 1
            else:
                logger.warning(f"Unknown dataset: {dataset}")

        logger.info("")
        logger.info("=== DOWNLOAD SUMMARY ===")
        logger.info(f"Successful downloads: {success_count}")
        logger.info(f"Total size downloaded: {self.downloaded_size:.2f} GB")
        logger.info(f"Target was: {self.max_size_gb} GB minimum for SO8T training")

        if self.downloaded_size >= 10.0:  # At least 10GB
            logger.info("✅ Sufficient data acquired for initial SO8T training")
            return True
        else:
            logger.warning("⚠️  Data volume still insufficient - consider manual downloads")
            return False


def main():
    parser = argparse.ArgumentParser(description="Download GB-scale datasets for SO8T training")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="D:/webdataset/datasets",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--max-size-gb",
        type=float,
        default=50.0,
        help="Maximum total size to download in GB"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["slimpajama", "starcoder", "fineweb-edu"],
        default=["slimpajama", "starcoder"],
        help="Datasets to download"
    )

    args = parser.parse_args()

    downloader = GBScaleDatasetDownloader(
        output_dir=Path(args.output_dir),
        max_size_gb=args.max_size_gb
    )

    success = downloader.run_download_pipeline(args.datasets)

    # Play audio notification
    try:
        import subprocess
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", "scripts/utils/play_audio_notification.ps1"
        ], capture_output=True)
    except:
        pass

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
