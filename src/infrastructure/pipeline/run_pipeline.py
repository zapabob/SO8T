# run_pipeline.py
"""Master pipeline to run data collection, preprocessing, and training.

Usage:
    python run_pipeline.py
"""
import subprocess
import sys
from pathlib import Path
import time

PYTHON = sys.executable
BASE_DIR = Path(__file__).parent
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_DIR = Path("D:/dataset")

def run_step(cmd, description):
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(">>> Success")
    except subprocess.CalledProcessError as e:
        print(f">>> Failed with exit code {e.returncode}")
        sys.exit(1)

def main():
    # 1. Collect ArXiv
    # run_step([PYTHON, str(SCRIPTS_DIR / "collect_arxiv.py"), "--category", "cs.AI", "--output", str(DATA_DIR / "arxiv")], "Collecting ArXiv Papers")
    
    # 2. Download Wikipedia
    run_step([PYTHON, str(SCRIPTS_DIR / "download_wikipedia.py"), "--output", str(DATA_DIR / "wikipedia")], "Downloading Wikipedia Categories")
    
    # 3. Scrape Playwright (Human-like)
    run_step([PYTHON, str(SCRIPTS_DIR / "scrape_playwright.py"), "--url-file", str(SCRIPTS_DIR / "urls.txt"), "--output", str(DATA_DIR / "playwright")], "Scraping Web Pages (Human-like)")
    
    # 4. Preprocess
    run_step([PYTHON, str(SCRIPTS_DIR / "preprocess.py"), "--input-dir", str(DATA_DIR), "--output", str(DATA_DIR / "processed.jsonl")], "Preprocessing Data")
    
    # 5. Create Dataset Splits
    run_step([PYTHON, str(SCRIPTS_DIR / "create_dataset.py"), "--input", str(DATA_DIR / "processed.jsonl"), "--output-dir", str(DATA_DIR / "final")], "Creating Train/Val/Test Splits")
    
    # 6. Verify
    run_step([PYTHON, str(SCRIPTS_DIR / "verify_dataset.py"), "--data-dir", str(DATA_DIR / "final")], "Verifying Dataset")
    
    print("\n=== Pipeline Complete ===")
    print("You can now run the fine-tuning script manually or uncomment it in this pipeline.")
    # run_step([PYTHON, "fine_tune/fine_tune_agiasi.py", ...], "Fine-tuning")

if __name__ == "__main__":
    main()
