import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Configuration
DATASET_DIR = Path("D:/webdataset")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def save_to_jsonl(dataset, output_path):
    """Save a HuggingFace dataset to JSONL format."""
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset):
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def download_mmlu():
    """Download MMLU (Massive Multitask Language Understanding)"""
    print("\nDownloading MMLU...")
    # MMLU has many configs (subjects). We'll download 'all' if available, or iterate.
    # 'cais/mmlu' is the standard.
    try:
        # Loading 'all' configuration
        ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
        save_to_jsonl(ds, DATASET_DIR / "mmlu_test.jsonl")
        
        ds_val = load_dataset("cais/mmlu", "all", split="validation", trust_remote_code=True)
        save_to_jsonl(ds_val, DATASET_DIR / "mmlu_val.jsonl")
        
        ds_dev = load_dataset("cais/mmlu", "all", split="dev", trust_remote_code=True)
        save_to_jsonl(ds_dev, DATASET_DIR / "mmlu_dev.jsonl")
        
    except Exception as e:
        print(f"Error downloading MMLU: {e}")

def download_gsm8k():
    """Download GSM8K (Grade School Math 8K)"""
    print("\nDownloading GSM8K...")
    try:
        ds_main = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
        save_to_jsonl(ds_main, DATASET_DIR / "gsm8k_test.jsonl")
        
        ds_train = load_dataset("gsm8k", "main", split="train", trust_remote_code=True)
        save_to_jsonl(ds_train, DATASET_DIR / "gsm8k_train.jsonl")
    except Exception as e:
        print(f"Error downloading GSM8K: {e}")

def download_math():
    """Download MATH (Mathematics Aptitude Test of Heuristics)"""
    print("\nDownloading MATH...")
    try:
        # competition_math is the HF ID
        ds_test = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
        save_to_jsonl(ds_test, DATASET_DIR / "math_test.jsonl")
        
        ds_train = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)
        save_to_jsonl(ds_train, DATASET_DIR / "math_train.jsonl")
    except Exception as e:
        print(f"Error downloading MATH: {e}")

def download_elyza100():
    """Download ELYZA-tasks-100"""
    print("\nDownloading ELYZA-tasks-100...")
    try:
        ds = load_dataset("elyza/ELYZA-tasks-100", split="test", trust_remote_code=True)
        save_to_jsonl(ds, DATASET_DIR / "elyza100_test.jsonl")
    except Exception as e:
        print(f"Error downloading ELYZA-100: {e}")

def download_agieval():
    """Download AGIEval"""
    print("\nDownloading AGIEval...")
    try:
        # AGIEval has English and Chinese subsets. We'll try to get English ones.
        # Since 'agieval' might load a specific config, we might need to iterate.
        # For simplicity, we'll try loading the 'english' subset if possible, or raw.
        # Using 'raw' usually gets everything.
        # Note: The official HF repo is often 'microsoft/AGIEval' but might require specific configs.
        # We will try a common mirror or the official one with default config.
        ds = load_dataset("microsoft/AGIEval", "english", split="test", trust_remote_code=True) # Hypothetical config
        save_to_jsonl(ds, DATASET_DIR / "agieval_en_test.jsonl")
    except Exception as e:
        print(f"Error downloading AGIEval (trying default): {e}")
        try:
             # Fallback to iterating common English tasks if 'english' config fails
             tasks = ["sat-math", "sat-en", "lsat-ar", "lsat-lr", "lsat-rc"]
             all_data = []
             for t in tasks:
                 ds = load_dataset("microsoft/AGIEval", t, split="test", trust_remote_code=True)
                 for item in ds:
                     item['task'] = t
                     all_data.append(item)
             save_to_jsonl(all_data, DATASET_DIR / "agieval_subset_test.jsonl")
        except Exception as e2:
            print(f"Error downloading AGIEval fallback: {e2}")

def download_hle():
    """Download Humanity's Last Exam (HLE)"""
    print("\nDownloading Humanity's Last Exam...")
    try:
        # HLE is very new. Checking if available on HF under 'cais/hle' or similar.
        # If not found, we might need to skip or use a placeholder.
        # Assuming 'cais/hle' for now.
        ds = load_dataset("cais/hle", split="test", trust_remote_code=True)
        save_to_jsonl(ds, DATASET_DIR / "hle_test.jsonl")
    except Exception as e:
        print(f"Error downloading HLE: {e}")
        print("HLE might not be publicly available on HF yet or requires specific access.")

def main():
    print(f"Starting dataset download to {DATASET_DIR}...")
    
    download_mmlu()
    download_gsm8k()
    download_math()
    download_elyza100()
    download_agieval()
    download_hle()
    
    print("\nDownload complete!")

if __name__ == "__main__":
    main()
