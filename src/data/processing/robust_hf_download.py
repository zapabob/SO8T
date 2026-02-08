import os
from huggingface_hub import snapshot_download
from pathlib import Path

def download_datasets():
    datasets = [
        {"repo_id": "ibm-research/Toucan", "local_dir": "c:/Users/downl/Desktop/SO8T/data/hf_datasets/Toucan"},
        {"repo_id": "AymanTarig/function-calling-v0.2-with-r1-cot", "local_dir": "c:/Users/downl/Desktop/SO8T/data/hf_datasets/R1-CoT"}
    ]
    
    for ds in datasets:
        print(f"Downloading {ds['repo_id']}...")
        try:
            snapshot_download(
                repo_id=ds["repo_id"],
                repo_type="dataset",
                local_dir=ds["local_dir"],
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded {ds['repo_id']} to {ds['local_dir']}")
        except Exception as e:
            print(f"Failed to download {ds['repo_id']}: {e}")

if __name__ == "__main__":
    download_datasets()
