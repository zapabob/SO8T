#!/usr/bin/env python3
"""
Upload AEGIS model to HuggingFace Hub
"""

import os
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, upload_folder, upload_file
import argparse

def upload_aegis_model(repo_name: str, token: str = None):
    """
    Upload AEGIS model to HuggingFace Hub

    Args:
        repo_name: HuggingFace repository name (e.g., "your-username/AEGIS-v2.0-Phi3.5-thinking")
        token: HuggingFace API token (optional, will use env var if not provided)
    """

    # Set up paths
    upload_dir = Path(r"D:\webdataset\models\aegis-huggingface-upload")
    model_dir = Path(r"models\aegis_adjusted")

    if not upload_dir.exists():
        raise FileNotFoundError(f"Upload directory not found: {upload_dir}")

    print("🚀 Starting AEGIS model upload to HuggingFace Hub")
    print(f"📁 Repository: {repo_name}")
    print(f"📂 Upload directory: {upload_dir}")

    # Initialize HuggingFace API
    api = HfApi(token=token)

    try:
        # Create repository if it doesn't exist
        try:
            api.repo_info(repo_name)
            print("✅ Repository already exists")
        except:
            api.create_repo(repo_name, repo_type="model")
            print("✅ Repository created")

        # Upload small files first (config, tokenizer, README)
        print("\n📤 Uploading configuration and documentation files...")

        small_files = [
            "README.md",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json"
        ]

        for file_name in small_files:
            file_path = upload_dir / file_name
            if file_path.exists():
                print(f"  📄 Uploading {file_name}...")
                upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_name,
                    repo_id=repo_name,
                    token=token
                )

        # Upload benchmark results
        print("\n📊 Uploading benchmark visualizations...")
        benchmark_dir = upload_dir / "benchmark_results"
        if benchmark_dir.exists():
            for png_file in benchmark_dir.glob("*.png"):
                print(f"  🖼️ Uploading {png_file.name}...")
                upload_file(
                    path_or_fileobj=str(png_file),
                    path_in_repo=f"benchmark_results/{png_file.name}",
                    repo_id=repo_name,
                    token=token
                )

        # Upload large model files
        print("\n🤖 Uploading model weights (this may take a while)...")

        safetensors_files = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors"
        ]

        for file_name in safetensors_files:
            model_file = model_dir / file_name
            if model_file.exists():
                print(f"  📦 Uploading {file_name} ({model_file.stat().st_size / (1024**3):.1f} GB)...")
                upload_file(
                    path_or_fileobj=str(model_file),
                    path_in_repo=file_name,
                    repo_id=repo_name,
                    token=token
                )
            else:
                print(f"  ⚠️ Model file not found: {model_file}")

        print("\n✅ Upload completed successfully!")
        print(f"🌐 Model available at: https://huggingface.co/{repo_name}")

        # Update repository metadata
        print("\n📝 Updating repository metadata...")

        api.update_repo_visibility(repo_name, private=False)

        # Add model tags
        api.add_space_tags(repo_name, tags=[
            "transformers",
            "phi-3",
            "enhanced-reasoning",
            "ethical-ai",
            "japanese",
            "reasoning",
            "safety",
            "transformer",
            "mathematical-reasoning",
            "quadruple-reasoning",
            "thinking-model"
        ])

        print("✅ Repository metadata updated")

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Upload AEGIS model to HuggingFace Hub")
    parser.add_argument("repo_name", help="HuggingFace repository name (e.g., 'your-username/AEGIS-v2.0-Phi3.5-thinking')")
    parser.add_argument("--token", help="HuggingFace API token", default=None)

    args = parser.parse_args()

    # Get token from environment if not provided
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("❌ HuggingFace token not provided. Set HF_TOKEN environment variable or use --token")
        return

    upload_aegis_model(args.repo_name, token)

if __name__ == "__main__":
    main()
