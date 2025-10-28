#!/usr/bin/env python3
"""
SO8T Model Download Script

Downloads and prepares base models for SO8T training and inference.
Supports Qwen2.5-7B-Instruct and other compatible models.

Usage:
    python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct --output_dir models/
    python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct --quantize --output_dir models/
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download, snapshot_download
import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDownloader:
    """
    Model downloader for SO8T base models.
    
    Handles downloading, caching, and preparing models for training and inference.
    """
    
    def __init__(self, output_dir: str = "models/"):
        """
        Initialize the model downloader.
        
        Args:
            output_dir: Directory to save downloaded models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported models
        self.supported_models = {
            "Qwen/Qwen2.5-7B-Instruct": {
                "size": "7B",
                "context_length": 128000,
                "max_generation": 8192,
                "languages": ["en", "zh", "ja", "ko", "fr", "de", "es", "ru"],
                "license": "Apache-2.0",
                "description": "Qwen2.5-7B-Instruct - Multilingual instruction-tuned model"
            },
            "Qwen/Qwen2.5-14B-Instruct": {
                "size": "14B",
                "context_length": 128000,
                "max_generation": 8192,
                "languages": ["en", "zh", "ja", "ko", "fr", "de", "es", "ru"],
                "license": "Apache-2.0",
                "description": "Qwen2.5-14B-Instruct - Larger multilingual instruction-tuned model"
            },
            "microsoft/DialoGPT-medium": {
                "size": "345M",
                "context_length": 1024,
                "max_generation": 1024,
                "languages": ["en"],
                "license": "MIT",
                "description": "DialoGPT Medium - Conversational model"
            }
        }
    
    def download_model(
        self,
        model_name: str,
        quantize: bool = False,
        force_download: bool = False,
        resume_download: bool = True
    ) -> Dict[str, Any]:
        """
        Download a model from Hugging Face Hub.
        
        Args:
            model_name: Name of the model to download
            quantize: Whether to download quantized version
            force_download: Whether to force re-download
            resume_download: Whether to resume interrupted downloads
            
        Returns:
            Dictionary with download information
        """
        if model_name not in self.supported_models:
            logger.warning(f"Model {model_name} not in supported list, proceeding anyway...")
        
        logger.info(f"Downloading model: {model_name}")
        
        # Create model directory
        model_dir = self.output_dir / model_name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists
        if not force_download and self._model_exists(model_dir):
            logger.info(f"Model {model_name} already exists at {model_dir}")
            return self._get_model_info(model_dir, model_name)
        
        try:
            # Download model files
            logger.info("Downloading model files...")
            snapshot_download(
                repo_id=model_name,
                local_dir=str(model_dir),
                force_download=force_download,
                resume_download=resume_download,
                local_files_only=False
            )
            
            # Download tokenizer
            logger.info("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(model_dir),
                force_download=force_download
            )
            
            # Save tokenizer
            tokenizer.save_pretrained(str(model_dir))
            
            # Create model info
            model_info = self._create_model_info(model_dir, model_name, quantize)
            
            # Save model info
            info_file = model_dir / "model_info.json"
            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Model {model_name} downloaded successfully to {model_dir}")
            return model_info
            
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            raise
    
    def _model_exists(self, model_dir: Path) -> bool:
        """Check if model already exists."""
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        return all((model_dir / file).exists() for file in required_files)
    
    def _get_model_info(self, model_dir: Path, model_name: str) -> Dict[str, Any]:
        """Get existing model information."""
        info_file = model_dir / "model_info.json"
        if info_file.exists():
            with open(info_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return self._create_model_info(model_dir, model_name, False)
    
    def _create_model_info(
        self,
        model_dir: Path,
        model_name: str,
        quantized: bool
    ) -> Dict[str, Any]:
        """Create model information dictionary."""
        # Get model config
        try:
            config = AutoConfig.from_pretrained(str(model_dir))
            config_dict = config.to_dict()
        except:
            config_dict = {}
        
        # Get file sizes
        file_sizes = {}
        total_size = 0
        
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                file_sizes[str(file_path.relative_to(model_dir))] = size
                total_size += size
        
        # Create model info
        model_info = {
            "model_name": model_name,
            "model_dir": str(model_dir),
            "quantized": quantized,
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024**3),
            "file_sizes": file_sizes,
            "config": config_dict,
            "supported_models": self.supported_models.get(model_name, {}),
            "download_timestamp": str(torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True)) if torch.cuda.is_available() else 0)
        }
        
        return model_info
    
    def list_models(self) -> None:
        """List available models."""
        print("Supported models:")
        print("-" * 80)
        
        for model_name, info in self.supported_models.items():
            print(f"Model: {model_name}")
            print(f"  Size: {info['size']}")
            print(f"  Context Length: {info['context_length']:,}")
            print(f"  Max Generation: {info['max_generation']:,}")
            print(f"  Languages: {', '.join(info['languages'])}")
            print(f"  License: {info['license']}")
            print(f"  Description: {info['description']}")
            print()
    
    def verify_model(self, model_name: str) -> bool:
        """Verify that a downloaded model is working correctly."""
        model_dir = self.output_dir / model_name.replace("/", "_")
        
        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            return False
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            
            # Test tokenization
            test_text = "Hello, world! こんにちは、世界！"
            tokens = tokenizer(test_text, return_tensors="pt")
            
            logger.info(f"Tokenizer test passed. Input: '{test_text}'")
            logger.info(f"Token IDs: {tokens['input_ids'].tolist()}")
            logger.info(f"Attention mask: {tokens['attention_mask'].tolist()}")
            
            # Load model config
            config = AutoConfig.from_pretrained(str(model_dir))
            logger.info(f"Model config loaded successfully")
            logger.info(f"Model type: {config.model_type}")
            logger.info(f"Hidden size: {config.hidden_size}")
            logger.info(f"Number of layers: {config.num_hidden_layers}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False
    
    def cleanup_old_models(self, keep_latest: int = 3) -> None:
        """Clean up old model downloads."""
        model_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        
        if len(model_dirs) <= keep_latest:
            logger.info("No models to clean up")
            return
        
        # Sort by modification time
        model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old models
        for model_dir in model_dirs[keep_latest:]:
            logger.info(f"Removing old model: {model_dir}")
            import shutil
            shutil.rmtree(model_dir)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download SO8T base models")
    parser.add_argument("--model", type=str, required=True, help="Model name to download")
    parser.add_argument("--output_dir", type=str, default="models/", help="Output directory")
    parser.add_argument("--quantize", action="store_true", help="Download quantized version")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded model")
    parser.add_argument("--list", action="store_true", help="List supported models")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old models")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ModelDownloader(args.output_dir)
    
    if args.list:
        downloader.list_models()
        return
    
    if args.cleanup:
        downloader.cleanup_old_models()
        return
    
    # Download model
    try:
        model_info = downloader.download_model(
            model_name=args.model,
            quantize=args.quantize,
            force_download=args.force
        )
        
        print(f"✅ Model downloaded successfully!")
        print(f"📁 Location: {model_info['model_dir']}")
        print(f"📊 Size: {model_info['total_size_gb']:.2f} GB")
        
        # Verify model if requested
        if args.verify:
            print("\n🔍 Verifying model...")
            if downloader.verify_model(args.model):
                print("✅ Model verification passed!")
            else:
                print("❌ Model verification failed!")
                return 1
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
