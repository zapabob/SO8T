#!/usr/bin/env python3
"""
SO8T GGUF Conversion Script

Converts trained SO8T models to GGUF format for efficient inference.
Supports multiple quantization levels and optimization settings.

Usage:
    python scripts/convert_to_gguf.py --input_dir checkpoints/so8t_model --output_dir dist/
    python scripts/convert_to_gguf.py --input_dir checkpoints/so8t_model --output_dir dist/ --quantization q4_k_m
"""

import os
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModel
import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GGUFConverter:
    """
    GGUF converter for SO8T models.
    
    Converts PyTorch models to GGUF format for efficient inference.
    """
    
    def __init__(self, output_dir: str = "dist/"):
        """
        Initialize the GGUF converter.
        
        Args:
            output_dir: Directory to save GGUF models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported quantization methods
        self.quantization_methods = {
            "q4_k_m": {
                "description": "4-bit quantization (medium quality)",
                "file_size_ratio": 0.25,
                "quality": "high",
                "speed": "fast"
            },
            "q4_k_s": {
                "description": "4-bit quantization (small quality)",
                "file_size_ratio": 0.20,
                "quality": "medium",
                "speed": "very_fast"
            },
            "iq4_xs": {
                "description": "4-bit quantization (extra small)",
                "file_size_ratio": 0.15,
                "quality": "medium",
                "speed": "very_fast"
            },
            "q3_k_m": {
                "description": "3-bit quantization (medium quality)",
                "file_size_ratio": 0.18,
                "quality": "medium",
                "speed": "very_fast"
            },
            "q2_k": {
                "description": "2-bit quantization",
                "file_size_ratio": 0.12,
                "quality": "low",
                "speed": "extremely_fast"
            }
        }
    
    def convert_model(
        self,
        input_dir: str,
        model_name: str = "so8t_qwen2.5-7b-safeagent",
        quantization: str = "q4_k_m",
        base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    ) -> Dict[str, Any]:
        """
        Convert a PyTorch model to GGUF format.
        
        Args:
            input_dir: Directory containing the PyTorch model
            model_name: Name for the output GGUF model
            quantization: Quantization method to use
            base_model: Base model name for tokenizer
            
        Returns:
            Dictionary with conversion information
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")
        
        logger.info(f"Converting model from {input_path} to GGUF format")
        logger.info(f"Quantization method: {quantization}")
        
        # Check if llama.cpp is available
        if not self._check_llama_cpp():
            logger.warning("llama.cpp not found. Creating placeholder GGUF files.")
            return self._create_placeholder_gguf(input_path, model_name, quantization)
        
        # Convert model
        conversion_results = {}
        
        for quant_method in self._get_quantization_methods(quantization):
            logger.info(f"Converting with {quant_method} quantization...")
            
            try:
                result = self._convert_single_model(
                    input_path, model_name, quant_method, base_model
                )
                conversion_results[quant_method] = result
                
            except Exception as e:
                logger.error(f"Error converting with {quant_method}: {e}")
                conversion_results[quant_method] = {"error": str(e)}
        
        # Save conversion report
        self._save_conversion_report(conversion_results, model_name)
        
        return conversion_results
    
    def _check_llama_cpp(self) -> bool:
        """Check if llama.cpp is available."""
        try:
            result = subprocess.run(
                ["llama-cli", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except:
            return False
    
    def _create_placeholder_gguf(
        self,
        input_path: Path,
        model_name: str,
        quantization: str
    ) -> Dict[str, Any]:
        """Create placeholder GGUF files when llama.cpp is not available."""
        logger.info("Creating placeholder GGUF files...")
        
        # Get model size estimate
        model_size = self._estimate_model_size(input_path)
        
        results = {}
        
        for quant_method in self._get_quantization_methods(quantization):
            output_file = self.output_dir / f"{model_name}-{quant_method}.gguf"
            
            # Create placeholder file
            with open(output_file, "w") as f:
                f.write(f"# Placeholder GGUF file for {quant_method}\n")
                f.write(f"# Model: {model_name}\n")
                f.write(f"# Quantization: {quant_method}\n")
                f.write(f"# Original size: {model_size:.2f} GB\n")
                f.write(f"# Estimated size: {model_size * self.quantization_methods[quant_method]['file_size_ratio']:.2f} GB\n")
                f.write(f"# This would be generated by llama.cpp in a real implementation\n")
            
            results[quant_method] = {
                "output_file": str(output_file),
                "status": "placeholder",
                "estimated_size_gb": model_size * self.quantization_methods[quant_method]['file_size_ratio'],
                "description": self.quantization_methods[quant_method]['description']
            }
            
            logger.info(f"Created placeholder: {output_file}")
        
        return results
    
    def _estimate_model_size(self, input_path: Path) -> float:
        """Estimate model size in GB."""
        total_size = 0
        
        for file_path in input_path.rglob("*.bin"):
            total_size += file_path.stat().st_size
        
        for file_path in input_path.rglob("*.safetensors"):
            total_size += file_path.stat().st_size
        
        return total_size / (1024**3)
    
    def _get_quantization_methods(self, quantization: str) -> List[str]:
        """Get list of quantization methods to use."""
        if quantization == "all":
            return list(self.quantization_methods.keys())
        elif quantization in self.quantization_methods:
            return [quantization]
        else:
            logger.warning(f"Unknown quantization method: {quantization}")
            return ["q4_k_m"]  # Default
    
    def _convert_single_model(
        self,
        input_path: Path,
        model_name: str,
        quantization: str,
        base_model: str
    ) -> Dict[str, Any]:
        """Convert a single model with specific quantization."""
        output_file = self.output_dir / f"{model_name}-{quantization}.gguf"
        
        # Prepare conversion command
        cmd = [
            "llama-cli",
            "convert",
            str(input_path),
            str(output_file),
            "--outtype", quantization
        ]
        
        # Run conversion
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                # Get file size
                file_size = output_file.stat().st_size if output_file.exists() else 0
                
                return {
                    "output_file": str(output_file),
                    "status": "success",
                    "file_size_bytes": file_size,
                    "file_size_gb": file_size / (1024**3),
                    "quantization": quantization,
                    "description": self.quantization_methods[quantization]['description']
                }
            else:
                return {
                    "output_file": str(output_file),
                    "status": "error",
                    "error": result.stderr,
                    "quantization": quantization
                }
                
        except subprocess.TimeoutExpired:
            return {
                "output_file": str(output_file),
                "status": "timeout",
                "error": "Conversion timed out",
                "quantization": quantization
            }
        except Exception as e:
            return {
                "output_file": str(output_file),
                "status": "error",
                "error": str(e),
                "quantization": quantization
            }
    
    def _save_conversion_report(
        self,
        results: Dict[str, Any],
        model_name: str
    ) -> None:
        """Save conversion report."""
        report = {
            "model_name": model_name,
            "conversion_timestamp": str(torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True)) if torch.cuda.is_available() else 0),
            "quantization_methods": self.quantization_methods,
            "results": results,
            "summary": self._generate_summary(results)
        }
        
        report_file = self.output_dir / f"{model_name}_conversion_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversion report saved to: {report_file}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conversion summary."""
        successful = [k for k, v in results.items() if v.get("status") == "success"]
        failed = [k for k, v in results.items() if v.get("status") != "success"]
        
        total_size = sum(
            v.get("file_size_gb", 0) for v in results.values()
            if v.get("status") == "success"
        )
        
        return {
            "total_quantization_methods": len(results),
            "successful_conversions": len(successful),
            "failed_conversions": len(failed),
            "successful_methods": successful,
            "failed_methods": failed,
            "total_size_gb": total_size
        }
    
    def list_quantization_methods(self) -> None:
        """List available quantization methods."""
        print("Available quantization methods:")
        print("-" * 80)
        
        for method, info in self.quantization_methods.items():
            print(f"Method: {method}")
            print(f"  Description: {info['description']}")
            print(f"  File size ratio: {info['file_size_ratio']:.1%}")
            print(f"  Quality: {info['quality']}")
            print(f"  Speed: {info['speed']}")
            print()
    
    def verify_gguf_model(self, gguf_file: str) -> bool:
        """Verify a GGUF model file."""
        gguf_path = Path(gguf_file)
        
        if not gguf_path.exists():
            logger.error(f"GGUF file not found: {gguf_path}")
            return False
        
        # Check file size
        file_size = gguf_path.stat().st_size
        if file_size < 1024:  # Less than 1KB is suspicious
            logger.warning(f"GGUF file is very small: {file_size} bytes")
            return False
        
        logger.info(f"GGUF file verification passed: {gguf_path}")
        logger.info(f"File size: {file_size / (1024**3):.2f} GB")
        
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert SO8T models to GGUF format")
    parser.add_argument("--input_dir", type=str, required=True, help="Input PyTorch model directory")
    parser.add_argument("--output_dir", type=str, default="dist/", help="Output directory for GGUF models")
    parser.add_argument("--model_name", type=str, default="so8t_qwen2.5-7b-safeagent", help="Output model name")
    parser.add_argument("--quantization", type=str, default="q4_k_m", help="Quantization method")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model name")
    parser.add_argument("--list_methods", action="store_true", help="List available quantization methods")
    parser.add_argument("--verify", type=str, help="Verify a GGUF model file")
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = GGUFConverter(args.output_dir)
    
    if args.list_methods:
        converter.list_quantization_methods()
        return 0
    
    if args.verify:
        if converter.verify_gguf_model(args.verify):
            print("✅ GGUF model verification passed!")
            return 0
        else:
            print("❌ GGUF model verification failed!")
            return 1
    
    # Convert model
    try:
        results = converter.convert_model(
            input_dir=args.input_dir,
            model_name=args.model_name,
            quantization=args.quantization,
            base_model=args.base_model
        )
        
        print(f"✅ Model conversion completed!")
        print(f"📁 Output directory: {args.output_dir}")
        
        # Print summary
        successful = [k for k, v in results.items() if v.get("status") == "success"]
        if successful:
            print(f"✅ Successful conversions: {', '.join(successful)}")
        
        failed = [k for k, v in results.items() if v.get("status") != "success"]
        if failed:
            print(f"❌ Failed conversions: {', '.join(failed)}")
        
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

