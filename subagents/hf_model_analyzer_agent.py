#!/usr/bin/env python3
"""
HF Model Analyzer Sub-Agent for SO8T
Specialized agent for downloading, analyzing, and optimizing HF models for SO8T implementation
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests

class HFModelAnalyzerAgent:
    """
    Specialized sub-agent for HF model analysis and SO8T integration
    Handles model downloading, architecture analysis, and optimization strategies
    """

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir)
        self.models_dir = self.workspace_dir / "models"
        self.analysis_dir = self.workspace_dir / "model_analysis"
        self.models_dir.mkdir(exist_ok=True)
        self.analysis_dir.mkdir(exist_ok=True)

        # RTX3060 constraints
        self.vram_limit = 12  # GB
        self.ram_limit = 32   # GB

        # SO8T preferred models
        self.so8t_models = [
            'Qwen/Qwen2.5-3B-Instruct',
            'Qwen/Qwen2.5-7B-Instruct',
            'google/vit-base-patch16-224',
            'facebook/dinov2-small',
            'microsoft/DialoGPT-medium',
            'sentence-transformers/all-MiniLM-L6-v2'
        ]

    def download_model(self, model_name: str, local_dir: Optional[str] = None) -> bool:
        """
        Download HF model using CLI
        """
        if local_dir is None:
            safe_name = model_name.replace('/', '_')
            local_dir = str(self.models_dir / safe_name)

        print(f"Downloading {model_name} to {local_dir}")

        try:
            # Check if huggingface-cli is available
            result = subprocess.run(['huggingface-cli', '--version'],
                                  capture_output=True, text=True)

            if result.returncode != 0:
                print("huggingface-cli not found. Installing...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub[cli]'],
                             check=True)

            # Download model
            cmd = ['huggingface-cli', 'download', model_name, '--local-dir', local_dir]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"Successfully downloaded {model_name}")
                return True
            else:
                print(f"Download failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"Download error: {e}")
            return False

    def analyze_model_architecture(self, model_path: str) -> Dict:
        """
        Analyze model architecture and check for SO8T compatibility
        """
        config_path = Path(model_path) / "config.json"

        if not config_path.exists():
            return {"error": "config.json not found"}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Extract key parameters
            analysis = {
                "model_path": model_path,
                "model_type": config.get("model_type", "unknown"),
                "architecture": config.get("architectures", []),
            }

            # Transformer-specific parameters
            if "hidden_size" in config:
                hidden_size = config["hidden_size"]
                analysis.update({
                    "hidden_size": hidden_size,
                    "num_hidden_layers": config.get("num_hidden_layers", 0),
                    "num_attention_heads": config.get("num_attention_heads", 0),
                    "intermediate_size": config.get("intermediate_size", 0),
                    "vocab_size": config.get("vocab_size", 0),
                })

                # Check multiples of 8
                analysis["hidden_size_multiple_of_8"] = hidden_size % 8 == 0
                analysis["attention_heads_multiple_of_8"] = config.get("num_attention_heads", 0) % 8 == 0

                # SO8T compatibility assessment
                analysis["so8t_compatibility"] = self._assess_so8t_compatibility(analysis)

                # Optimization recommendations
                analysis["optimization_recommendations"] = self._get_optimization_recommendations(analysis)

            # Vision-specific parameters
            elif "num_channels" in config:  # ViT models
                analysis.update({
                    "hidden_size": config.get("hidden_size", 0),
                    "num_hidden_layers": config.get("num_hidden_layers", 0),
                    "num_attention_heads": config.get("num_attention_heads", 0),
                    "patch_size": config.get("patch_size", 0),
                    "num_channels": config.get("num_channels", 0),
                })

            return analysis

        except Exception as e:
            return {"error": f"Analysis failed: {e}"}

    def _assess_so8t_compatibility(self, analysis: Dict) -> Dict:
        """
        Assess SO8T compatibility based on model architecture
        """
        compatibility = {
            "overall_score": 0,
            "triality_suitable": False,
            "grape_suitable": False,
            "rtx3060_compatible": False,
            "recommendations": []
        }

        hidden_size = analysis.get("hidden_size", 0)
        num_heads = analysis.get("num_attention_heads", 0)

        # Triality compatibility (SO(8) vector-spinor operations)
        if hidden_size % 8 == 0:
            compatibility["triality_suitable"] = True
            compatibility["overall_score"] += 2

        # GRAPE compatibility (Group Representational Position Encoding)
        if hidden_size >= 256 and num_heads >= 8:
            compatibility["grape_suitable"] = True
            compatibility["overall_score"] += 2

        # RTX3060 compatibility
        estimated_vram = self._estimate_vram_usage(analysis)
        if estimated_vram <= self.vram_limit:
            compatibility["rtx3060_compatible"] = True
            compatibility["overall_score"] += 1

        # Recommendations
        if not compatibility["triality_suitable"]:
            compatibility["recommendations"].append(
                f"Consider padding hidden_size {hidden_size} to nearest multiple of 8 for Triality operations"
            )

        if not compatibility["grape_suitable"]:
            compatibility["recommendations"].append(
                "Model may need dimension adjustments for GRAPE position encoding"
            )

        if not compatibility["rtx3060_compatible"]:
            compatibility["recommendations"].append(
                f"Estimated VRAM usage ({estimated_vram}GB) exceeds RTX3060 limit ({self.vram_limit}GB)"
            )

        return compatibility

    def _estimate_vram_usage(self, analysis: Dict) -> float:
        """
        Estimate VRAM usage for RTX3060 compatibility check
        """
        hidden_size = analysis.get("hidden_size", 0)
        num_layers = analysis.get("num_hidden_layers", 0)
        vocab_size = analysis.get("vocab_size", 0)

        # Rough estimation
        if hidden_size and num_layers:
            # Model parameters
            params_gb = (hidden_size * num_layers * 2) / (1024**3)  # 2 for fp16

            # KV cache (rough estimate for 2048 tokens)
            kv_cache_gb = (hidden_size * num_layers * 2048 * 2) / (1024**3)

            # Embeddings
            if vocab_size:
                embed_gb = (vocab_size * hidden_size) / (1024**3)
            else:
                embed_gb = 0.5  # default estimate

            total_gb = params_gb + kv_cache_gb + embed_gb + 1  # +1GB for buffers
            return min(total_gb, 50)  # Cap at reasonable maximum

        return 8.0  # Default estimate

    def _get_optimization_recommendations(self, analysis: Dict) -> List[str]:
        """
        Get optimization recommendations for the model
        """
        recommendations = []

        hidden_size = analysis.get("hidden_size", 0)
        num_heads = analysis.get("num_attention_heads", 0)

        # Multiples of 8 recommendations
        if hidden_size % 8 != 0:
            nearest_8 = ((hidden_size + 7) // 8) * 8
            recommendations.append(
                f"Consider resizing hidden_size from {hidden_size} to {nearest_8} for TensorRT optimization"
            )

        if num_heads % 8 != 0:
            recommendations.append(
                f"num_attention_heads ({num_heads}) is not multiple of 8 - may impact parallelization"
            )

        # SO8T-specific recommendations
        if hidden_size < 512:
            recommendations.append(
                "Hidden size relatively small - may limit SO8T geometric operations"
            )

        if analysis.get("num_hidden_layers", 0) < 12:
            recommendations.append(
                "Few layers - consider deeper architecture for complex reasoning"
            )

        return recommendations

    def get_model_info_from_hf(self, model_name: str) -> Dict:
        """
        Get model information from Hugging Face API
        """
        try:
            api_url = f"https://huggingface.co/api/models/{model_name}"
            response = requests.get(api_url, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned {response.status_code}"}

        except Exception as e:
            return {"error": str(e)}

    def create_so8t_model_report(self, model_name: str) -> Dict:
        """
        Create comprehensive SO8T compatibility report for a model
        """
        # Download model if not exists
        safe_name = model_name.replace('/', '_')
        model_path = self.models_dir / safe_name

        if not model_path.exists():
            print(f"Downloading {model_name}...")
            if not self.download_model(model_name, str(model_path)):
                return {"error": "Download failed"}

        # Analyze architecture
        architecture = self.analyze_model_architecture(str(model_path))

        # Get HF metadata
        hf_info = self.get_model_info_from_hf(model_name)

        # Create report
        report = {
            "model_name": model_name,
            "local_path": str(model_path),
            "download_timestamp": str(Path(model_path).stat().st_mtime),
            "architecture_analysis": architecture,
            "hf_metadata": hf_info,
            "so8t_assessment": {
                "compatibility_score": architecture.get("so8t_compatibility", {}).get("overall_score", 0),
                "strengths": [],
                "weaknesses": [],
                "recommended_actions": []
            }
        }

        # Assess strengths and weaknesses
        compat = architecture.get("so8t_compatibility", {})

        if compat.get("triality_suitable"):
            report["so8t_assessment"]["strengths"].append("Suitable for SO(8) Triality operations")
        else:
            report["so8t_assessment"]["weaknesses"].append("Not optimized for SO(8) Triality operations")

        if compat.get("grape_suitable"):
            report["so8t_assessment"]["strengths"].append("Compatible with GRAPE position encoding")
        else:
            report["so8t_assessment"]["weaknesses"].append("May need modifications for GRAPE")

        if compat.get("rtx3060_compatible"):
            report["so8t_assessment"]["strengths"].append("RTX3060 compatible")
        else:
            report["so8t_assessment"]["weaknesses"].append("May exceed RTX3060 VRAM limits")

        # Add recommendations
        recommendations = architecture.get("optimization_recommendations", [])
        compat_recs = compat.get("recommendations", [])
        report["so8t_assessment"]["recommended_actions"] = recommendations + compat_recs

        return report

    def batch_analyze_so8t_models(self) -> List[Dict]:
        """
        Analyze all SO8T preferred models
        """
        reports = []

        for model_name in self.so8t_models:
            print(f"\\nAnalyzing {model_name}...")
            try:
                report = self.create_so8t_model_report(model_name)
                reports.append(report)

                # Save individual report
                safe_name = model_name.replace('/', '_')
                report_file = self.analysis_dir / f"{safe_name}_so8t_analysis.json"

                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)

                print(f"Report saved: {report_file}")

            except Exception as e:
                print(f"Analysis failed for {model_name}: {e}")
                reports.append({"model_name": model_name, "error": str(e)})

        return reports

def main():
    agent = HFModelAnalyzerAgent()

    print("HF Model Analyzer Sub-Agent for SO8T")
    print("=" * 50)

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "download":
            if len(sys.argv) > 2:
                model_name = sys.argv[2]
                agent.download_model(model_name)
            else:
                print("Usage: python hf_model_analyzer_agent.py download <model_name>")

        elif command == "analyze":
            if len(sys.argv) > 2:
                model_name = sys.argv[2]
                report = agent.create_so8t_model_report(model_name)
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                print("Usage: python hf_model_analyzer_agent.py analyze <model_name>")

        elif command == "batch":
            reports = agent.batch_analyze_so8t_models()
            print(f"\\nAnalyzed {len(reports)} models")

            # Save summary
            summary_file = agent.analysis_dir / "so8t_models_batch_analysis.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(reports, f, indent=2, ensure_ascii=False)
            print(f"Batch summary saved: {summary_file}")

        else:
            print("Commands: download, analyze, batch")

    else:
        print("SO8T HF Model Analyzer Sub-Agent")
        print("Commands:")
        print("  download <model_name>  - Download a specific model")
        print("  analyze <model_name>   - Analyze a downloaded model")
        print("  batch                  - Analyze all SO8T preferred models")

if __name__ == '__main__':
    main()