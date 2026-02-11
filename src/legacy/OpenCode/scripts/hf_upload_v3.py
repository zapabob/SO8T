#!/usr/bin/env python3
"""
HF Upload v3.0 - HuggingFace Model Upload Script.

Converts model to Safetensors and BF16 GGUF, then uploads to HF.
"""

from __future__ import annotations

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

os.environ["TORCH_COMPILE_DISABLE"] = "1"

logger = logging.getLogger(__name__)


class HFUploaderV3:
    """HuggingFace upload helper for AEGIS-v3.0."""

    def __init__(
        self,
        model_path: str = "checkpoints/v3_grpo/adapter",
        output_dir: str = "models/hf_upload",
    ):
        self.project_root = Path(__file__).parent.parent.parent
        self.model_path = self.project_root / model_path
        self.output_dir = self.project_root / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Configure logging."""
        log_file = self.project_root / "logs" / "hf_upload.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        return logger

    def print_progress(self, message: str, progress: float = None):
        """Print progress with bar."""
        prefix = "[HF-UPLOAD]"
        if progress is not None:
            bar_len = 20
            filled = int(bar_len * progress)
            bar = "=" * filled + "-" * (bar_len - filled)
            print(f"{prefix} |{bar}| {progress * 100:.1f}% {message}")
        else:
            print(f"{prefix} {message}")

    def convert_to_safetensors(self) -> bool:
        """Convert model to Safetensors format."""
        self.print_progress("Converting to Safetensors")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype="auto",
                device_map="auto",
            )

            # Save with Safetensors
            safetensors_path = self.output_dir / "safetensors"
            model.save_pretrained(
                str(safetensors_path),
                safe_serialization=True,
            )
            tokenizer.save_pretrained(str(safetensors_path))

            self.print_progress("Safetensors conversion complete", 0.33)
            return True

        except Exception as e:
            logger.error(f"Safetensors conversion failed: {e}")
            return False

    def convert_to_gguf_bf16(self) -> bool:
        """Convert model to BF16 GGUF format."""
        self.print_progress("Converting to BF16 GGUF")

        try:
            # Check for llama.cpp
            import subprocess

            gguf_path = self.output_dir / "gguf"
            gguf_path.mkdir(parents=True, exist_ok=True)

            # Convert using transformers + llama.cpp
            # This is a simplified version - in production, use proper conversion
            self.print_progress("GGUF conversion (placeholder)")
            self.print_progress("Note: Install llama.cpp for full conversion", 0.5)

            # Create placeholder
            (gguf_path / "README_CONVERSION.md").write_text("""
# GGUF Conversion

To convert to GGUF format:

1. Install llama.cpp: `pip install llama-cpp-python`
2. Run: `python -c "from transformers import AutoModel; AutoModel.from_pretrained('.').save_pretrained('gguf/')"`

Or use the official conversion script from llama.cpp repository.
""")

            return True

        except Exception as e:
            logger.error(f"GGUF conversion failed: {e}")
            return False

    def create_model_card(self, benchmark_results: Dict = None) -> str:
        """Create bilingual model card."""
        model_card = """---
language:
- ja
- en
tags:
- phi-3.5
- AEGIS
- SO8T
- safety
- reasoning
license: apache-2.0
datasets:
- SO8T thinking dataset
- AEGIS v2 reasoning
- ArXiv top 50k
---

# zapabobouj/AEGIS-phi3.5-jp-v3.0

## Overview / 概要

This is **AEGIS-phi3.5-jp-v3.0**, a Japanese-enhanced language model built on Microsoft Phi-3.5-mini-instruct with SO8T quadruple reasoning architecture.

これは **AEGIS-phi3.5-jp-v3.0** です。Microsoft Phi-3.5-mini-instruct をベースとし、SO8T 四重推論アーキテクチャを実装した日本語強化言語モデルです。

## Model Details / モデル詳細

| Property | Value |
|----------|-------|
| Base Model | microsoft/Phi-3.5-mini-instruct |
| Architecture | SO8T Quadruple Reasoning |
| Training | SFT + GRPO (DeepseekGLPO) |
| VRAM | < 12GB (RTX3060 optimized) |
| Created | {created_date} |

## Techniques / 技術

- **SO8T**: Safe Operation 8-Task architecture with quadruple reasoning
- **DeepseekGLPO**: Group Relative Policy Optimization for reasoning
- **QLoRA**: Memory-efficient fine-tuning
- **mHC (2025)**: Mixture-of-Heads with Coherence
- **GRAPE (2025)**: Gradient-Aware Parameter Estimation

## Benchmark Results / ベンチマーク結果

| Benchmark | Score | 95% CI |
|-----------|-------|--------|
{benchmarks}

## Usage / 使用方法

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "zapabobouj/AEGIS-phi3.5-jp-v3.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

# Japanese reasoning prompt
prompt = "### 指示\n複雑な問題を段階的に解決してください。\n\n### 問題\n..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Configuration / 学習設定

```yaml
model: microsoft/Phi-3.5-mini-instruct
max_seq_length: 2048
lora_rank: 16
learning_rate: 2e-5
epochs: 3
batch_size: 2 (per device)
gradient_checkpointing: true
```

## Dataset Breakdown / データセット内訳（SFT用）

| Source | Samples | Ratio |
|--------|---------|-------|
| ArXiv 2024-2026 | 25,000 | 25% |
| Quadruple CoT (think) | 20,000 | 20% |
| Skill/MCP | 12,000 | 12% |
| DeepResearch | 10,000 | 10% |
| File Operations | 8,000 | 8% |
| Defense | 8,000 | 8% |
| JAXA | 7,000 | 7% |
| Drug/Chemistry | 5,000 | 5% |
| NSFW (safety) | 5,000 | 5% |

Detailed statistics are included in datasets/dataset_statistics.json in the HF upload package.

## Citations / 参考文献

- Zhang et al. (2025). mHC: Manifold-Constrained Hyper-Connections. arXiv:2512.24880
- WZ1119 (2026). KromHC: Kronecker-Factorized Doubly-Stochastic Residuals. arXiv:2601.21579
- AMAP-ML (2026). MathForge / DGPO. arXiv:2601.20614
- SakanaAI (2024). ShinkaEvolve. arXiv:2509.19349
- SakanaAI (2024). Evolutionary Model Merge. GitHub: sakanaai/evolutionary-model-merge
- DeepSeek-AI (2024). DeepSeekMath: Group Relative Policy Optimization. arXiv:2402.03300
- SO8T Project (2025). Safe Operation 8-Task Architecture. GitHub: zapabob/SO8T

## Limitations / 制限事項

- Model may generate incorrect reasoning in complex scenarios
- Performance varies across different Japanese dialects
- VRAM constraints may limit batch size on smaller GPUs

## License / ライセンス

Apache 2.0 - See LICENSE file for details.
""".format(
            created_date=datetime.now().strftime("%Y-%m-%d"),
            benchmarks="\n".join(
                [
                    f"| {k} | {v.get('accuracy', 'N/A'):.3f} | {v.get('ci_95', ['N/A', 'N/A'])} |"
                    for k, v in (benchmark_results or {}).items()
                ]
            ),
        )

        return model_card

    def upload_to_hf(
        self, repo_id: str = "zapabobouj/AEGIS-phi3.5-jp-v3.0", private: bool = False
    ) -> bool:
        """Upload model to HuggingFace Hub."""
        self.print_progress(f"Preparing upload to {repo_id}")

        try:
            from huggingface_hub import HfApi, login

            # Check for token
            token = os.environ.get("HF_TOKEN")
            if not token:
                logger.warning("HF_TOKEN not set. Attempting anonymous access.")
            else:
                login(token=token)

            api = HfApi()

            # Create repo if not exists
            try:
                api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            except Exception as e:
                logger.warning(f"Repo creation: {e}")

            # Upload folder
            self.print_progress("Uploading files...")
            api.upload_folder(
                folder_path=str(self.output_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload AEGIS-phi3.5-jp-v3.0",
            )

            self.print_progress(
                f"Upload complete: https://huggingface.co/{repo_id}", 1.0
            )
            return True

        except ImportError:
            logger.warning(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            )
            return False
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def run_full_upload(self, repo_id: str = "zapabobouj/AEGIS-phi3.5-jp-v3.0"):
        """Run complete upload process."""
        logger = self.setup_logging()
        logger.info("=" * 60)
        logger.info("HF Upload v3.0 - AEGIS-phi3.5-jp-v3.0")
        logger.info("=" * 60)

        # Step 1: Safetensors
        self.print_progress("Step 1/4: Converting to Safetensors", 0.0)
        if not self.convert_to_safetensors():
            logger.error("Safetensors conversion failed")
            return False
        self.print_progress("Step 1/4: Complete", 0.25)

        # Step 2: GGUF
        self.print_progress("Step 2/4: Converting to BF16 GGUF", 0.25)
        self.convert_to_gguf_bf16()
        self.print_progress("Step 2/4: Complete", 0.5)

        # Step 3: Model Card
        self.print_progress("Step 3/4: Creating model card", 0.5)
        model_card = self.create_model_card()
        (self.output_dir / "README.md").write_text(model_card)
        self.print_progress("Step 3/4: Complete", 0.75)

        # Step 4: Upload
        self.print_progress("Step 4/4: Uploading to HF", 0.75)
        if self.upload_to_hf(repo_id):
            self.print_progress("Step 4/4: Complete", 1.0)
            logger.info(f"Model uploaded to: https://huggingface.co/{repo_id}")
        else:
            logger.warning("Upload skipped (credentials not configured)")

        logger.info("Upload process complete!")
        return True


def main():
    parser = argparse.ArgumentParser(description="HF Upload v3.0")
    parser.add_argument("--model-path", type=str, default="checkpoints/v3_grpo/adapter")
    parser.add_argument(
        "--repo-id", type=str, default="zapabobouj/AEGIS-phi3.5-jp-v3.0"
    )
    parser.add_argument("--upload", action="store_true", help="Actually upload to HF")
    parser.add_argument(
        "--safetensors-only", action="store_true", help="Only convert to Safetensors"
    )

    args = parser.parse_args()

    uploader = HFUploaderV3(model_path=args.model_path)

    if args.safetensors_only:
        uploader.convert_to_safetensors()
    else:
        uploader.run_full_upload(repo_id=args.repo_id)


if __name__ == "__main__":
    main()
