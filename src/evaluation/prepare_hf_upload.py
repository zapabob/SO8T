#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HFアップロード準備スクリプト

A/Bテスト結果をHugging Face Hubアップロード用に整形
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# tqdm for progress bars
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class HFUploadPreparer:
    """HFアップロード準備クラス"""

    def __init__(self, results_dir: str = "results/ab_test_results",
                 upload_dir: str = "hf_upload_package"):
        self.results_dir = Path(results_dir)
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # サブディレクトリ作成
        self.models_dir = self.upload_dir / "models"
        self.results_dir_upload = self.upload_dir / "evaluation_results"
        self.dataset_dir = self.upload_dir / "datasets"
        self.stats_dir = self.upload_dir / "statistics"

        for dir_path in [self.models_dir, self.results_dir_upload,
                        self.dataset_dir, self.stats_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def copy_gguf_models(self):
        """GGUFモデルファイルをコピー"""
        print("📦 Copying GGUF model files...")

        # baselineモデル
        baseline_src = Path("D:/webdataset/gguf_models/baseline_phi35_bf16")
        if baseline_src.exists():
            baseline_dst = self.models_dir / "baseline_phi35_bf16"
            shutil.copytree(baseline_src, baseline_dst, dirs_exist_ok=True)
            print(f"[OK] Copied baseline model to {baseline_dst}")

        # AEGISモデル
        aegis_src = Path("D:/webdataset/gguf_models/aegis_phi35_so8t")
        if aegis_src.exists():
            aegis_dst = self.models_dir / "aegis_phi35_so8t"
            shutil.copytree(aegis_src, aegis_dst, dirs_exist_ok=True)
            print(f"[OK] Copied AEGIS model to {aegis_dst}")

    def copy_evaluation_results(self):
        """評価結果をコピー"""
        print("[STATS] Copying evaluation results...")

        # A/Bテスト結果
        ab_results = list(self.results_dir.glob("ab_test_results_final_*.json"))
        if ab_results:
            latest_result = max(ab_results, key=lambda x: x.stat().st_mtime)
            shutil.copy2(latest_result, self.results_dir_upload / "ab_test_results.json")
            print(f"[OK] Copied A/B test results: {latest_result.name}")

        # 統計分析結果
        stats_dir = self.results_dir / "statistics"
        if stats_dir.exists():
            shutil.copytree(stats_dir, self.stats_dir, dirs_exist_ok=True)
            print("[OK] Copied statistical analysis results")

        # プロット
        plots_dir = self.results_dir / "plots"
        if plots_dir.exists():
            plots_dst = self.results_dir_upload / "plots"
            shutil.copytree(plots_dir, plots_dst, dirs_exist_ok=True)
            print("[OK] Copied evaluation plots")

    def copy_datasets(self):
        """使用したデータセットをコピー"""
        print("📚 Copying evaluation datasets...")

        # AEGISデータセット
        aegis_dataset = Path("data/aegis_high_quality_dataset.jsonl")
        if aegis_dataset.exists():
            shutil.copy2(aegis_dataset, self.dataset_dir / "aegis_training_dataset.jsonl")
            print("[OK] Copied AEGIS training dataset")

        # 統計情報
        stats_file = Path("data/aegis_high_quality/dataset_statistics.json")
        if stats_file.exists():
            shutil.copy2(stats_file, self.dataset_dir / "dataset_statistics.json")
            print("[OK] Copied dataset statistics")

        # ELYZA-100
        elyza_dataset = Path("data/evaluation/elyza_100.jsonl")
        if elyza_dataset.exists():
            shutil.copy2(elyza_dataset, self.dataset_dir / "elyza_100_evaluation.jsonl")
            print("[OK] Copied ELYZA-100 evaluation dataset")

    def create_readme_and_metadata(self):
        """READMEとメタデータファイル作成"""
        print("[NOTE] Creating README and metadata files...")

        # 統計結果読み込み
        stats_file = self.stats_dir / "statistical_analysis_results.json"
        overall_stats = {}
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats_data = json.load(f)
                overall_stats = stats_data.get("overall_comparison", {})

        # README作成
        readme_content = f"""# AEGIS vs Baseline A/B Test Results

This repository contains the complete results of an A/B test comparing the AEGIS model (with SO(8) NKAT theory and high-quality training data) against a baseline Phi-3.5 model.

## Overview

- **Baseline Model**: Microsoft Phi-3.5-mini-instruct (BF16 GGUF)
- **AEGIS Model**: Phi-3.5 with SO(8) residual adapters and NKAT theory, trained on high-quality datasets
- **Evaluation**: llama.cpp.python with ELYZA-100 and other benchmark tasks

## Key Results

### Overall Performance
- **Baseline Accuracy**: {overall_stats.get('baseline_mean', 0):.4f}
- **AEGIS Accuracy**: {overall_stats.get('aegis_mean', 0):.4f}
- **Improvement**: {overall_stats.get('improvement', 0):.4f} ({overall_stats.get('improvement', 0) * 100:.2f}%)
- **Effect Size**: {overall_stats.get('effect_size_cohen_d', 0):.4f} ({overall_stats.get('effect_size_interpretation', 'unknown')})
- **Statistical Significance**: p = {overall_stats.get('p_value', 1.0):.6f}

### AEGIS Training Data Composition
- Nobel Prize/Fields Medal level mathematics and science: {self.get_dataset_stats('mathematics_nobel_fields', 0)}
- Physics Nobel Prize content: {self.get_dataset_stats('physics_nobel', 0)}
- Chemistry Nobel Prize content: {self.get_dataset_stats('chemistry_nobel', 0)}
- Biology Nobel Prize content: {self.get_dataset_stats('biology_nobel', 0)}
- Arxiv top 20% cited papers: {self.get_dataset_stats('arxiv_top_cited', 0)}
- NSFW drug detection (safety-focused): {self.get_dataset_stats('nsfw_drug_detection', 0)}
- NKAT thinking/reasoning patterns: {self.get_dataset_stats('thinking_reasoning', 0)}

## Files Structure

```
{self.upload_dir.name}/
├── models/
│   ├── baseline_phi35_bf16/
│   └── aegis_phi35_so8t/
├── evaluation_results/
│   ├── ab_test_results.json
│   └── plots/
├── datasets/
│   ├── aegis_training_dataset.jsonl
│   ├── elyza_100_evaluation.jsonl
│   └── dataset_statistics.json
└── statistics/
    ├── statistical_analysis_results.json
    └── statistical_analysis_report.md
```

## Usage

### Model Inference
```python
from llama_cpp import Llama

# Load AEGIS model
llm = Llama(model_path="models/aegis_phi35_so8t/aegis_phi35_so8t_Q8_0.gguf")

# Generate response
response = llm("Explain the concept of SO(8) rotation groups in neural networks.")
```

### Statistical Analysis
See `statistics/statistical_analysis_report.md` for detailed statistical analysis including:
- ANOVA results for few-shot learning effects
- Effect sizes for each evaluation task
- p-values and confidence intervals
- Error bar plots

## Technical Details

### AEGIS Architecture
- **Base Model**: Microsoft Phi-3.5-mini-instruct
- **Adapters**: SO(8) residual adapters with NKAT theory
- **Training**: RLPO (Reinforcement Learning with Policy Optimization)
- **Precision**: Mixed precision (FP32 for critical calculations, FP16 for main computation)

### Evaluation Methodology
- **Framework**: llama.cpp.python for GGUF model inference
- **Tasks**: ELYZA-100 (Japanese), ARC-Challenge, HellaSwag, TruthfulQA, Winogrande, GSM8K
- **Few-shot**: 0-shot, 5-shot, 10-shot evaluation
- **Metrics**: Exact match accuracy, statistical significance testing

### Statistical Analysis
- **Tests**: Student's t-test, ANOVA, Cohen's d effect size
- **Significance Level**: α = 0.05
- **Error Bars**: Standard error of mean
- **Visualization**: Matplotlib/Seaborn plots with confidence intervals

## Citation

If you use these results in your research, please cite:

```bibtex
@misc{{aegis_ab_test_2024,
  title={{AEGIS vs Baseline A/B Test Results: SO(8) NKAT Theory Evaluation}},
  author={{AI Assistant}},
  year={{2024}},
  note={{Complete evaluation results with statistical analysis}}
}}
```

## License

This evaluation results package is released under the MIT License. The models and datasets included may have their own licenses - please check individual files for licensing information.
"""

        readme_file = self.upload_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        # metadata.json作成
        metadata = {
            "name": "aegis-ab-test-results",
            "version": "1.0.0",
            "description": "Complete A/B test results comparing AEGIS (SO(8) NKAT) vs Baseline Phi-3.5 models",
            "license": "mit",
            "tags": [
                "llm-evaluation",
                "ab-testing",
                "statistical-analysis",
                "llama-cpp",
                "japanese-evaluation",
                "mathematical-reasoning",
                "scientific-reasoning"
            ],
            "key_metrics": {
                "baseline_accuracy": overall_stats.get("baseline_mean", 0),
                "aegis_accuracy": overall_stats.get("aegis_mean", 0),
                "improvement": overall_stats.get("improvement", 0),
                "effect_size": overall_stats.get("effect_size_cohen_d", 0),
                "p_value": overall_stats.get("p_value", 1.0),
                "statistically_significant": overall_stats.get("p_value", 1.0) < 0.05
            },
            "models": {
                "baseline": {
                    "name": "baseline_phi35_bf16",
                    "architecture": "Phi-3.5-mini-instruct",
                    "quantization": "BF16",
                    "parameters": "3.8B"
                },
                "aegis": {
                    "name": "aegis_phi35_so8t",
                    "architecture": "Phi-3.5-mini-instruct + SO(8) NKAT adapters",
                    "quantization": "Q8_0",
                    "parameters": "3.8B + adapters",
                    "training_data": "High-quality scientific/mathematical + Arxiv top 20% + safety-focused NSFW"
                }
            },
            "evaluation": {
                "framework": "llama.cpp.python",
                "tasks": ["elyza_100", "arc_challenge", "hellaswag", "truthfulqa_mc2", "winogrande", "gsm8k"],
                "fewshot_settings": [0, 5, 10],
                "metrics": ["exact_match", "statistical_significance", "effect_size", "anova"]
            },
            "created_at": datetime.now().isoformat(),
            "upload_ready": True
        }

        metadata_file = self.upload_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print("[OK] Created README.md and metadata.json")

    def get_dataset_stats(self, category: str, default: int = 0) -> int:
        """データセット統計取得"""
        stats_file = Path("data/aegis_high_quality/dataset_statistics.json")
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    return stats.get(category, default)
            except:
                pass
        return default

    def create_upload_archive(self) -> Path:
        """アップロード用アーカイブ作成"""
        print("📦 Creating upload archive...")

        import zipfile

        archive_name = f"aegis_ab_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        archive_path = self.upload_dir.parent / archive_name

        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.upload_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.upload_dir.parent)
                    zipf.write(file_path, arcname)

        print(f"[OK] Created upload archive: {archive_path}")
        print(f"📏 Archive size: {archive_path.stat().st_size / (1024*1024):.2f} MB")

        return archive_path

    def run_preparation(self):
        """アップロード準備実行"""
        print("[START] Preparing HF upload package")
        print("=" * 50)

        try:
            # 各コンポーネントコピー
            self.copy_gguf_models()
            self.copy_evaluation_results()
            self.copy_datasets()
            self.create_readme_and_metadata()

            # アーカイブ作成
            archive_path = self.create_upload_archive()

            print("\n[DONE] HF upload preparation completed!")
            print(f"📦 Upload package: {self.upload_dir}")
            print(f"📦 Archive: {archive_path}")
            print("")
            print("📤 Ready for HF Hub upload!")
            print("   Use: huggingface-cli upload <username>/aegis-ab-test-results")
            print(f"   Local path: {self.upload_dir}")
            return True

        except Exception as e:
            # Avoid encoding issues (e.g., cp932) by encoding errors if needed
            try:
                print(f"[NG] Preparation failed: {e}")
            except UnicodeEncodeError:
                print("[NG] Preparation failed: <UnicodeEncodeError (cp932等)>")
            return False

def main():
    parser = argparse.ArgumentParser(description="Prepare A/B Test Results for HF Upload")
    parser.add_argument("--results_dir", type=str, default="results/ab_test_results",
                       help="Directory containing A/B test results")
    parser.add_argument("--upload_dir", type=str, default="hf_upload_package",
                       help="Directory for HF upload package")
    parser.add_argument("--create_archive", action="store_true",
                       help="Create ZIP archive for upload")

    args = parser.parse_args()

    preparer = HFUploadPreparer(args.results_dir, args.upload_dir)
    success = preparer.run_preparation()

    if success:
        print("\n[OK] HF upload preparation completed!")
        print("[TARGET] Next steps:")
        print("   1. Review the upload package in hf_upload_package/")
        print("   2. Create HF repository: huggingface-cli repo create aegis-ab-test-results")
        print("   3. Upload: huggingface-cli upload <username>/aegis-ab-test-results hf_upload_package/")
    else:
        print("\n[NG] HF upload preparation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
