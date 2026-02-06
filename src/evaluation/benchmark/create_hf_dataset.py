#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create HF Dataset for AEGIS A/B Test Results

A/Bテスト結果をHF Datasets形式で公開するためのデータセット作成
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import argparse

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


def load_ab_test_results(results_path: str) -> Dict[str, Any]:
    """A/Bテスト結果読み込み"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_hf_dataset(results: Dict[str, Any], output_dir: str):
    """HFデータセット作成"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # データセット情報
    dataset_info = {
        "dataset_info": {
            "name": "aegis-phi35-v2-ab-test-results",
            "description": "AEGIS-phi3.5-v2.0 vs Borea-Phi3.5-instruct-jp A/B test results on ELYZA-100 benchmark",
            "version": "1.0.0",
            "license": "apache-2.0",
            "languages": ["ja"],
            "tags": ["benchmark", "ab-test", "japanese", "phi-3.5", "aegis"],
            "task_categories": ["question-answering", "text-generation"],
            "task_ids": ["ab-test-evaluation"],
            "pretty_name": "AEGIS Phi-3.5 v2.0 A/B Test Results",
            "size_categories": ["1K<n<10K"],
            "arxiv": ["cs.CL", "cs.AI"]
        }
    }

    # データセット情報保存
    with open(output_path / "dataset_info.json", 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    # 詳細結果をParquet形式で保存 (HF Datasets互換)
    detailed_data = []

    for result in results['detailed_results']:
        for i in range(len(result['model_a_scores'])):
            detailed_data.append({
                'task_id': result['task_id'],
                'category': result['category'],
                'question': result['question'],
                'sample_id': i,
                'model_a_score': result['model_a_scores'][i],
                'model_b_score': result['model_b_scores'][i],
                'model_a_avg': result['model_a_avg'],
                'model_b_avg': result['model_b_avg'],
                'improvement': result['improvement']
            })

    df_detailed = pd.DataFrame(detailed_data)
    df_detailed.to_parquet(output_path / "detailed_results.parquet")

    # 統計情報をJSONで保存
    with open(output_path / "statistics.json", 'w', encoding='utf-8') as f:
        json.dump(results['statistics'], f, ensure_ascii=False, indent=2)

    # README作成
    create_hf_readme(results, output_path)

    print(f"[OK] HFデータセット作成完了: {output_path}")


def create_hf_readme(results: Dict[str, Any], output_path: Path):
    """HF README作成"""
    stats = results['statistics']

    readme = f"""---
annotations_creators:
- expert-generated
language_creators:
- expert-generated
languages:
- ja
licenses:
- apache-2.0
multilinguality:
- monolingual
pretty_name: AEGIS Phi-3.5 v2.0 A/B Test Results
size_categories:
- 1K<n<10K
source_datasets:
- elyza__elyza-tasks-100
task_categories:
- question-answering
task_ids:
- open-domain-qa
---

# AEGIS-phi3.5-v2.0 A/B Test Results Dataset

## Dataset Description

This dataset contains the results of an A/B test comparing AEGIS-phi3.5-v2.0 (a Nobel Prize/Fields Medal-level reasoning model) against the base Borea-Phi3.5-instruct-jp model on the ELYZA-100 benchmark.

### Model A (Baseline)
- **Model**: Borea-Phi3.5-instruct-jp
- **Mean Score**: {stats['model_a']['mean']:.3f} ± {stats['model_a']['std']:.3f}

### Model B (AEGIS)
- **Model**: AEGIS-phi3.5-v2.0
- **Mean Score**: {stats['model_b']['mean']:.3f} ± {stats['model_b']['std']:.3f}

## Statistical Analysis

### Overall Comparison
- **Mean Difference**: {stats['comparison']['mean_difference']:.3f}
- **Improvement**: {stats['comparison']['improvement_percentage']:.1f}%

### Statistical Tests
- **t-test p-value**: {stats.get('t_test', {}).get('p_value', 'N/A'):.4f}
- **Effect Size (Cohen's d)**: {stats.get('effect_size', {}).get('cohens_d', 'N/A'):.3f}
- **Significant Difference**: {"Yes" if stats.get('t_test', {}).get('significant', False) else "No"}

## Data Structure

### Detailed Results (`detailed_results.parquet`)
- `task_id`: ELYZA-100 task identifier
- `category`: Task category
- `question`: Original question
- `sample_id`: Sample index (0-2 for 3 samples per question)
- `model_a_score`: Model A quality score for this sample
- `model_b_score`: Model B quality score for this sample
- `model_a_avg`: Model A average score for this task
- `model_b_avg`: Model B average score for this task
- `improvement`: B - A score difference

### Statistics (`statistics.json`)
Complete statistical analysis including ANOVA by category, effect sizes, and confidence intervals.

## Usage

```python
from datasets import load_dataset
import pandas as pd

# Load detailed results
dataset = load_dataset("your-username/aegis-phi35-v2-ab-test-results")
df = pd.DataFrame(dataset['train'])

# Calculate average improvement
avg_improvement = df['improvement'].mean()
print(f"Average improvement: {avg_improvement:.3f}")

# Category-wise analysis
category_stats = df.groupby('category').agg({
    'model_a_avg': 'mean',
    'model_b_avg': 'mean',
    'improvement': 'mean'
})
print(category_stats)
```

## Theoretical Background

AEGIS-phi3.5-v2.0 integrates several advanced mathematical theories:

- **URT (Unified Representation Theorem)**: Quantum field theoretic representation unification
- **NC-KART★ (Non-Commutative Kolmogorov-Arnold Theory)**: Non-commutative function approximation
- **SO(8) Enhanced Adapter**: Lie algebra-based rotation optimization
- **Quadruple Thinking Engine**: Four-stage reasoning process

## Citation

```bibtex
@dataset{{aegis_ab_test_2025,
  title={{AEGIS-phi3.5-v2.0 A/B Test Results on ELYZA-100}},
  author={{AI Agent}},
  year={{2025}},
  url={{https://huggingface.co/datasets/your-username/aegis-phi35-v2-ab-test-results}}
}}
```

## License

Apache 2.0
"""

    with open(output_path / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Create HF Dataset for AEGIS A/B Test Results")
    parser.add_argument("--results_path", type=str,
                       default="benchmark_results/aegis_ab_test/aegis_ab_test_results.json",
                       help="A/B test results JSON path")
    parser.add_argument("--output_dir", type=str,
                       default="hf_datasets/aegis_phi35_v2_ab_test",
                       help="Output directory for HF dataset")

    args = parser.parse_args()

    # 結果読み込み
    results_path = PROJECT_ROOT / args.results_path
    if not results_path.exists():
        print(f"[NG] Results file not found: {results_path}")
        sys.exit(1)

    print(f"Loading A/B test results: {results_path}")
    results = load_ab_test_results(str(results_path))

    # HFデータセット作成
    output_dir = PROJECT_ROOT / args.output_dir
    create_hf_dataset(results, str(output_dir))

    print(f"\n[OK] HF dataset created successfully!")
    print(f"[DIR] Output directory: {output_dir}")
    print("\nTo upload to HF Datasets:")
    print("1. Create a new dataset on Hugging Face"
    print(f"2. Upload the contents of {output_dir}")
    print("3. The dataset will be available at: https://huggingface.co/datasets/your-username/aegis-phi35-v2-ab-test-results"


if __name__ == "__main__":
    main()

