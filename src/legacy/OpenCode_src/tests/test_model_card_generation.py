#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデルカード生成テストスクリプト
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_model_card_generation():
    """モデルカード生成テスト"""
    print("Testing enhanced model card generation...")

    try:
        # シンプルなテストモデルカード生成
        model_path = "models/test_model"
        os.makedirs(model_path, exist_ok=True)

        # テスト用の統計データ
        benchmark_stats = {
            "gsm8k": {"mean": 77.0, "std": 1.2, "ci_95": 2.4, "cohens_d": 1.8, "p_value": 0.001},
            "math": {"mean": 43.0, "std": 2.1, "ci_95": 4.1, "cohens_d": 2.2, "p_value": 0.0001},
            "arc_challenge": {"mean": 74.0, "std": 1.8, "ci_95": 3.5, "cohens_d": 1.9, "p_value": 0.001},
            "elyza_tasks": {"mean": 83.0, "std": 1.1, "ci_95": 2.2, "cohens_d": 2.1, "p_value": 0.0001}
        }

        import time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # シンプルなモデルカード
        model_card = f"""---
language:
- en
- ja
license: apache-2.0
tags:
- so8-quadrality-inference
- mathematical-reasoning
datasets:
- gsm8k
- math
- ai2_arc
- elyza/ELYZA-tasks-100
metrics:
- accuracy
library_name: transformers
---

# AEGIS v2.5: SO8T Quadrality Inference + imatrix Protection

## Benchmark Results

| Benchmark | Score | Std Dev | 95% CI | Cohen's d | p-value |
|-----------|-------|---------|--------|-----------|---------|
| GSM8K | {benchmark_stats['gsm8k']['mean']:.1f}% | ±{benchmark_stats['gsm8k']['std']:.2f} | ±{benchmark_stats['gsm8k']['ci_95']:.2f} | {benchmark_stats['gsm8k']['cohens_d']:.2f} | {benchmark_stats['gsm8k']['p_value']:.3f} |
| MATH | {benchmark_stats['math']['mean']:.1f}% | ±{benchmark_stats['math']['std']:.2f} | ±{benchmark_stats['math']['ci_95']:.2f} | {benchmark_stats['math']['cohens_d']:.2f} | {benchmark_stats['math']['p_value']:.3f} |
| ARC-Challenge | {benchmark_stats['arc_challenge']['mean']:.1f}% | ±{benchmark_stats['arc_challenge']['std']:.2f} | ±{benchmark_stats['arc_challenge']['ci_95']:.2f} | {benchmark_stats['arc_challenge']['cohens_d']:.2f} | {benchmark_stats['arc_challenge']['p_value']:.3f} |
| ELYZA Tasks 100 | {benchmark_stats['elyza_tasks']['mean']:.1f}% | ±{benchmark_stats['elyza_tasks']['std']:.2f} | ±{benchmark_stats['elyza_tasks']['ci_95']:.2f} | {benchmark_stats['elyza_tasks']['cohens_d']:.2f} | {benchmark_stats['elyza_tasks']['p_value']:.3f} |

## Citations

```bibtex
@article{{so8t2024,
  title={{SO(8) Quadrality Inference for Advanced Language Models}},
  author={{SO8T Research Initiative}},
  journal={{arXiv preprint}},
  year={{2024}}
}}

@article{{deepseek2025,
  title={{DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning}},
  author={{DeepSeek-AI Team}},
  journal={{Nature}},
  year={{2025}}
}}
```

---
Generated: {timestamp}
"""

        readme_path = f"{model_path}/README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(model_card)

        print("SUCCESS: Model card generation successful!")
        print(f"Generated at: {readme_path}")
        print(f"Content length: {len(model_card)} characters")

        # 重要なセクションの確認
        sections_to_check = [
            "## Benchmark Results",
            "## Citations",
            "Cohen's d",
            "95% CI"
        ]

        for section in sections_to_check:
            if section in model_card:
                print(f"Found section: {section}")
            else:
                print(f"Missing section: {section}")

    except Exception as e:
        print(f"ERROR: Model card generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_card_generation()