#!/usr/bin/env python3
"""
GGUFモデルでのMMLU/GSM8K評価スクリプト
llama.cpp.python + lm_evalを使ってGGUFモデルを評価
"""

import os
import json
import time
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats

# llama.cpp.python
from llama_cpp import Llama
import lm_eval
from lm_eval import evaluator, tasks


class GGUFEvaluator:
    """GGUFモデル評価クラス"""

    def __init__(self, model_path: str, model_name: str):
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.model = None
        self.results_dir = Path("results/ab_test_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """GGUFモデルをロード"""
        print(f"[LOAD] Loading GGUF model: {self.model_path.name}")

        try:
            # GPU使用で高速化
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=4096,  # MMLU/GSM8K用にコンテキスト長を増やす
                n_threads=4,
                n_gpu_layers=-1,  # GPU使用
                verbose=False
            )
            print(f"[OK] Model loaded: {self.model_path.name}")

        except Exception as e:
            print(f"[ERROR] Model loading failed: {e}")
            raise

    def evaluate_mmlu(self) -> Dict[str, Any]:
        """MMLU評価を実行"""
        print("[EVAL] Starting MMLU evaluation...")

        try:
            # MMLUタスクを取得
            mmlu_task = tasks.get_task("mmlu", "hendrycksTest")

            # 評価を実行 (few-shot: 5)
            results = evaluator.evaluate(
                model=self.model,
                tasks=["mmlu"],
                num_fewshot=5,
                batch_size=1,
                device="cuda" if torch.cuda.is_available() else "cpu",
                use_cache=False
            )

            print("[OK] MMLU evaluation completed")
            return results

        except Exception as e:
            print(f"[ERROR] MMLU evaluation failed: {e}")
            return {}

    def evaluate_gsm8k(self) -> Dict[str, Any]:
        """GSM8K評価を実行"""
        print("[EVAL] Starting GSM8K evaluation...")

        try:
            # GSM8Kタスクを取得
            gsm8k_task = tasks.get_task("gsm8k", "main")

            # 評価を実行 (few-shot: 8)
            results = evaluator.evaluate(
                model=self.model,
                tasks=["gsm8k"],
                num_fewshot=8,
                batch_size=1,
                device="cuda" if torch.cuda.is_available() else "cpu",
                use_cache=False
            )

            print("[OK] GSM8K evaluation completed")
            return results

        except Exception as e:
            print(f"[ERROR] GSM8K evaluation failed: {e}")
            return {}

    def run_evaluation(self) -> Dict[str, Any]:
        """完全な評価を実行"""
        print(f"[START] Evaluating {self.model_name}")
        print("=" * 50)

        try:
            # モデルロード
            self.load_model()

            # MMLU評価
            mmlu_results = self.evaluate_mmlu()

            # GSM8K評価
            gsm8k_results = self.evaluate_gsm8k()

            # 結果統合
            results = {
                'model_name': self.model_name,
                'mmlu': mmlu_results,
                'gsm8k': gsm8k_results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            # 保存
            result_file = self.results_dir / f"gguf_{self.model_name.lower().replace(' ', '_')}_mmlu_gsm8k_results.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            print(f"[SUCCESS] Results saved to: {result_file}")

            # 結果表示
            self.display_results(results)

            return results

        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            raise

        finally:
            # モデル解放
            if self.model:
                del self.model
            torch.cuda.empty_cache()

    def display_results(self, results: Dict[str, Any]):
        """結果を表示"""
        print("\n" + "="*60)
        print(f"RESULTS FOR {results['model_name']}")
        print("="*60)

        # MMLU結果
        if 'mmlu' in results and results['mmlu']:
            mmlu = results['mmlu']
            print("\n[MMLU]")
            for key, value in mmlu.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        print(f"  {key}.{sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")

        # GSM8K結果
        if 'gsm8k' in results and results['gsm8k']:
            gsm8k = results['gsm8k']
            print("\n[GSM8K]")
            for key, value in gsm8k.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        print(f"  {key}.{sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")


def run_ab_comparison():
    """A/B比較を実行"""
    print("[START] Running GGUF A/B evaluation (MMLU/GSM8K)")
    print("=" * 60)

    # モデルパス
    models = {
        'base': 'H:/from_D/webdataset/gguf_models/base_model_q8_0.gguf',
        'aegis': 'H:/from_D/webdataset/gguf_models/aegis_model_q8_0.gguf'
    }

    results = {}

    for model_name, model_path in models.items():
        if not Path(model_path).exists():
            print(f"[SKIP] Model not found: {model_path}")
            continue

        try:
            evaluator = GGUFEvaluator(model_path, model_name)
            model_results = evaluator.run_evaluation()
            results[model_name] = model_results

        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_name}: {e}")
            continue

    # A/B比較分析
    if len(results) == 2:
        analyze_comparison(results)

    return results


def analyze_comparison(results: Dict[str, Any]):
    """A/B比較分析"""
    print("\n" + "="*60)
    print("A/B COMPARISON ANALYSIS")
    print("="*60)

    base_results = results.get('base', {})
    aegis_results = results.get('aegis', {})

    # MMLU比較
    print("\n[MMLU COMPARISON]")
    if 'mmlu' in base_results and 'mmlu' in aegis_results:
        base_mmlu = base_results['mmlu']
        aegis_mmlu = aegis_results['mmlu']

        # acc,noneを取得
        base_acc = base_mmlu.get('acc,none', 0)
        aegis_acc = aegis_mmlu.get('acc,none', 0)

        print(".4f")
        print(".4f")
        print(".4f")

    # GSM8K比較
    print("\n[GSM8K COMPARISON]")
    if 'gsm8k' in base_results and 'gsm8k' in aegis_results:
        base_gsm8k = base_results['gsm8k']
        aegis_gsm8k = aegis_results['gsm8k']

        # acc,noneを取得
        base_acc = base_gsm8k.get('acc,none', 0)
        aegis_acc = aegis_gsm8k.get('acc,none', 0)

        print(".4f")
        print(".4f")
        print(".4f")


def main():
    """メイン関数"""
    # A/B比較を実行
    results = run_ab_comparison()

    # 保存
    output_file = Path("results/ab_test_results/gguf_mmlu_gsm8k_ab_comparison.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[SUCCESS] A/B comparison completed!")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
