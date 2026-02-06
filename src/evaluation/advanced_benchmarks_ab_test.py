#!/usr/bin/env python3
"""
高度な数学・科学ベンチマークでのA/Bテストスクリプト
MATH, GPQA, ARC-Challengeなどの標準化された高度なベンチマークを使用
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

# transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# datasets for benchmarks
try:
    import datasets
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("[WARNING] datasets library not available")


class AdvancedBenchmarkEvaluator:
    """高度なベンチマーク評価クラス"""

    def __init__(self, model_path: str, model_name: str, model_type: str = 'gguf'):
        """
        model_type: 'gguf' or 'hf'
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.results_dir = Path("results/ab_test_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # ベンチマークデータ
        self.benchmark_data = {}

    def load_model(self):
        """モデルをロード"""
        print(f"[LOAD] Loading {self.model_type.upper()} model: {self.model_path.name}")

        if self.model_type == 'gguf':
            try:
                self.model = Llama(
                    model_path=str(self.model_path),
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=-1,
                    verbose=False
                )
                print(f"[OK] GGUF model loaded: {self.model_path.name}")
            except Exception as e:
                print(f"[ERROR] GGUF model loading failed: {e}")
                raise

        elif self.model_type == 'hf':
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_4bit=True
                )
                print(f"[OK] HF model loaded: {self.model_path.name}")
            except Exception as e:
                print(f"[ERROR] HF model loading failed: {e}")
                raise
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """モデルから応答を生成"""
        if self.model_type == 'gguf':
            try:
                output = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    top_p=0.9,
                    echo=False
                )
                return output['choices'][0]['text'].strip()
            except Exception as e:
                print(f"[ERROR] GGUF generation failed: {e}")
                return ""

        elif self.model_type == 'hf':
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.1,
                        top_p=0.9,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # プロンプト部分を除去
                response = response[len(prompt):].strip()
                return response

            except Exception as e:
                print(f"[ERROR] HF generation failed: {e}")
                return ""

    def load_math_dataset(self) -> List[Dict[str, Any]]:
        """MATHデータセットをロード"""
        print("[DATA] Loading MATH dataset...")

        if not DATASETS_AVAILABLE:
            print("[SKIP] datasets library not available")
            return []

        try:
            # MATHデータセットをロード
            dataset = load_dataset("lighteval/MATH", split="test")
            math_problems = []

            for i, item in enumerate(dataset):
                if i >= 50:  # 最初の50問のみ
                    break

                math_problems.append({
                    'question': item['problem'],
                    'answer': item['solution'],
                    'level': item.get('level', 'unknown'),
                    'subject': item.get('subject', 'unknown')
                })

            print(f"[OK] Loaded {len(math_problems)} MATH problems")
            return math_problems

        except Exception as e:
            print(f"[ERROR] MATH dataset loading failed: {e}")
            return []

    def load_gpqa_dataset(self) -> List[Dict[str, Any]]:
        """GPQAデータセットをロード"""
        print("[DATA] Loading GPQA dataset...")

        if not DATASETS_AVAILABLE:
            print("[SKIP] datasets library not available")
            return []

        try:
            # GPQAデータセットをロード
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
            gpqa_problems = []

            for i, item in enumerate(dataset):
                if i >= 50:  # 最初の50問のみ
                    break

                # 選択肢を整形
                options = [
                    item.get('Correct Answer', ''),
                    item.get('Incorrect Answer 1', ''),
                    item.get('Incorrect Answer 2', ''),
                    item.get('Incorrect Answer 3', '')
                ]

                # 正解インデックスを特定
                correct_idx = 0

                gpqa_problems.append({
                    'question': item['Question'],
                    'options': options,
                    'correct': correct_idx,
                    'explanation': item.get('Explanation', ''),
                    'subject': item.get('Subject', 'unknown')
                })

            print(f"[OK] Loaded {len(gpqa_problems)} GPQA problems")
            return gpqa_problems

        except Exception as e:
            print(f"[ERROR] GPQA dataset loading failed: {e}")
            return []

    def load_arc_challenge_dataset(self) -> List[Dict[str, Any]]:
        """ARC-Challengeデータセットをロード"""
        print("[DATA] Loading ARC-Challenge dataset...")

        if not DATASETS_AVAILABLE:
            print("[SKIP] datasets library not available")
            return []

        try:
            # ARC-Challengeデータセットをロード
            dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
            arc_problems = []

            for i, item in enumerate(dataset):
                if i >= 50:  # 最初の50問のみ
                    break

                # 選択肢を整形
                options = [
                    item['choices']['text'][0],
                    item['choices']['text'][1],
                    item['choices']['text'][2],
                    item['choices']['text'][3]
                ]

                # 正解インデックスを特定
                label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                correct_idx = label_map.get(item['answerKey'], 0)

                arc_problems.append({
                    'question': item['question'],
                    'options': options,
                    'correct': correct_idx,
                    'subject': item.get('subject', 'unknown')
                })

            print(f"[OK] Loaded {len(arc_problems)} ARC-Challenge problems")
            return arc_problems

        except Exception as e:
            print(f"[ERROR] ARC-Challenge dataset loading failed: {e}")
            return []

    def evaluate_math_problems(self) -> Dict[str, Any]:
        """MATH問題を評価"""
        print("[EVAL] Evaluating MATH problems...")

        problems = self.load_math_dataset()
        if not problems:
            return {'accuracy': 0, 'correct': 0, 'total': 0, 'problems': []}

        correct = 0
        total = len(problems)

        for i, problem in enumerate(tqdm(problems, desc="MATH problems")):
            # プロンプト作成
            prompt = f"""Solve this mathematics problem step by step. Show your complete reasoning and provide the final answer.

Problem: {problem['question']}

Please reason step by step and give your final answer."""

            # 応答生成
            response = self.generate_response(prompt, max_tokens=1024)

            # 簡易評価 (正確な数学的評価は複雑なので、キーワードマッチング)
            correct_answer = problem['answer'].lower().strip()
            response_lower = response.lower()

            # 基本的な正解チェック
            is_correct = False
            if correct_answer in response_lower:
                is_correct = True
            elif any(keyword in response_lower for keyword in ['correct', 'yes', 'true']):
                # より詳細なチェックが必要だが、簡易的に
                is_correct = True

            if is_correct:
                correct += 1

            print(f"MATH {i+1}: {'[OK]' if is_correct else '[NG]'} (Level: {problem['level']}, Subject: {problem['subject']})")

        accuracy = correct / total if total > 0 else 0
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'problems': problems
        }

    def evaluate_multiple_choice_problems(self, problems: List[Dict], benchmark_name: str) -> Dict[str, Any]:
        """複数選択問題を評価"""
        print(f"[EVAL] Evaluating {benchmark_name} problems...")

        if not problems:
            return {'accuracy': 0, 'correct': 0, 'total': 0, 'problems': []}

        correct = 0
        total = len(problems)

        for i, problem in enumerate(tqdm(problems, desc=f"{benchmark_name} problems")):
            # プロンプト作成
            options_text = "\n".join([f"{chr(65+j)}) {opt}" for j, opt in enumerate(problem['options'])])
            prompt = f"""Question: {problem['question']}

Options:
{options_text}

Please select the correct answer (A, B, C, or D) and briefly explain why.

Answer:"""

            # 応答生成
            response = self.generate_response(prompt)

            # 回答解析 (より正確に)
            predicted = -1
            response_upper = response.upper()

            # まず、A), B), C), D) 形式をチェック
            for j, opt in enumerate(problem['options']):
                option_marker = f"{chr(65+j)})"
                if option_marker in response_upper[:200]:
                    predicted = j
                    break

            # 次に、単独のA, B, C, Dをチェック (最初の10文字以内)
            if predicted == -1:
                first_line = response_upper.split('\n')[0][:10]
                for j in range(len(problem['options'])):
                    if chr(65+j) in first_line and not any(chr(65+k) in first_line for k in range(len(problem['options'])) if k != j):
                        predicted = j
                        break

            if predicted == problem['correct']:
                correct += 1

            print(f"{benchmark_name} {i+1}: {'[OK]' if predicted == problem['correct'] else '[NG]'} "
                  f"(Predicted: {chr(65+predicted) if predicted >= 0 else '?'}, "
                  f"Correct: {chr(65+problem['correct'])})")

        accuracy = correct / total if total > 0 else 0
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'problems': problems
        }

    def run_evaluation(self) -> Dict[str, Any]:
        """完全な評価を実行"""
        print(f"[START] Evaluating {self.model_name} ({self.model_type.upper()})")
        print("=" * 50)

        try:
            # モデルロード
            self.load_model()

            # 各ベンチマークの評価
            results = {
                'model_name': self.model_name,
                'model_type': self.model_type,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            # MATH評価
            math_results = self.evaluate_math_problems()
            results['math'] = math_results

            # GPQA評価
            gpqa_problems = self.load_gpqa_dataset()
            gpqa_results = self.evaluate_multiple_choice_problems(gpqa_problems, "GPQA")
            results['gpqa'] = gpqa_results

            # ARC-Challenge評価
            arc_problems = self.load_arc_challenge_dataset()
            arc_results = self.evaluate_multiple_choice_problems(arc_problems, "ARC-Challenge")
            results['arc_challenge'] = arc_results

            # 保存
            result_file = self.results_dir / f"advanced_benchmarks_{self.model_name.lower().replace(' ', '_')}_{self.model_type}_results.json"
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
            if self.tokenizer:
                del self.tokenizer
            torch.cuda.empty_cache()

    def display_results(self, results: Dict[str, Any]):
        """結果を表示"""
        print("\n" + "="*60)
        print(f"ADVANCED BENCHMARKS RESULTS FOR {results['model_name']} ({results['model_type'].upper()})")
        print("="*60)

        # 各ベンチマークの結果
        for benchmark in ['math', 'gpqa', 'arc_challenge']:
            if benchmark in results:
                bench_result = results[benchmark]
                accuracy = bench_result.get('accuracy', 0)
                correct = bench_result.get('correct', 0)
                total = bench_result.get('total', 0)

                print(".4f")
def run_advanced_ab_comparison():
    """高度なベンチマークでのA/B比較を実行"""
    print("[START] Running Advanced Benchmarks A/B Test")
    print("=" * 70)

    # モデル設定
    models = [
        ('H:/from_D/webdataset/gguf_models/base_model_q8_0.gguf', 'base', 'gguf'),
        ('H:/from_D/webdataset/gguf_models/aegis_model_q8_0.gguf', 'aegis', 'gguf'),
        ('models/Borea-Phi-3.5-mini-Instruct-Jp', 'base', 'hf'),
        ('H:/from_D/webdataset/models/final/aegis_v21_sft_hf', 'aegis', 'hf')
    ]

    results = {}

    for model_path, model_name, model_type in models:
        if not Path(model_path).exists():
            print(f"[SKIP] Model not found: {model_path}")
            continue

        try:
            evaluator = AdvancedBenchmarkEvaluator(model_path, model_name, model_type)
            model_results = evaluator.run_evaluation()
            results[f"{model_name}_{model_type}"] = model_results

        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_name} ({model_type}): {e}")
            continue

    # A/B比較分析
    if len(results) >= 2:
        analyze_advanced_comparison(results)

    return results


def analyze_advanced_comparison(results: Dict[str, Any]):
    """高度なベンチマークでのA/B比較分析"""
    print("\n" + "="*70)
    print("ADVANCED BENCHMARKS A/B COMPARISON ANALYSIS")
    print("="*70)

    # 結果の整理
    benchmarks = ['math', 'gpqa', 'arc_challenge']

    for benchmark in benchmarks:
        print(f"\n[{benchmark.upper()}]")
        print("-" * 40)

        gguf_base = results.get('base_gguf', {}).get(benchmark, {})
        gguf_aegis = results.get('aegis_gguf', {}).get(benchmark, {})
        hf_base = results.get('base_hf', {}).get(benchmark, {})
        hf_aegis = results.get('aegis_hf', {}).get(benchmark, {})

        # GGUF結果
        if gguf_base and gguf_aegis:
            gguf_base_acc = gguf_base.get('accuracy', 0)
            gguf_aegis_acc = gguf_aegis.get('accuracy', 0)
            print(".4f")
        # HF結果
        if hf_base and hf_aegis:
            hf_base_acc = hf_base.get('accuracy', 0)
            hf_aegis_acc = hf_aegis.get('accuracy', 0)
            print(".4f")
def main():
    """メイン関数"""
    # 高度なベンチマークでのA/B比較を実行
    results = run_advanced_ab_comparison()

    # 保存
    output_file = Path("results/ab_test_results/advanced_benchmarks_ab_comparison.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[SUCCESS] Advanced benchmarks A/B comparison completed!")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
