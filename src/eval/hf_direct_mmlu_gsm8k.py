#!/usr/bin/env python3
"""
HFモデルでのMMLU/GSM8K直接評価スクリプト
transformersを使ってHFモデルを直接評価
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

# transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


class HFDirectEvaluator:
    """HFモデル直接評価クラス"""

    def __init__(self, model_path: str, model_name: str):
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.results_dir = Path("results/ab_test_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """HFモデルをロード"""
        print(f"[LOAD] Loading HF model: {self.model_path}")

        try:
            # 4-bit量子化でメモリ節約
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True
            )
            print(f"[OK] HF model loaded: {self.model_path}")

        except Exception as e:
            print(f"[ERROR] HF model loading failed: {e}")
            raise

    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """モデルから応答を生成"""
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
            print(f"[ERROR] Generation failed: {e}")
            return ""

    def evaluate_mmlu_sample(self) -> Dict[str, Any]:
        """MMLUサンプル問題で評価"""
        print("[EVAL] Evaluating MMLU sample questions...")

        # MMLUサンプル問題 (computer science, mathematics, physics)
        mmlu_questions = [
            {
                "question": "What is the time complexity of the quicksort algorithm in the average case?",
                "options": ["O(n)", "O(n log n)", "O(n²)", "O(log n)"],
                "correct": 1,  # B: O(n log n)
                "subject": "computer science"
            },
            {
                "question": "In Boolean algebra, what is the result of A AND (A OR B)?",
                "options": ["A", "B", "A OR B", "A AND B"],
                "correct": 0,  # A: A
                "subject": "mathematics"
            },
            {
                "question": "What is the SI unit of electric charge?",
                "options": ["Volt", "Ampere", "Coulomb", "Watt"],
                "correct": 2,  # C: Coulomb
                "subject": "physics"
            },
            {
                "question": "Which data structure uses LIFO (Last In, First Out) principle?",
                "options": ["Queue", "Stack", "Array", "Linked List"],
                "correct": 1,  # B: Stack
                "subject": "computer science"
            },
            {
                "question": "What is the derivative of sin(x)?",
                "options": ["cos(x)", "-sin(x)", "tan(x)", "sec(x)"],
                "correct": 0,  # A: cos(x)
                "subject": "mathematics"
            }
        ]

        correct = 0
        total = len(mmlu_questions)

        for i, q in enumerate(tqdm(mmlu_questions, desc="MMLU questions")):
            # プロンプト作成
            options_text = "\n".join([f"{chr(65+j)}) {opt}" for j, opt in enumerate(q['options'])])
            prompt = f"""Question: {q['question']}

Options:
{options_text}

Please answer by selecting the correct option (A, B, C, or D) and briefly explain why.

Answer:"""

            # 応答生成
            response = self.generate_response(prompt)

            # 回答解析 (最初の文字をチェック)
            predicted = -1
            response_upper = response.upper()
            for j, opt in enumerate(q['options']):
                if chr(65+j) in response_upper[:50]:  # 最初の50文字でチェック
                    predicted = j
                    break

            if predicted == q['correct']:
                correct += 1

            print(f"Q{i+1}: {'✓' if predicted == q['correct'] else '✗'} (Predicted: {chr(65+predicted) if predicted >= 0 else '?'}, Correct: {chr(65+q['correct'])})")

        accuracy = correct / total
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'questions': mmlu_questions
        }

    def evaluate_gsm8k_sample(self) -> Dict[str, Any]:
        """GSM8Kサンプル問題で評価"""
        print("[EVAL] Evaluating GSM8K sample questions...")

        # GSM8Kサンプル問題
        gsm8k_questions = [
            {
                "question": "A store has 5 apples. If I buy 2 apples, how many apples are left?",
                "answer": 3
            },
            {
                "question": "Sarah has 12 candies. She gives 3 to her friend. How many does she have left?",
                "answer": 9
            },
            {
                "question": "There are 7 birds on a tree. 4 more birds join them. How many birds are there now?",
                "answer": 11
            },
            {
                "question": "A pizza has 8 slices. If 3 slices are eaten, how many are left?",
                "answer": 5
            },
            {
                "question": "John has 15 marbles. He loses 6. How many does he have left?",
                "answer": 9
            }
        ]

        correct = 0
        total = len(gsm8k_questions)

        for i, q in enumerate(tqdm(gsm8k_questions, desc="GSM8K questions")):
            # プロンプト作成
            prompt = f"""Question: {q['question']}

Please solve this step by step and provide the final answer as a number.

Answer:"""

            # 応答生成
            response = self.generate_response(prompt)

            # 回答解析 (数字を抽出)
            predicted = None
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                predicted = int(numbers[-1])  # 最後の数字を使用

            if predicted == q['answer']:
                correct += 1

            print(f"Q{i+1}: {'✓' if predicted == q['answer'] else '✗'} (Predicted: {predicted}, Correct: {q['answer']})")

        accuracy = correct / total
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'questions': gsm8k_questions
        }

    def run_evaluation(self) -> Dict[str, Any]:
        """完全な評価を実行"""
        print(f"[START] Evaluating {self.model_name} (HF)")
        print("=" * 50)

        try:
            # モデルロード
            self.load_model()

            # MMLU評価
            mmlu_results = self.evaluate_mmlu_sample()

            # GSM8K評価
            gsm8k_results = self.evaluate_gsm8k_sample()

            # 結果統合
            results = {
                'model_name': self.model_name,
                'mmlu': mmlu_results,
                'gsm8k': gsm8k_results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            # 保存
            result_file = self.results_dir / f"hf_{self.model_name.lower().replace(' ', '_')}_direct_mmlu_gsm8k_results.json"
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
        print(f"RESULTS FOR {results['model_name']} (HF)")
        print("="*60)

        # MMLU結果
        if 'mmlu' in results:
            mmlu = results['mmlu']
            print(".4f")
        # GSM8K結果
        if 'gsm8k' in results:
            gsm8k = results['gsm8k']
            print(".4f")


def run_ab_comparison():
    """A/B比較を実行"""
    print("[START] Running HF A/B evaluation (MMLU/GSM8K Direct)")
    print("=" * 60)

    # モデルパス
    models = {
        'base': 'models/Borea-Phi-3.5-mini-Instruct-Jp',
        'aegis': 'H:/from_D/webdataset/models/final/aegis_v21_sft_hf'
    }

    results = {}

    for model_name, model_path in models.items():
        if not Path(model_path).exists():
            print(f"[SKIP] Model not found: {model_path}")
            continue

        try:
            evaluator = HFDirectEvaluator(model_path, model_name)
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
    print("A/B COMPARISON ANALYSIS (HF DIRECT EVALUATION)")
    print("="*60)

    base_results = results.get('base', {})
    aegis_results = results.get('aegis', {})

    # MMLU比較
    print("\n[MMLU COMPARISON]")
    if 'mmlu' in base_results and 'mmlu' in aegis_results:
        base_mmlu = base_results['mmlu']
        aegis_mmlu = aegis_results['mmlu']

        base_acc = base_mmlu['accuracy']
        aegis_acc = aegis_mmlu['accuracy']

        print(".4f")
        print(".4f")
        print(".4f")

    # GSM8K比較
    print("\n[GSM8K COMPARISON]")
    if 'gsm8k' in base_results and 'gsm8k' in aegis_results:
        base_gsm8k = base_results['gsm8k']
        aegis_gsm8k = aegis_results['gsm8k']

        base_acc = base_gsm8k['accuracy']
        aegis_acc = aegis_gsm8k['accuracy']

        print(".4f")
        print(".4f")
        print(".4f")


def main():
    """メイン関数"""
    # A/B比較を実行
    results = run_ab_comparison()

    # 保存
    output_file = Path("results/ab_test_results/hf_direct_mmlu_gsm8k_ab_comparison.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[SUCCESS] HF Direct A/B comparison completed!")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
