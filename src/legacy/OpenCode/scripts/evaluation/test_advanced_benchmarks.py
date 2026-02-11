#!/usr/bin/env python3
"""
高度なベンチマークテストスクリプト (GGUFのみ)
"""

import json
import torch
from pathlib import Path
from llama_cpp import Llama

def test_math_benchmark():
    """MATHベンチマークをテスト"""
    print("[TEST] Testing MATH benchmark with GGUF model...")

    # GGUFモデルロード
    model_path = "H:/from_D/webdataset/gguf_models/aegis_model_q8_0.gguf"
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    try:
        model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=-1,
            verbose=False
        )
        print("[OK] Model loaded")

        # サンプルMATH問題
        math_problem = {
            "problem": "Solve for x: 2x + 3 = 7",
            "solution": "x = 2"
        }

        prompt = f"""Solve this mathematics problem step by step. Show your complete reasoning and provide the final answer.

Problem: {math_problem['problem']}

Please reason step by step and give your final answer."""

        print(f"[PROMPT] {prompt}")

        # 応答生成
        response = model(
            prompt,
            max_tokens=256,
            temperature=0.1,
            top_p=0.9,
            echo=False
        )['choices'][0]['text'].strip()

        print(f"[RESPONSE] {response}")

        # 正解チェック
        correct_answer = math_problem['solution'].lower().strip()
        response_lower = response.lower()

        is_correct = correct_answer in response_lower
        print(f"[RESULT] {'✓ Correct' if is_correct else '✗ Incorrect'}")
        print(f"Expected: {correct_answer}")

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

def test_multiple_choice_benchmark():
    """複数選択ベンチマークをテスト"""
    print("[TEST] Testing multiple choice benchmark...")

    model_path = "H:/from_D/webdataset/gguf_models/aegis_model_q8_0.gguf"
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    try:
        model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=-1,
            verbose=False
        )

        # サンプルGPQA問題
        gpqa_problem = {
            "question": "What is the capital of France?",
            "options": ["London", "Berlin", "Paris", "Madrid"],
            "correct": 2  # Paris
        }

        options_text = "\n".join([f"{chr(65+j)}) {opt}" for j, opt in enumerate(gpqa_problem['options'])])
        prompt = f"""Question: {gpqa_problem['question']}

Options:
{options_text}

Please select the correct answer (A, B, C, or D) and briefly explain why.

Answer:"""

        print(f"[PROMPT] {prompt}")

        # 応答生成
        response = model(
            prompt,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
            echo=False
        )['choices'][0]['text'].strip()

        print(f"[RESPONSE] {response}")

        # 回答解析
        predicted = -1
        response_upper = response.upper()
        for j, opt in enumerate(gpqa_problem['options']):
            if chr(65+j) in response_upper[:50]:
                predicted = j
                break

        correct_answer = gpqa_problem['correct']
        is_correct = predicted == correct_answer

        print(f"[RESULT] {'✓ Correct' if is_correct else '✗ Incorrect'}")
        print(f"Predicted: {chr(65+predicted) if predicted >= 0 else '?'} ({gpqa_problem['options'][predicted] if predicted >= 0 else 'Unknown'})")
        print(f"Expected: {chr(65+correct_answer)} ({gpqa_problem['options'][correct_answer]})")

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Testing Advanced Benchmarks")
    print("=" * 40)

    test_math_benchmark()
    print("\n" + "-" * 40)
    test_multiple_choice_benchmark()

    print("\n[COMPLETE] Test finished")
