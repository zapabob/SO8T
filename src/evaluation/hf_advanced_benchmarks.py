#!/usr/bin/env python3
"""
HFモデルでの高度なベンチマーク評価スクリプト
MATH, GPQA, ARC-ChallengeをHFモデルで評価
"""

import json
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_sample_problems():
    """サンプル問題を定義"""
    return {
        'math': [
            {
                "problem": "Solve for x: 2x + 3 = 7",
                "solution": "x = 2"
            },
            {
                "problem": "If 3x - 2 = 10, what is x?",
                "solution": "x = 4"
            },
            {
                "problem": "Simplify: (2 + 3) × 4 - 6 ÷ 2",
                "solution": "20"
            },
            {
                "problem": "What is the square root of 144?",
                "solution": "12"
            },
            {
                "problem": "If a triangle has angles 30°, 60°, and 90°, what type of triangle is it?",
                "solution": "right triangle"
            }
        ],
        'gpqa': [
            {
                "question": "What is the capital of France?",
                "options": ["London", "Berlin", "Paris", "Madrid"],
                "correct": 2
            },
            {
                "question": "Which planet is known as the Red Planet?",
                "options": ["Venus", "Mars", "Jupiter", "Saturn"],
                "correct": 1
            },
            {
                "question": "What is the chemical symbol for water?",
                "options": ["CO2", "H2O", "O2", "N2"],
                "correct": 1
            },
            {
                "question": "Which of these is NOT a programming paradigm?",
                "options": ["Object-oriented", "Functional", "Procedural", "Algorithmic"],
                "correct": 3
            },
            {
                "question": "What is the powerhouse of the cell?",
                "options": ["Nucleus", "Mitochondria", "Ribosome", "Endoplasmic reticulum"],
                "correct": 1
            }
        ],
        'arc_challenge': [
            {
                "question": "What happens to the density of a substance when it is heated?",
                "options": ["Increases", "Decreases", "Stays the same", "Depends on the substance"],
                "correct": 3
            },
            {
                "question": "Which of the following is NOT a renewable energy source?",
                "options": ["Solar", "Wind", "Coal", "Hydroelectric"],
                "correct": 2
            },
            {
                "question": "What is the main function of the mitochondria in a cell?",
                "options": ["Protein synthesis", "Energy production", "Waste removal", "Cell division"],
                "correct": 1
            },
            {
                "question": "Which gas makes up the majority of Earth's atmosphere?",
                "options": ["Oxygen", "Carbon dioxide", "Nitrogen", "Argon"],
                "correct": 2
            },
            {
                "question": "What type of rock is formed from cooled lava?",
                "options": ["Sedimentary", "Metamorphic", "Igneous", "Fossil"],
                "correct": 2
            }
        ]
    }

def evaluate_hf_model(model_path, model_name, problems):
    """HFモデルを評価"""
    print(f"[EVAL] Evaluating {model_name} (HF)")
    print("=" * 40)

    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        return {}

    try:
        # モデルロード (4-bit量子化)
        print("[LOAD] Loading HF model with 4-bit quantization...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        print("[OK] HF model loaded")

        results = {}

        # MATH評価
        print("[MATH] Evaluating MATH problems...")
        math_correct = 0
        for i, problem in enumerate(problems['math']):
            prompt = f"""Solve this mathematics problem step by step. Show your complete reasoning and provide the final answer.

Problem: {problem['problem']}

Please reason step by step and give your final answer."""

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            # 正解チェック
            correct_answer = problem['solution'].lower().strip()
            response_lower = response.lower()
            is_correct = correct_answer in response_lower

            if is_correct:
                math_correct += 1

            print(f"  MATH {i+1}: {'[OK]' if is_correct else '[NG]'}")

        # GPQA評価
        print("[GPQA] Evaluating GPQA problems...")
        gpqa_correct = 0
        for i, problem in enumerate(problems['gpqa']):
            options_text = "\n".join([f"{chr(65+j)}) {opt}" for j, opt in enumerate(problem['options'])])
            prompt = f"""Question: {problem['question']}

Options:
{options_text}

Please select the correct answer (A, B, C, or D) and briefly explain why.

Answer:"""

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            # 回答解析
            predicted = -1
            response_upper = response.upper()

            # A), B), C), D) 形式をチェック
            for j, opt in enumerate(problem['options']):
                option_marker = f"{chr(65+j)})"
                if option_marker in response_upper[:200]:
                    predicted = j
                    break

            # 単独のA, B, C, Dをチェック
            if predicted == -1:
                first_line = response_upper.split('\n')[0][:10]
                for j in range(len(problem['options'])):
                    if chr(65+j) in first_line and not any(chr(65+k) in first_line for k in range(len(problem['options'])) if k != j):
                        predicted = j
                        break

            is_correct = predicted == problem['correct']
            if is_correct:
                gpqa_correct += 1

            print(f"  GPQA {i+1}: {'[OK]' if is_correct else '[NG]'} (Predicted: {chr(65+predicted) if predicted >= 0 else '?'}, Correct: {chr(65+problem['correct'])})")

        # ARC-Challenge評価
        print("[ARC] Evaluating ARC-Challenge problems...")
        arc_correct = 0
        for i, problem in enumerate(problems['arc_challenge']):
            options_text = "\n".join([f"{chr(65+j)}) {opt}" for j, opt in enumerate(problem['options'])])
            prompt = f"""Question: {problem['question']}

Options:
{options_text}

Please select the correct answer (A, B, C, or D) and briefly explain why.

Answer:"""

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            # 回答解析 (GPQAと同じ)
            predicted = -1
            response_upper = response.upper()

            for j, opt in enumerate(problem['options']):
                option_marker = f"{chr(65+j)})"
                if option_marker in response_upper[:200]:
                    predicted = j
                    break

            if predicted == -1:
                first_line = response_upper.split('\n')[0][:10]
                for j in range(len(problem['options'])):
                    if chr(65+j) in first_line and not any(chr(65+k) in first_line for k in range(len(problem['options'])) if k != j):
                        predicted = j
                        break

            is_correct = predicted == problem['correct']
            if is_correct:
                arc_correct += 1

            print(f"  ARC {i+1}: {'[OK]' if is_correct else '[NG]'} (Predicted: {chr(65+predicted) if predicted >= 0 else '?'}, Correct: {chr(65+problem['correct'])})")

        # 結果集計
        results = {
            'model_name': model_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'math': {
                'accuracy': math_correct / len(problems['math']),
                'correct': math_correct,
                'total': len(problems['math'])
            },
            'gpqa': {
                'accuracy': gpqa_correct / len(problems['gpqa']),
                'correct': gpqa_correct,
                'total': len(problems['gpqa'])
            },
            'arc_challenge': {
                'accuracy': arc_correct / len(problems['arc_challenge']),
                'correct': arc_correct,
                'total': len(problems['arc_challenge'])
            }
        }

        return results

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

    finally:
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()

def main():
    """メイン関数"""
    print("[START] HF Advanced Benchmarks Evaluation")
    print("=" * 50)

    # サンプル問題ロード
    problems = load_sample_problems()

    # HFモデル設定
    models = [
        ('models/Borea-Phi-3.5-mini-Instruct-Jp', 'base_hf'),
        ('H:/from_D/webdataset/models/final/aegis_v21_sft_hf', 'aegis_hf')
    ]

    results = {}

    for model_path, model_name in models:
        result = evaluate_hf_model(model_path, model_name, problems)
        if result:
            results[model_name] = result

            # 結果表示
            print(f"\n[RESULTS] {model_name.upper()}")
            for benchmark in ['math', 'gpqa', 'arc_challenge']:
                if benchmark in result:
                    acc = result[benchmark]['accuracy']
                    correct = result[benchmark]['correct']
                    total = result[benchmark]['total']
                    print(".4f")
    # A/B比較
    if len(results) == 2:
        print(f"\n[COMPARISON] HF A/B Analysis")
        print("-" * 30)

        base_result = results.get('base_hf', {})
        aegis_result = results.get('aegis_hf', {})

        for benchmark in ['math', 'gpqa', 'arc_challenge']:
            if benchmark in base_result and benchmark in aegis_result:
                base_acc = base_result[benchmark]['accuracy']
                aegis_acc = aegis_result[benchmark]['accuracy']
                improvement = aegis_acc - base_acc

                print(".4f")
    # 保存
    output_file = Path("results/ab_test_results/hf_advanced_benchmarks_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] HF Results saved to: {output_file}")

if __name__ == "__main__":
    main()
