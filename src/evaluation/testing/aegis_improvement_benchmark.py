#!/usr/bin/env python3
"""
AEGIS Improvement Benchmark Script
Alpha Gate調整と論理チューニング後の改善を評価
"""

import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def run_ollama_benchmark(model_name: str, questions: List[str], timeout: int = 30) -> List[Dict[str, Any]]:
    """Ollamaモデルでベンチマーク実行"""

    results = []

    for i, question in enumerate(questions, 1):
        print(f"[BENCHMARK] Testing {model_name} - Question {i}/{len(questions)}")

        try:
            # Ollamaコマンド実行
            cmd = f"ollama run {model_name} \"{question}\""

            start_time = time.time()
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )
            end_time = time.time()

            response_time = end_time - start_time
            response = result.stdout.strip() if result.returncode == 0 else f"ERROR: {result.stderr}"

            results.append({
                "question_id": i,
                "question": question,
                "response": response,
                "response_time": response_time,
                "success": result.returncode == 0,
                "error": result.stderr if result.returncode != 0 else None
            })

        except subprocess.TimeoutExpired:
            results.append({
                "question_id": i,
                "question": question,
                "response": "TIMEOUT",
                "response_time": timeout,
                "success": False,
                "error": "Timeout"
            })

        except Exception as e:
            results.append({
                "question_id": i,
                "question": question,
                "response": f"EXCEPTION: {str(e)}",
                "response_time": 0,
                "success": False,
                "error": str(e)
            })

    return results

def evaluate_mathematical_correctness(response: str, expected_answer: str) -> Dict[str, Any]:
    """数学的正確性の評価"""

    evaluation = {
        "correct": False,
        "partial_credit": 0.0,
        "reasoning_quality": "poor",
        "hallucination_detected": False,
        "calculation_preserved": True
    }

    response_lower = response.lower()
    expected_lower = expected_answer.lower()

    # 正確性チェック
    if expected_answer in response:
        evaluation["correct"] = True
        evaluation["partial_credit"] = 1.0
    else:
        # 部分一致チェック
        expected_words = set(expected_lower.split())
        response_words = set(response_lower.split())

        overlap = len(expected_words.intersection(response_words))
        if overlap > 0:
            evaluation["partial_credit"] = overlap / len(expected_words)

    # 推論品質評価
    reasoning_indicators = ["step", "calculate", "add", "subtract", "multiply", "divide", "equals"]
    reasoning_matches = sum(1 for indicator in reasoning_indicators if indicator in response_lower)

    if reasoning_matches >= 3:
        evaluation["reasoning_quality"] = "excellent"
    elif reasoning_matches >= 2:
        evaluation["reasoning_quality"] = "good"
    elif reasoning_matches >= 1:
        evaluation["reasoning_quality"] = "fair"

    # 幻覚検知（計算結果の置き換え）
    import re
    numbers = re.findall(r'\d+\.?\d*', response)
    if len(numbers) > 5:  # 過剰な数字出現
        evaluation["hallucination_detected"] = True

    # 計算結果保持チェック（一貫性の欠如）
    if "calculate" in response_lower or "compute" in response_lower:
        # 計算過程があるのに最終答えが合わない
        if not evaluation["correct"] and evaluation["partial_credit"] < 0.5:
            evaluation["calculation_preserved"] = False

    return evaluation

def create_gsm8k_test_set() -> List[Dict[str, Any]]:
    """GSM8Kテストセット作成"""

    test_cases = [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "expected_answer": "72",
            "difficulty": "basic_arithmetic"
        },
        {
            "question": "A robe takes 2 bolts of blue fiber and half times that much white fiber. How many bolts in total does it take?",
            "expected_answer": "3",
            "difficulty": "fraction_arithmetic"
        },
        {
            "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "expected_answer": "$70,000",
            "difficulty": "percentage_calculation"
        },
        {
            "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the evening to follow her daily routine?",
            "expected_answer": "30",
            "difficulty": "multiplication_distribution"
        },
        {
            "question": "A company produces 420 units of a product per day. The company operates 6 days a week. How many units does the company produce in a week?",
            "expected_answer": "2520",
            "difficulty": "basic_multiplication"
        }
    ]

    return test_cases

def create_logical_reasoning_test_set() -> List[Dict[str, Any]]:
    """論理的推論テストセット作成"""

    test_cases = [
        {
            "question": "If all roses are flowers and some flowers are red, does it necessarily follow that some roses are red? Explain your reasoning.",
            "expected_answer": "No",
            "difficulty": "logical_fallacy"
        },
        {
            "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "expected_answer": "$0.05",
            "difficulty": "intuitive_trap"
        },
        {
            "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "expected_answer": "5 minutes",
            "difficulty": "scaling_logic"
        },
        {
            "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 30 days for the patch to cover the entire lake, on which day was the lake half covered?",
            "expected_answer": "Day 29",
            "difficulty": "exponential_reasoning"
        }
    ]

    return test_cases

def analyze_improvements(baseline_results: Dict, improved_results: Dict) -> Dict[str, Any]:
    """改善分析"""

    analysis = {
        "overall_improvement": {},
        "category_breakdown": {},
        "hallucination_reduction": {},
        "reasoning_quality_improvement": {},
        "response_time_change": {}
    }

    # 全体的な改善度
    baseline_correct = sum(1 for r in baseline_results["evaluations"] if r["correct"])
    improved_correct = sum(1 for r in improved_results["evaluations"] if r["correct"])

    baseline_total = len(baseline_results["evaluations"])
    improved_total = len(improved_results["evaluations"])

    analysis["overall_improvement"] = {
        "baseline_accuracy": baseline_correct / baseline_total,
        "improved_accuracy": improved_correct / improved_total,
        "improvement": (improved_correct / improved_total) - (baseline_correct / baseline_total),
        "baseline_correct": baseline_correct,
        "improved_correct": improved_correct
    }

    # 幻覚現象の低減
    baseline_hallucinations = sum(1 for r in baseline_results["evaluations"] if r["hallucination_detected"])
    improved_hallucinations = sum(1 for r in improved_results["evaluations"] if r["hallucination_detected"])

    analysis["hallucination_reduction"] = {
        "baseline_hallucination_rate": baseline_hallucinations / baseline_total,
        "improved_hallucination_rate": improved_hallucinations / improved_total,
        "reduction": (baseline_hallucinations / baseline_total) - (improved_hallucinations / improved_total)
    }

    # 推論品質の改善
    reasoning_qualities = ["poor", "fair", "good", "excellent"]
    baseline_avg_quality = sum(reasoning_qualities.index(r["reasoning_quality"]) for r in baseline_results["evaluations"]) / baseline_total
    improved_avg_quality = sum(reasoning_qualities.index(r["reasoning_quality"]) for r in improved_results["evaluations"]) / improved_total

    analysis["reasoning_quality_improvement"] = {
        "baseline_avg_quality": baseline_avg_quality,
        "improved_avg_quality": improved_avg_quality,
        "improvement": improved_avg_quality - baseline_avg_quality
    }

    # 応答時間の変化
    baseline_avg_time = sum(r["response_time"] for r in baseline_results["results"]) / len(baseline_results["results"])
    improved_avg_time = sum(r["response_time"] for r in improved_results["results"]) / len(improved_results["results"])

    analysis["response_time_change"] = {
        "baseline_avg_time": baseline_avg_time,
        "improved_avg_time": improved_avg_time,
        "change": improved_avg_time - baseline_avg_time,
        "faster": improved_avg_time < baseline_avg_time
    }

    return analysis

def generate_improvement_report(baseline_results: Dict, improved_results: Dict, analysis: Dict, output_dir: str):
    """改善レポート生成"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(output_dir) / f"aegis_improvement_report_{timestamp}.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# AEGIS Improvement Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n\n")

        improvement = analysis["overall_improvement"]["improvement"]
        hallucination_reduction = analysis["hallucination_reduction"]["reduction"]

        if improvement > 0:
            f.write(f"[OK] **Positive improvement detected:** +{improvement:.1%} accuracy increase\n")
        else:
            f.write(f"[WARN] **No significant improvement:** {improvement:.1%} accuracy change\n")

        if hallucination_reduction > 0:
            f.write(f"[OK] **Hallucination reduction:** -{hallucination_reduction:.1%} hallucination rate\n")
        else:
            f.write(f"[WARN] **Hallucination unchanged:** {hallucination_reduction:.1%} change\n")

        f.write("\n## Detailed Results\n\n")

        # 全体性能比較
        f.write("### Overall Performance\n\n")
        f.write("| Metric | Baseline | Improved | Change |\n")
        f.write("|--------|----------|----------|--------|\n")
        f.write(".1%")
        f.write(".1%")
        f.write("+.1%")
        f.write(".1%")
        f.write("+.1%")
        f.write(".1%")
        f.write("+.1%")
        f.write(".1%")
        f.write("+.1%")
        f.write("\n")

        # 各テストケースの詳細
        f.write("### Test Case Analysis\n\n")

        for i, (baseline, improved) in enumerate(zip(baseline_results["results"], improved_results["results"]), 1):
            f.write(f"#### Test Case {i}\n")
            f.write(f"**Question:** {baseline['question']}\n\n")

            f.write("**Baseline Model:**\n")
            f.write(f"- Response: {baseline['response'][:200]}...\n" if len(baseline['response']) > 200 else f"- Response: {baseline['response']}\n")
            f.write(".2f")
            eval_b = baseline_results["evaluations"][i-1]
            f.write(f"- Correct: {'[OK]' if eval_b['correct'] else '[NG]'}\n")
            f.write(f"- Reasoning Quality: {eval_b['reasoning_quality']}\n")
            f.write(f"- Hallucination: {'[WARN]' if eval_b['hallucination_detected'] else '[OK]'}\n\n")

            f.write("**Improved Model:**\n")
            f.write(f"- Response: {improved['response'][:200]}...\n" if len(improved['response']) > 200 else f"- Response: {improved['response']}\n")
            f.write(".2f")
            eval_i = improved_results["evaluations"][i-1]
            f.write(f"- Correct: {'[OK]' if eval_i['correct'] else '[NG]'}\n")
            f.write(f"- Reasoning Quality: {eval_i['reasoning_quality']}\n")
            f.write(f"- Hallucination: {'[WARN]' if eval_i['hallucination_detected'] else '[OK]'}\n\n")

        f.write("## Recommendations\n\n")

        if improvement > 0.1:
            f.write("### [TARGET] Strong Positive Results\n")
            f.write("- Current improvements are significant and promising\n")
            f.write("- Consider deploying improved model for production testing\n")
            f.write("- Further optimization may yield additional gains\n\n")

        elif improvement > 0:
            f.write("### 📈 Moderate Improvement\n")
            f.write("- Improvements detected but may need further tuning\n")
            f.write("- Consider adjusting Alpha Gate scale factor or training parameters\n")
            f.write("- Additional logic tuning cycles may be beneficial\n\n")

        else:
            f.write("### 🔄 Needs Further Investigation\n")
            f.write("- No significant improvement detected\n")
            f.write("- Consider different approaches:\n")
            f.write("  - Reduce Alpha Gate scale factor further (e.g., 0.6)\n")
            f.write("  - Increase logic training epochs\n")
            f.write("  - Use different dataset combinations\n")
            f.write("  - Review SO(8) transformation parameters\n\n")

        f.write("## Technical Details\n\n")
        f.write("### Configuration Used\n")
        f.write("- Alpha Gate Scale Factor: 0.8\n")
        f.write("- Logic Training Epochs: 2\n")
        f.write("- LoRA Rank: 8\n")
        f.write("- Training Dataset: GSM8K + Synthetic Logic Data\n\n")

        f.write("### Test Methodology\n")
        f.write("- 9 comprehensive test cases\n")
        f.write("- Mathematical correctness evaluation\n")
        f.write("- Reasoning quality assessment\n")
        f.write("- Hallucination detection\n")
        f.write("- Response time measurement\n\n")

    print(f"[REPORT] Improvement analysis report saved: {report_path}")
    return str(report_path)

def main():
    """メイン実行関数"""
    import argparse

    parser = argparse.ArgumentParser(description="AEGIS Improvement Benchmark")
    parser.add_argument("--baseline-model", default="model-a:q8_0", help="Baseline model name for Ollama")
    parser.add_argument("--improved-model", required=True, help="Improved model name for Ollama")
    parser.add_argument("--output-dir", default="_docs/benchmark_results/improvement_tests", help="Output directory")
    parser.add_argument("--test-type", choices=["gsm8k", "logic", "both"], default="both", help="Test type to run")

    args = parser.parse_args()

    print("=" * 60)
    print("AEGIS IMPROVEMENT BENCHMARK")
    print("=" * 60)
    print(f"Baseline Model: {args.baseline_model}")
    print(f"Improved Model: {args.improved_model}")
    print(f"Test Type: {args.test_type}")

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # テストケース準備
    test_cases = []
    if args.test_type in ["gsm8k", "both"]:
        test_cases.extend(create_gsm8k_test_set())
    if args.test_type in ["logic", "both"]:
        test_cases.extend(create_logical_reasoning_test_set())

    print(f"[BENCHMARK] Running {len(test_cases)} test cases...")

    # ベースラインモデルテスト
    print("\n" + "="*40)
    print("TESTING BASELINE MODEL")
    print("="*40)
    baseline_results = run_ollama_benchmark(args.baseline_model, [tc["question"] for tc in test_cases])

    # 評価実行
    baseline_evaluations = []
    for result, test_case in zip(baseline_results, test_cases):
        evaluation = evaluate_mathematical_correctness(result["response"], test_case["expected_answer"])
        evaluation["difficulty"] = test_case["difficulty"]
        baseline_evaluations.append(evaluation)

    baseline_data = {
        "model": args.baseline_model,
        "results": baseline_results,
        "evaluations": baseline_evaluations,
        "test_cases": test_cases
    }

    # 改善モデルテスト
    print("\n" + "="*40)
    print("TESTING IMPROVED MODEL")
    print("="*40)
    improved_results = run_ollama_benchmark(args.improved_model, [tc["question"] for tc in test_cases])

    # 評価実行
    improved_evaluations = []
    for result, test_case in zip(improved_results, test_cases):
        evaluation = evaluate_mathematical_correctness(result["response"], test_case["expected_answer"])
        evaluation["difficulty"] = test_case["difficulty"]
        improved_evaluations.append(evaluation)

    improved_data = {
        "model": args.improved_model,
        "results": improved_results,
        "evaluations": improved_evaluations,
        "test_cases": test_cases
    }

    # 改善分析
    print("\n" + "="*40)
    print("ANALYZING IMPROVEMENTS")
    print("="*40)
    analysis = analyze_improvements(baseline_data, improved_data)

    # レポート生成
    report_path = generate_improvement_report(baseline_data, improved_data, analysis, str(output_dir))

    # JSON結果保存
    results_path = output_dir / f"improvement_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "baseline": baseline_data,
            "improved": improved_data,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED!")
    print("=" * 60)
    print(f"Report: {report_path}")
    print(f"Raw Results: {results_path}")
    print(".1%")
    print(".1%")
    print("+.1%")

    if analysis["hallucination_reduction"]["reduction"] > 0:
        print(".1%")

    print("\nNext steps:")
    print("1. Review the detailed report for specific improvement areas")
    print("2. Adjust parameters based on findings")
    print("3. Run additional training cycles if needed")
    print("4. Consider production deployment if improvements are satisfactory")

if __name__ == "__main__":
    main()
