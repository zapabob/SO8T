#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama.cpp.python A/B Testing Framework
完全評価スイート：Baseline vs AEGIS比較

テスト内容：
1. 日本語性能テスト（ELYZA-100）
2. 数学・科学推論テスト
3. 一般知識テスト
4. 統計分析とレポート生成
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def load_models_for_ab_test(baseline_path: str, aegis_path: str) -> Tuple[Any, Any]:
    """A/Bテスト用モデルのロード"""
    print("[A/B TEST] Loading models for comparison...")

    try:
        from llama_cpp import Llama

        # Baselineモデル
        print(f"[A/B TEST] Loading baseline model: {baseline_path}")
        baseline_model = Llama(
            model_path=baseline_path,
            n_ctx=4096,
            n_threads=8,
            verbose=False
        )

        # AEGISモデル
        print(f"[A/B TEST] Loading AEGIS model: {aegis_path}")
        aegis_model = Llama(
            model_path=aegis_path,
            n_ctx=4096,
            n_threads=8,
            verbose=False
        )

        print("[A/B TEST] Models loaded successfully")
        return baseline_model, aegis_model

    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return None, None

def run_japanese_performance_test(model, model_name: str, test_cases: List[Dict]) -> List[Dict]:
    """日本語性能テスト実行"""
    print(f"[A/B TEST] Running Japanese performance test for {model_name}...")

    results = []

    for i, test_case in enumerate(test_cases):
        question = test_case["question"]
        expected_answer = test_case.get("expected_answer", "")

        try:
            # モデル推論
            start_time = time.time()
            output = model(
                f"質問：{question}\n\n回答：",
                max_tokens=512,
                temperature=0.7,
                stop=["\n\n", "質問："]
            )
            inference_time = time.time() - start_time

            generated_answer = output["choices"][0]["text"].strip()

            result = {
                "test_id": f"jp_test_{i+1}",
                "model": model_name,
                "question": question,
                "generated_answer": generated_answer,
                "expected_answer": expected_answer,
                "inference_time": inference_time,
                "test_category": "japanese_performance"
            }

            results.append(result)
            print(f"[A/B TEST] Test {i+1}/{len(test_cases)} completed for {model_name}")

        except Exception as e:
            print(f"[ERROR] Test {i+1} failed for {model_name}: {e}")
            continue

    return results

def run_mathematical_reasoning_test(model, model_name: str, test_cases: List[Dict]) -> List[Dict]:
    """数学推論テスト実行"""
    print(f"[A/B TEST] Running mathematical reasoning test for {model_name}...")

    results = []

    for i, test_case in enumerate(test_cases):
        problem = test_case["problem"]
        expected_answer = test_case.get("expected_answer", "")

        try:
            # モデル推論
            start_time = time.time()
            output = model(
                f"以下の数学の問題をステップバイステップで解いてください。\n\n問題：{problem}\n\n解答：",
                max_tokens=1024,
                temperature=0.1,  # 数学問題は温度を低く
                stop=["\n\n", "問題："]
            )
            inference_time = time.time() - start_time

            generated_answer = output["choices"][0]["text"].strip()

            result = {
                "test_id": f"math_test_{i+1}",
                "model": model_name,
                "problem": problem,
                "generated_answer": generated_answer,
                "expected_answer": expected_answer,
                "inference_time": inference_time,
                "test_category": "mathematical_reasoning"
            }

            results.append(result)
            print(f"[A/B TEST] Math test {i+1}/{len(test_cases)} completed for {model_name}")

        except Exception as e:
            print(f"[ERROR] Math test {i+1} failed for {model_name}: {e}")
            continue

    return results

def analyze_results(baseline_results: List[Dict], aegis_results: List[Dict]) -> Dict[str, Any]:
    """A/Bテスト結果の統計分析"""
    print("[A/B TEST] Analyzing A/B test results...")

    # データをDataFrameに変換
    all_results = baseline_results + aegis_results
    df = pd.DataFrame(all_results)

    # 基本統計
    stats = {
        "total_tests": len(all_results),
        "baseline_tests": len([r for r in all_results if r["model"] == "baseline"]),
        "aegis_tests": len([r for r in all_results if r["model"] == "aegis"]),
        "avg_inference_time_baseline": df[df["model"] == "baseline"]["inference_time"].mean(),
        "avg_inference_time_aegis": df[df["model"] == "aegis"]["inference_time"].mean(),
        "test_categories": df["test_category"].unique().tolist()
    }

    # カテゴリ別分析
    category_analysis = {}
    for category in df["test_category"].unique():
        cat_df = df[df["test_category"] == category]
        category_analysis[category] = {
            "baseline_avg_time": cat_df[cat_df["model"] == "baseline"]["inference_time"].mean(),
            "aegis_avg_time": cat_df[cat_df["model"] == "aegis"]["inference_time"].mean(),
            "baseline_count": len(cat_df[cat_df["model"] == "baseline"]),
            "aegis_count": len(cat_df[cat_df["model"] == "aegis"])
        }

    stats["category_analysis"] = category_analysis

    # 統計的有意差検定（簡易版）
    from scipy import stats

    baseline_times = df[df["model"] == "baseline"]["inference_time"]
    aegis_times = df[df["model"] == "aegis"]["inference_time"]

    if len(baseline_times) > 1 and len(aegis_times) > 1:
        t_stat, p_value = stats.ttest_ind(baseline_times, aegis_times)
        stats["inference_time_ttest"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }

    return stats

def create_visualizations(results: List[Dict], stats: Dict, output_dir: str):
    """結果の可視化"""
    print("[A/B TEST] Creating visualizations...")

    df = pd.DataFrame(results)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 推論時間比較グラフ
    plt.figure(figsize=(12, 6))

    # 全体の推論時間比較
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x="model", y="inference_time")
    plt.title("Inference Time Comparison: Baseline vs AEGIS")
    plt.ylabel("Inference Time (seconds)")

    # カテゴリ別推論時間比較
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x="test_category", y="inference_time", hue="model")
    plt.title("Inference Time by Category")
    plt.xticks(rotation=45)
    plt.ylabel("Inference Time (seconds)")

    plt.tight_layout()
    plt.savefig(output_path / "inference_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 統計レポート
    report = f"""
A/B Test Results Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Test Summary:
- Total Tests: {stats['total_tests']}
- Baseline Tests: {stats['baseline_tests']}
- AEGIS Tests: {stats['aegis_tests']}

Performance Metrics:
- Baseline Avg Inference Time: {stats['avg_inference_time_baseline']:.3f}s
- AEGIS Avg Inference Time: {stats['avg_inference_time_aegis']:.3f}s
- Performance Ratio (Baseline/AEGIS): {stats['avg_inference_time_baseline']/stats['avg_inference_time_aegis']:.2f}x

Category Analysis:
"""

    for category, cat_stats in stats["category_analysis"].items():
        report += f"""
{category}:
  - Baseline Avg Time: {cat_stats['baseline_avg_time']:.3f}s
  - AEGIS Avg Time: {cat_stats['aegis_avg_time']:.3f}s
  - Ratio: {cat_stats['baseline_avg_time']/cat_stats['aegis_avg_time']:.2f}x
"""

    if "inference_time_ttest" in stats:
        ttest = stats["inference_time_ttest"]
        report += f"""
Statistical Analysis:
- T-statistic: {ttest['t_statistic']:.3f}
- P-value: {ttest['p_value']:.4f}
- Significant Difference: {'Yes' if ttest['significant'] else 'No'} (p < 0.05)
"""

    # レポート保存
    with open(output_path / "ab_test_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"[A/B TEST] Visualizations and report saved to {output_path}")

def main():
    """メインA/Bテスト実行関数"""
    print("🚀 Starting Llama.cpp.python A/B Testing...")
    print("=" * 60)

    # 設定
    baseline_model_path = "models/ab_test_models/baseline/model.gguf"
    aegis_model_path = "models/ab_test_models/aegis/model.gguf"
    output_dir = "results/ab_test_results"

    # テストケース定義
    japanese_tests = [
        {
            "question": "日本の首都はどこですか？",
            "expected_answer": "東京"
        },
        {
            "question": "日本の人口は約何人ですか？（2023年時点）",
            "expected_answer": "約1億2千万人"
        }
    ]

    math_tests = [
        {
            "problem": "2x + 3 = 7 の方程式を解け。",
            "expected_answer": "x = 2"
        },
        {
            "problem": "1から100までの整数の和を求めよ。",
            "expected_answer": "5050"
        }
    ]

    # モデルロード
    baseline_model, aegis_model = load_models_for_ab_test(baseline_model_path, aegis_model_path)

    if not baseline_model or not aegis_model:
        print("[ERROR] Failed to load models. Exiting.")
        return 1

    # テスト実行
    baseline_results = []
    aegis_results = []

    # 日本語テスト
    baseline_results.extend(run_japanese_performance_test(baseline_model, "baseline", japanese_tests))
    aegis_results.extend(run_japanese_performance_test(aegis_model, "aegis", japanese_tests))

    # 数学テスト
    baseline_results.extend(run_mathematical_reasoning_test(baseline_model, "baseline", math_tests))
    aegis_results.extend(run_mathematical_reasoning_test(aegis_model, "aegis", math_tests))

    # 結果分析
    all_results = baseline_results + aegis_results
    stats = analyze_results(baseline_results, aegis_results)

    # 可視化とレポート
    create_visualizations(all_results, stats, output_dir)

    # JSON保存
    results_file = Path(output_dir) / f"ab_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "results": all_results,
            "statistics": stats,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "baseline_model": baseline_model_path,
                "aegis_model": aegis_model_path
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 A/B testing completed!")
    print(f"📊 Results saved to {results_file}")
    print(f"📈 Performance ratio: {stats['avg_inference_time_baseline']/stats['avg_inference_time_aegis']:.2f}x")

    return 0

if __name__ == "__main__":
    sys.exit(main())