#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/Bテスト結果の統計解析スクリプト

ANOVA、効果量、p値、エラーバー付きグラフを含む統計分析を実行
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, cohen_d
import matplotlib.pyplot as plt
import seaborn as sns

# tqdm for progress bars
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class ABTestStatisticalAnalyzer:
    """A/Bテスト統計解析クラス"""

    def __init__(self, results_dir: str = "results/ab_test_results"):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.stats_dir = self.results_dir / "statistics"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        # スタイル設定
        plt.style.use('default')
        sns.set_palette("husl")

    def load_latest_results(self) -> Dict[str, Any]:
        """最新のA/Bテスト結果を読み込み"""
        result_files = list(self.results_dir.glob("ab_test_results_final_*.json"))

        if not result_files:
            raise FileNotFoundError("No final A/B test results found")

        # 最新のファイルを取得
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)

        print(f"📂 Loading results from {latest_file}")

        with open(latest_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        return results

    def prepare_data_for_analysis(self, results: Dict[str, Any]) -> pd.DataFrame:
        """統計分析用データ準備"""
        data_rows = []

        baseline = results.get("baseline", {})
        aegis = results.get("aegis", {})

        for task in results.get("comparison", {}):
            for fewshot_str in results["comparison"][task]:
                fewshot = int(fewshot_str)

                baseline_result = baseline.get(task, {}).get(fewshot_str, {})
                aegis_result = aegis.get(task, {}).get(fewshot_str, {})

                baseline_acc = baseline_result.get("accuracy", 0)
                aegis_acc = aegis_result.get("accuracy", 0)

                # 個別のサンプル結果を取得
                baseline_samples = baseline_result.get("results", [])
                aegis_samples = aegis_result.get("results", [])

                # 正解/不正解のバイナリデータを作成
                for i, (b_sample, a_sample) in enumerate(zip(baseline_samples, aegis_samples)):
                    data_rows.append({
                        "task": task,
                        "fewshot": fewshot,
                        "model": "baseline",
                        "sample_id": i,
                        "correct": b_sample.get("exact_match", False),
                        "accuracy": baseline_acc,
                        "inference_time": b_sample.get("inference_time", 0)
                    })

                    data_rows.append({
                        "task": task,
                        "fewshot": fewshot,
                        "model": "aegis",
                        "sample_id": i,
                        "correct": a_sample.get("exact_match", False),
                        "accuracy": aegis_acc,
                        "inference_time": a_sample.get("inference_time", 0)
                    })

        df = pd.DataFrame(data_rows)
        return df

    def calculate_effect_size(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Cohen's d効果量計算"""
        try:
            return cohen_d(group1, group2).statistic
        except:
            # 手動計算
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            n1, n2 = len(group1), len(group2)

            # pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

            if pooled_std == 0:
                return 0.0

            return (mean2 - mean1) / pooled_std

    def perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """統計検定実行（ANOVA、t-test、効果量）"""
        print("[RESEARCH] Performing statistical analysis...")

        stats_results = {
            "overall_comparison": {},
            "task_wise_comparison": {},
            "fewshot_analysis": {},
            "anova_results": {},
            "effect_sizes": {}
        }

        # 全体比較（全タスク・全fewshot統合）
        baseline_overall = df[df["model"] == "baseline"]["correct"].astype(int)
        aegis_overall = df[df["model"] == "aegis"]["correct"].astype(int)

        # t-test
        t_stat, p_value = ttest_ind(baseline_overall, aegis_overall)

        # 効果量
        effect_size = self.calculate_effect_size(baseline_overall.values, aegis_overall.values)

        stats_results["overall_comparison"] = {
            "baseline_mean": baseline_overall.mean(),
            "aegis_mean": aegis_overall.mean(),
            "improvement": aegis_overall.mean() - baseline_overall.mean(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "effect_size_cohen_d": effect_size,
            "effect_size_interpretation": self.interpret_effect_size(effect_size),
            "sample_size_baseline": len(baseline_overall),
            "sample_size_aegis": len(aegis_overall)
        }

        # タスク別比較
        for task in df["task"].unique():
            task_df = df[df["task"] == task]

            baseline_task = task_df[task_df["model"] == "baseline"]["correct"].astype(int)
            aegis_task = task_df[task_df["model"] == "aegis"]["correct"].astype(int)

            if len(baseline_task) > 0 and len(aegis_task) > 0:
                t_stat_task, p_value_task = ttest_ind(baseline_task, aegis_task)
                effect_size_task = self.calculate_effect_size(baseline_task.values, aegis_task.values)

                stats_results["task_wise_comparison"][task] = {
                    "baseline_mean": baseline_task.mean(),
                    "aegis_mean": aegis_task.mean(),
                    "improvement": aegis_task.mean() - baseline_task.mean(),
                    "t_statistic": t_stat_task,
                    "p_value": p_value_task,
                    "effect_size_cohen_d": effect_size_task,
                    "effect_size_interpretation": self.interpret_effect_size(effect_size_task),
                    "sample_size_baseline": len(baseline_task),
                    "sample_size_aegis": len(aegis_task)
                }

        # Few-shot分析
        for fewshot in df["fewshot"].unique():
            fewshot_df = df[df["fewshot"] == fewshot]

            baseline_fs = fewshot_df[fewshot_df["model"] == "baseline"]["correct"].astype(int)
            aegis_fs = fewshot_df[fewshot_df["model"] == "aegis"]["correct"].astype(int)

            if len(baseline_fs) > 0 and len(aegis_fs) > 0:
                t_stat_fs, p_value_fs = ttest_ind(baseline_fs, aegis_fs)
                effect_size_fs = self.calculate_effect_size(baseline_fs.values, aegis_fs.values)

                stats_results["fewshot_analysis"][str(fewshot)] = {
                    "baseline_mean": baseline_fs.mean(),
                    "aegis_mean": aegis_fs.mean(),
                    "improvement": aegis_fs.mean() - baseline_fs.mean(),
                    "t_statistic": t_stat_fs,
                    "p_value": p_value_fs,
                    "effect_size_cohen_d": effect_size_fs,
                    "effect_size_interpretation": self.interpret_effect_size(effect_size_fs)
                }

        # ANOVA分析（fewshotの効果）
        try:
            anova_data = []
            for fewshot in df["fewshot"].unique():
                fs_data = df[df["fewshot"] == fewshot]["correct"].astype(int)
                anova_data.append(fs_data.values)

            if len(anova_data) >= 2:
                f_stat, p_anova = f_oneway(*anova_data)
                stats_results["anova_results"]["fewshot_effect"] = {
                    "f_statistic": f_stat,
                    "p_value": p_anova,
                    "significant": p_anova < 0.05
                }
        except Exception as e:
            print(f"[WARN] ANOVA analysis failed: {e}")

        return stats_results

    def interpret_effect_size(self, d: float) -> str:
        """Cohen's dの解釈"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def create_comparison_plots(self, df: pd.DataFrame, stats_results: Dict[str, Any]):
        """比較グラフ作成（エラーバー付き）"""
        print("[STATS] Creating comparison plots...")

        # タスク別比較グラフ
        plt.figure(figsize=(15, 10))

        # 集計データ作成
        summary_df = df.groupby(["task", "model"])["correct"].agg(["mean", "std", "count"]).reset_index()
        summary_df["se"] = summary_df["std"] / np.sqrt(summary_df["count"])  # 標準誤差

        # タスク別プロット
        plt.subplot(2, 2, 1)
        tasks = summary_df["task"].unique()
        x = np.arange(len(tasks))
        width = 0.35

        baseline_data = summary_df[summary_df["model"] == "baseline"]
        aegis_data = summary_df[summary_df["model"] == "aegis"]

        plt.bar(x - width/2, baseline_data["mean"], width, label="Baseline",
                yerr=baseline_data["se"], capsize=5, alpha=0.8)
        plt.bar(x + width/2, aegis_data["mean"], width, label="AEGIS",
                yerr=aegis_data["se"], capsize=5, alpha=0.8)

        plt.xlabel("Task")
        plt.ylabel("Accuracy")
        plt.title("Model Comparison by Task (with Error Bars)")
        plt.xticks(x, tasks, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Few-shot分析グラフ
        plt.subplot(2, 2, 2)
        fewshot_summary = df.groupby(["fewshot", "model"])["correct"].agg(["mean", "std", "count"]).reset_index()
        fewshot_summary["se"] = fewshot_summary["std"] / np.sqrt(fewshot_summary["count"])

        fewshots = sorted(fewshot_summary["fewshot"].unique())
        x_fs = np.arange(len(fewshots))

        baseline_fs = fewshot_summary[fewshot_summary["model"] == "baseline"]
        aegis_fs = fewshot_summary[fewshot_summary["model"] == "aegis"]

        plt.bar(x_fs - width/2, baseline_fs["mean"], width, label="Baseline",
                yerr=baseline_fs["se"], capsize=5, alpha=0.8)
        plt.bar(x_fs + width/2, aegis_fs["mean"], width, label="AEGIS",
                yerr=aegis_fs["se"], capsize=5, alpha=0.8)

        plt.xlabel("Few-shot Examples")
        plt.ylabel("Accuracy")
        plt.title("Few-shot Analysis (with Error Bars)")
        plt.xticks(x_fs, fewshots)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 効果量プロット
        plt.subplot(2, 2, 3)
        if "task_wise_comparison" in stats_results:
            tasks_effect = list(stats_results["task_wise_comparison"].keys())
            effect_sizes = [stats_results["task_wise_comparison"][task]["effect_size_cohen_d"]
                          for task in tasks_effect]

            colors = ['red' if x < 0 else 'green' for x in effect_sizes]
            plt.barh(tasks_effect, effect_sizes, color=colors, alpha=0.7)
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.xlabel("Effect Size (Cohen's d)")
            plt.title("Effect Size by Task")
            plt.grid(True, alpha=0.3)

        # p値分布
        plt.subplot(2, 2, 4)
        if "task_wise_comparison" in stats_results:
            p_values = [stats_results["task_wise_comparison"][task]["p_value"]
                       for task in tasks_effect]

            plt.scatter(range(len(p_values)), p_values, alpha=0.7, s=50)
            plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
            plt.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='p=0.01')
            plt.xlabel("Task Index")
            plt.ylabel("p-value")
            plt.title("p-values by Task")
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = self.plots_dir / "ab_test_comparison_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[STATS] Plots saved to {plot_file}")

    def generate_statistical_report(self, stats_results: Dict[str, Any]) -> str:
        """統計レポート生成"""
        report = []
        report.append("# A/B Test Statistical Analysis Report")
        report.append("")
        report.append("## Overall Comparison")
        overall = stats_results["overall_comparison"]
        report.append(f"- Baseline mean: {overall['mean_baseline']:.4f}")
        report.append(f"- Baseline std: {overall['std_baseline']:.4f}")
        report.append(f"- AEGIS mean: {overall['mean_aegis']:.4f}")
        report.append(f"- AEGIS std: {overall['std_aegis']:.4f}")
        report.append(f"- Effect Size (Cohen's d): {overall['effect_size_cohen_d']:.4f} ({overall['effect_size_interpretation']})")
        report.append(f"- p-value: {overall['p_value']:.6f}")
        report.append(f"- Sample sizes: Baseline={overall['sample_size_baseline']}, AEGIS={overall['sample_size_aegis']}")
        report.append("")
        report.append("## Task-wise Comparison")
        for task, results in stats_results["task_wise_comparison"].items():
            report.append(f"### {task}")
            report.append(f"- Baseline mean: {results['mean_baseline']:.4f}")
            report.append(f"- Baseline std: {results['std_baseline']:.4f}")
            report.append(f"- AEGIS mean: {results['mean_aegis']:.4f}")
            report.append(f"- AEGIS std: {results['std_aegis']:.4f}")
            report.append(f"- Effect Size: {results['effect_size_cohen_d']:.4f} ({results['effect_size_interpretation']})")
            report.append(f"- p-value: {results['p_value']:.6f}")
            report.append("")
        report.append("## Few-shot Analysis")
        for fewshot, results in stats_results["fewshot_analysis"].items():
            report.append(f"### {fewshot}-shot")
            report.append(f"- Baseline mean: {results['mean_baseline']:.4f}")
            report.append(f"- Baseline std: {results['std_baseline']:.4f}")
            report.append(f"- AEGIS mean: {results['mean_aegis']:.4f}")
            report.append(f"- AEGIS std: {results['std_aegis']:.4f}")
            report.append(f"- Effect Size: {results['effect_size_cohen_d']:.4f} ({results['effect_size_interpretation']})")
            report.append(f"- p-value: {results['p_value']:.6f}")
            report.append("")

        if "anova_results" in stats_results and stats_results["anova_results"]:
            report.append("## ANOVA Results")
            anova = stats_results["anova_results"]["fewshot_effect"]
            report.append(f"- F-value: {anova['f_value']:.4f}")
            report.append(f"- p-value: {anova['p_value']:.6f}")
            report.append(f"- Significant: {'Yes' if anova['significant'] else 'No'}")
            report.append("")

        report.append("## Interpretation")
        report.append("- **p < 0.05**: Statistically significant difference")
        report.append("- **Effect Size**: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)")
        report.append("- **ANOVA**: Tests if few-shot examples significantly affect performance")

        return "\n".join(report)

    def run_analysis(self):
        """統計分析実行"""
        print("[START] Starting A/B Test Statistical Analysis")
        print("=" * 60)

        try:
            # 結果読み込み
            results = self.load_latest_results()

            # データ準備
            df = self.prepare_data_for_analysis(results)
            print(f"[INFO] Prepared {len(df)} data points for analysis")

            # 統計検定
            stats_results = self.perform_statistical_tests(df)

            # グラフ作成
            self.create_comparison_plots(df, stats_results)

            # レポート生成
            report = self.generate_statistical_report(stats_results)

            # 結果保存
            stats_file = self.stats_dir / "statistical_analysis_results.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_results, f, indent=2, ensure_ascii=False)

            report_file = self.stats_dir / "statistical_analysis_report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)

            # CSVエクスポート
            df.to_csv(self.stats_dir / "ab_test_raw_data.csv", index=False)

            print("\n[SUCCESS] Statistical analysis completed!")
            print(f"[INFO] Results saved to {self.stats_dir}")
            print(f"[INFO] Plots saved to {self.plots_dir}")

            # 主要結果表示
            overall = stats_results["overall_comparison"]
            print("\n[INFO] Key Results:")
            print(f"[INFO] Mean A: {overall['mean_a']:.4f}")
            print(f"[INFO] Mean B: {overall['mean_b']:.4f}")
            print(f"[INFO] Std A: {overall['std_a']:.4f}")
            print(f"[INFO] Std B: {overall['std_b']:.4f}")
            print(f"[INFO] Effect Size: {overall['effect_size_cohen_d']:.4f} ({overall['effect_size_interpretation']})")
            print(f"[INFO] P-value: {overall['p_value']:.6f}")
            return stats_results

        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Analyze A/B Test Results Statistically")
    parser.add_argument("--results_dir", type=str, default="results/ab_test_results",
                       help="Directory containing A/B test results")

    args = parser.parse_args()

    analyzer = ABTestStatisticalAnalyzer(args.results_dir)
    results = analyzer.run_analysis()

    if results:
        print("\n[OK] Statistical analysis completed!")
        print("[STATS] Next: Prepare for HF upload with scripts/evaluation/prepare_hf_upload.py")
    else:
        print("\n[NG] Statistical analysis failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
