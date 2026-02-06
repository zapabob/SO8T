#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS v2.1 A/Bテスト実行スクリプト
元モデル vs AEGIS v2.1 の性能比較
lm-eval-harness + ELYZA-100使用
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 環境設定
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class AEGISABTester:
    """AEGIS v2.1 A/Bテストクラス"""

    def __init__(self):
        self.base_model = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
        self.aegis_model = "H:/from_D/webdataset/models/final/aegis_v21_sft_hf"
        self.results_dir = Path("results/ab_test_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 評価タスク設定
        self.tasks = [
            "elyza_tasks_100",  # ELYZA-100
            "jcommonsenseqa",   # 日本語常識QA
            "jsquad",          # 日本語QA
            "jnli",            # 日本語NLI
            "mgsm",            # 数学推論
            "gsm8k",           # 数学問題
        ]

    def run_single_evaluation(self, model_path, model_name, task):
        """単一モデルの単一タスク評価"""
        print(f"[EVAL] Running {model_name} on {task}")

        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True",
            "--tasks", task,
            "--device", "cuda:0",
            "--batch_size", "auto",
            "--output_path", str(self.results_dir / f"{model_name}_{task}.json"),
            "--log_samples", "false"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                print(f"[OK] {model_name} {task} completed")
                return True
            else:
                print(f"[ERROR] {model_name} {task} failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"[ERROR] Exception in {model_name} {task}: {e}")
            return False

    def run_ab_test(self):
        """A/Bテスト実行"""
        print("[AB-TEST] Starting A/B evaluation")
        print(f"Base model: {self.base_model}")
        print(f"AEGIS model: {self.aegis_model}")
        print("=" * 60)

        all_results = {
            "base_model": {},
            "aegis_model": {}
        }

        # 各タスクで両モデルを評価
        for task in self.tasks:
            print(f"\n[TASK] {task}")
            print("-" * 40)

            # Base model evaluation
            if self.run_single_evaluation(self.base_model, "base", task):
                result_file = self.results_dir / f"base_{task}.json"
                if result_file.exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_results["base_model"][task] = data

            # AEGIS model evaluation
            if self.run_single_evaluation(self.aegis_model, "aegis", task):
                result_file = self.results_dir / f"aegis_{task}.json"
                if result_file.exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_results["aegis_model"][task] = data

        # 結果保存
        with open(self.results_dir / "ab_test_raw_results.json", 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Raw results saved to {self.results_dir / 'ab_test_raw_results.json'}")
        return all_results

    def analyze_results(self, raw_results):
        """結果分析"""
        print("[ANALYSIS] Analyzing A/B test results")

        # 結果をデータフレームに変換
        results_data = []

        for model_type, tasks in raw_results.items():
            for task_name, task_data in tasks.items():
                if "results" in task_data:
                    for metric, value in task_data["results"].items():
                        if isinstance(value, dict) and "acc" in value:
                            acc = value["acc"]
                        elif isinstance(value, (int, float)):
                            acc = value
                        else:
                            continue

                        results_data.append({
                            "model": "Base Model" if model_type == "base_model" else "AEGIS v2.1",
                            "task": task_name,
                            "metric": metric,
                            "accuracy": acc
                        })

        df = pd.DataFrame(results_data)
        print(f"[DATA] Collected {len(df)} data points")

        # 統計分析
        stats_results = self.perform_statistical_analysis(df)

        # グラフ作成
        self.create_comparison_plots(df, stats_results)

        # レポート作成
        self.create_ab_test_report(df, stats_results)

        return df, stats_results

    def perform_statistical_analysis(self, df):
        """統計分析実行"""
        print("[STATS] Performing statistical analysis")

        stats_results = {}

        # 各タスクごとの比較
        for task in df['task'].unique():
            task_data = df[df['task'] == task]

            if len(task_data) >= 2:
                base_scores = task_data[task_data['model'] == 'Base Model']['accuracy'].values
                aegis_scores = task_data[task_data['model'] == 'AEGIS v2.1']['accuracy'].values

                if len(base_scores) > 0 and len(aegis_scores) > 0:
                    # t-test
                    t_stat, p_value = stats.ttest_ind(base_scores, aegis_scores, equal_var=False)

                    # Cohen's d (effect size)
                    mean_diff = np.mean(aegis_scores) - np.mean(base_scores)
                    pooled_std = np.sqrt((np.std(base_scores)**2 + np.std(aegis_scores)**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

                    # ANOVA (単純比較だが参考値として)
                    f_stat, anova_p = stats.f_oneway(base_scores, aegis_scores)

                    stats_results[task] = {
                        "base_mean": float(np.mean(base_scores)),
                        "base_std": float(np.std(base_scores)),
                        "aegis_mean": float(np.mean(aegis_scores)),
                        "aegis_std": float(np.std(aegis_scores)),
                        "mean_difference": float(mean_diff),
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "cohens_d": float(cohens_d),
                        "anova_f": float(f_stat),
                        "anova_p": float(anova_p),
                        "significant": p_value < 0.05,
                        "effect_size_interpretation": self.interpret_effect_size(cohens_d)
                    }

        # 全体比較
        if len(df) > 0:
            overall_base = df[df['model'] == 'Base Model']['accuracy'].mean()
            overall_aegis = df[df['model'] == 'AEGIS v2.1']['accuracy'].mean()

            stats_results["overall"] = {
                "base_mean": float(overall_base),
                "aegis_mean": float(overall_aegis),
                "improvement": float(overall_aegis - overall_base),
                "improvement_pct": float((overall_aegis - overall_base) / overall_base * 100) if overall_base > 0 else 0
            }

        print(f"[STATS] Analysis completed for {len(stats_results)-1} tasks")
        return stats_results

    def interpret_effect_size(self, d):
        """効果量の解釈"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def create_comparison_plots(self, df, stats_results):
        """比較グラフ作成"""
        print("[PLOTS] Creating comparison plots")

        # スタイル設定
        plt.style.use('default')
        sns.set_palette("husl")

        # タスク別比較グラフ
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AEGIS v2.1 vs Base Model Performance Comparison', fontsize=16, fontweight='bold')

        # 1. 全体比較棒グラフ
        ax = axes[0, 0]
        overall_data = df.groupby('model')['accuracy'].agg(['mean', 'std']).reset_index()
        bars = ax.bar(overall_data['model'], overall_data['mean'], yerr=overall_data['std'],
                      capsize=5, alpha=0.8, color=['skyblue', 'lightcoral'])
        ax.set_title('Overall Performance Comparison', fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)

        # 値ラベル追加
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   '.3f', ha='center', va='bottom', fontweight='bold')

        # 2. タスク別比較
        ax = axes[0, 1]
        task_comparison = df.pivot_table(values='accuracy', index='task', columns='model', aggfunc='mean')
        task_comparison.plot(kind='bar', ax=ax, rot=45)
        ax.set_title('Task-wise Performance Comparison', fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3)

        # 3. 改善度分布
        ax = axes[1, 0]
        if "overall" in stats_results:
            improvement = stats_results["overall"]["improvement_pct"]
            colors = ['red' if x < 0 else 'green' for x in [improvement]]
            ax.bar(['AEGIS v2.1'], [improvement], color=colors, alpha=0.7)
            ax.set_title('Overall Performance Improvement', fontweight='bold')
            ax.set_ylabel('Improvement (%)')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 4. 統計的有意性の表示
        ax = axes[1, 1]
        significant_tasks = []
        non_significant_tasks = []

        for task, stats in stats_results.items():
            if task != "overall" and "significant" in stats:
                if stats["significant"]:
                    significant_tasks.append((task, stats["cohens_d"]))
                else:
                    non_significant_tasks.append((task, stats["cohens_d"]))

        if significant_tasks:
            tasks, effects = zip(*significant_tasks)
            ax.bar(tasks, effects, color='green', alpha=0.7, label='Significant')
        if non_significant_tasks:
            tasks, effects = zip(*non_significant_tasks)
            ax.bar(tasks, effects, color='gray', alpha=0.7, label='Non-significant')

        ax.set_title('Effect Size by Task (Cohen\'s d)', fontweight='bold')
        ax.set_ylabel('Effect Size')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plot_path = self.results_dir / "ab_test_comparison_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[PLOTS] Comparison plots saved to {plot_path}")
        plt.close()

        # エラーバー付き詳細グラフ
        self.create_detailed_errorbar_plot(df)

    def create_detailed_errorbar_plot(self, df):
        """エラーバー付き詳細グラフ作成"""
        print("[PLOTS] Creating detailed errorbar plot")

        plt.figure(figsize=(12, 8))

        # データ準備
        summary = df.groupby(['model', 'task'])['accuracy'].agg(['mean', 'std', 'count']).reset_index()

        # モデル別カラーマッピング
        colors = {'Base Model': 'skyblue', 'AEGIS v2.1': 'lightcoral'}

        # タスクごとにプロット
        tasks = sorted(df['task'].unique())
        x_pos = np.arange(len(tasks))
        width = 0.35

        for i, model in enumerate(['Base Model', 'AEGIS v2.1']):
            model_data = summary[summary['model'] == model]

            means = []
            errors = []
            for task in tasks:
                task_data = model_data[model_data['task'] == task]
                if len(task_data) > 0:
                    means.append(task_data['mean'].iloc[0])
                    errors.append(task_data['std'].iloc[0])
                else:
                    means.append(0)
                    errors.append(0)

            plt.bar(x_pos + i*width, means, width, yerr=errors,
                   label=model, color=colors[model], alpha=0.8,
                   capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})

        plt.xlabel('Task', fontweight='bold')
        plt.ylabel('Accuracy', fontweight='bold')
        plt.title('AEGIS v2.1 vs Base Model: Task Performance with Error Bars',
                 fontweight='bold', fontsize=14)
        plt.xticks(x_pos + width/2, tasks, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.0)

        # 値ラベル追加
        for i, model in enumerate(['Base Model', 'AEGIS v2.1']):
            model_data = summary[summary['model'] == model]
            for j, task in enumerate(tasks):
                task_data = model_data[model_data['task'] == task]
                if len(task_data) > 0:
                    height = task_data['mean'].iloc[0]
                    plt.text(x_pos[j] + i*width, height + 0.02,
                           '.3f', ha='center', va='bottom',
                           fontweight='bold', fontsize=8)

        plt.tight_layout()
        errorbar_path = self.results_dir / "ab_test_errorbar_detailed.png"
        plt.savefig(errorbar_path, dpi=300, bbox_inches='tight')
        print(f"[PLOTS] Errorbar plot saved to {errorbar_path}")
        plt.close()

    def create_ab_test_report(self, df, stats_results):
        """A/Bテストレポート作成"""
        print("[REPORT] Creating A/B test report")

        report = f"""# AEGIS v2.1 A/B Test Report

## Overview
This report presents the results of A/B testing between the base model (Borea-Phi-3.5-mini-Instruct-Jp) and AEGIS v2.1, an enhanced version with SO(8) optimization and fine-tuning.

## Models Compared
- **Base Model**: {self.base_model}
- **AEGIS v2.1**: SO(8) optimized Phi-3.5 with fine-tuning on 50,000 samples

## Evaluation Tasks
- **ELYZA-100**: Comprehensive Japanese language understanding
- **JCommonsenseQA**: Japanese commonsense reasoning
- **JSQuAD**: Japanese question answering
- **JNLI**: Japanese natural language inference
- **MGSM**: Multilingual grade school math
- **GSM8K**: Grade school math problems

## Statistical Analysis Results

### Overall Performance
"""

        if "overall" in stats_results:
            overall = stats_results["overall"]
            report += f"""- **Base Model Mean**: {overall['base_mean']:.4f}
- **AEGIS v2.1 Mean**: {overall['aegis_mean']:.4f}
- **Improvement**: {overall['improvement']:.4f} ({overall['improvement_pct']:+.2f}%)

"""

        report += """
### Task-wise Statistical Analysis

| Task | Base Mean | AEGIS Mean | Mean Diff | t-statistic | p-value | Cohen's d | Effect Size | Significant | ANOVA F | ANOVA p |
|------|-----------|------------|-----------|-------------|---------|-----------|-------------|-------------|----------|----------|
"""

        for task, stats in stats_results.items():
            if task != "overall":
                report += f"""| {task} | {stats['base_mean']:.4f} | {stats['aegis_mean']:.4f} | {stats['mean_difference']:+.4f} | {stats['t_statistic']:.3f} | {stats['p_value']:.4f} | {stats['cohens_d']:.3f} | {stats['effect_size_interpretation']} | {'[OK]' if stats['significant'] else '[NG]'} | {stats['anova_f']:.3f} | {stats['anova_p']:.4f} |
"""

        report += f"""

## Key Findings

### Performance Improvements
"""

        improvements = []
        for task, stats in stats_results.items():
            if task != "overall" and stats['mean_difference'] > 0:
                improvements.append((task, stats['mean_difference'], stats['significant']))

        if improvements:
            improvements.sort(key=lambda x: x[1], reverse=True)
            for task, diff, sig in improvements:
                sig_mark = " (significant)" if sig else ""
                report += f"- **{task}**: +{diff:.4f}{sig_mark}\n"
        else:
            report += "No significant improvements detected.\n"

        report += f"""

### Statistical Significance
- **Significantly improved tasks**: {sum(1 for s in stats_results.values() if isinstance(s, dict) and s.get('significant', False))}
- **Total tasks analyzed**: {len([s for s in stats_results.keys() if s != 'overall'])}
- **Significance threshold**: p < 0.05

### Effect Sizes
- **Large effect (d ≥ 0.8)**: {sum(1 for s in stats_results.values() if isinstance(s, dict) and abs(s.get('cohens_d', 0)) >= 0.8)}
- **Medium effect (0.5 ≤ d < 0.8)**: {sum(1 for s in stats_results.values() if isinstance(s, dict) and 0.5 <= abs(s.get('cohens_d', 0)) < 0.8)}
- **Small effect (0.2 ≤ d < 0.5)**: {sum(1 for s in stats_results.values() if isinstance(s, dict) and 0.2 <= abs(s.get('cohens_d', 0)) < 0.5)}

## Conclusion

AEGIS v2.1 demonstrates """

        if "overall" in stats_results and stats_results["overall"]["improvement"] > 0:
            report += f"""an overall improvement of {stats_results['overall']['improvement_pct']:+.2f}% compared to the base model."""
        else:
            report += "mixed performance compared to the base model."

        report += """

The statistical analysis shows """

        sig_count = sum(1 for s in stats_results.values() if isinstance(s, dict) and s.get('significant', False))
        if sig_count > 0:
            report += f"""significant improvements in {sig_count} out of {len([s for s in stats_results.keys() if s != 'overall'])} tasks."""
        else:
            report += "no statistically significant differences in most tasks."

        report += f"""

## Files Generated
- `ab_test_raw_results.json`: Raw evaluation results
- `ab_test_comparison_plots.png`: Performance comparison plots
- `ab_test_errorbar_detailed.png`: Detailed errorbar plots
- `ab_test_statistics.json`: Statistical analysis results

---
*Report generated automatically by AEGIS A/B testing framework*
"""

        # レポート保存
        report_path = self.results_dir / "ab_test_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # 統計結果をJSONで保存
        stats_path = self.results_dir / "ab_test_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_results, f, indent=2)

        print(f"[REPORT] Report saved to {report_path}")
        print(f"[STATS] Statistics saved to {stats_path}")

def main():
    """メイン実行関数"""
    print("[START] AEGIS v2.1 A/B Testing Framework")
    print("=" * 60)

    tester = AEGISABTester()

    # A/Bテスト実行
    raw_results = tester.run_ab_test()

    # 結果分析
    if raw_results:
        df, stats_results = tester.analyze_results(raw_results)

        print("\n[SUCCESS] A/B testing completed!")
        print(f"Results saved to: {tester.results_dir}")

        # HFアップロード用パッケージ作成
        tester.create_hf_submission_package(df, stats_results)
    else:
        print("\n[ERROR] A/B testing failed")

def create_hf_submission_package(self, df, stats_results):
    """HFアップロード用パッケージ作成"""
    print("[HF-PACKAGE] Creating HF submission package")

    package_dir = Path("hf_upload_package")
    package_dir.mkdir(exist_ok=True)

    # 統計サマリー作成
    summary = {
        "model_name": "AEGIS v2.1",
        "base_model": "Borea-Phi-3.5-mini-Instruct-Jp",
        "improvement_summary": stats_results.get("overall", {}),
        "task_results": {},
        "statistical_analysis": {}
    }

    for task, stats in stats_results.items():
        if task != "overall":
            summary["task_results"][task] = {
                "base_accuracy": stats["base_mean"],
                "aegis_accuracy": stats["aegis_mean"],
                "improvement": stats["mean_difference"],
                "significant": stats["significant"],
                "effect_size": stats["cohens_d"]
            }
            summary["statistical_analysis"][task] = {
                "t_statistic": stats["t_statistic"],
                "p_value": stats["p_value"],
                "cohens_d": stats["cohens_d"],
                "anova_f": stats["anova_f"],
                "anova_p": stats["anova_p"]
            }

    # パッケージ保存
    with open(package_dir / "evaluation_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # グラフをコピー
    import shutil
    for plot_file in ["ab_test_comparison_plots.png", "ab_test_errorbar_detailed.png"]:
        src = self.results_dir / plot_file
        if src.exists():
            shutil.copy2(src, package_dir / plot_file)

    # レポートをコピー
    shutil.copy2(self.results_dir / "ab_test_report.md", package_dir / "EVALUATION_REPORT.md")

    print(f"[HF-PACKAGE] Package created at: {package_dir}")
    print("Contents:")
    for item in package_dir.glob("*"):
        print(f"  - {item.name}")

if __name__ == "__main__":
    main()
