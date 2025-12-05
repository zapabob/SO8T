#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B Test Statistical Analysis
ANOVA, Cohen's d, p-values, error bars付き統計分析

分析内容：
1. 記述統計
2. 推論統計（t検定、ANOVA）
3. 効果量分析（Cohen's d）
4. 信頼区間とエラーバー
5. 視覚化レポート
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def load_ab_test_results(results_dir: str) -> pd.DataFrame:
    """A/Bテスト結果の読み込み"""
    print("[STATS] Loading A/B test results...")

    results_files = list(Path(results_dir).glob("ab_test_results_*.json"))

    if not results_files:
        print("[ERROR] No A/B test results found")
        return pd.DataFrame()

    # 最新の結果ファイルを使用
    latest_file = max(results_files, key=lambda x: x.stat().st_mtime)

    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data["results"])
    print(f"[STATS] Loaded {len(df)} test results from {latest_file}")

    return df

def perform_descriptive_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """記述統計の実行"""
    print("[STATS] Computing descriptive statistics...")

    desc_stats = {}

    # モデル別統計
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        desc_stats[model] = {
            "count": len(model_df),
            "mean_inference_time": model_df["inference_time"].mean(),
            "std_inference_time": model_df["inference_time"].std(),
            "min_inference_time": model_df["inference_time"].min(),
            "max_inference_time": model_df["inference_time"].max(),
            "median_inference_time": model_df["inference_time"].median()
        }

    # カテゴリ別統計
    category_stats = {}
    for category in df["test_category"].unique():
        cat_df = df[df["test_category"] == category]
        category_stats[category] = {
            "baseline_mean": cat_df[cat_df["model"] == "baseline"]["inference_time"].mean(),
            "aegis_mean": cat_df[cat_df["model"] == "aegis"]["inference_time"].mean(),
            "baseline_std": cat_df[cat_df["model"] == "baseline"]["inference_time"].std(),
            "aegis_std": cat_df[cat_df["model"] == "aegis"]["inference_time"].std(),
            "sample_size": len(cat_df)
        }

    return {
        "model_stats": desc_stats,
        "category_stats": category_stats,
        "overall_stats": {
            "total_tests": len(df),
            "models_tested": len(df["model"].unique()),
            "categories_tested": len(df["test_category"].unique())
        }
    }

def perform_statistical_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """統計検定の実行"""
    print("[STATS] Performing statistical tests...")

    results = {}

    # モデル間の推論時間比較（t検定）
    baseline_times = df[df["model"] == "baseline"]["inference_time"]
    aegis_times = df[df["model"] == "aegis"]["inference_time"]

    if len(baseline_times) > 1 and len(aegis_times) > 1:
        t_stat, p_value = stats.ttest_ind(baseline_times, aegis_times)

        # Cohen's d (効果量)
        pooled_std = np.sqrt((baseline_times.var() + aegis_times.var()) / 2)
        cohens_d = abs(baseline_times.mean() - aegis_times.mean()) / pooled_std

        results["inference_time_comparison"] = {
            "test_type": "independent_t_test",
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "cohens_d": cohens_d,
            "effect_size_interpretation": interpret_cohens_d(cohens_d),
            "baseline_mean": baseline_times.mean(),
            "aegis_mean": aegis_times.mean(),
            "baseline_std": baseline_times.std(),
            "aegis_std": aegis_times.std()
        }

    # カテゴリ別ANOVA（3つ以上のグループがある場合）
    if len(df["test_category"].unique()) > 2:
        try:
            model = ols('inference_time ~ C(model) + C(test_category) + C(model):C(test_category)',
                       data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            results["anova_analysis"] = {
                "model_effect": {
                    "f_statistic": anova_table.loc["C(model)", "F"],
                    "p_value": anova_table.loc["C(model)", "PR(>F)"],
                    "significant": anova_table.loc["C(model)", "PR(>F)"] < 0.05
                },
                "category_effect": {
                    "f_statistic": anova_table.loc["C(test_category)", "F"],
                    "p_value": anova_table.loc["C(test_category)", "PR(>F)"],
                    "significant": anova_table.loc["C(test_category)", "PR(>F)"] < 0.05
                },
                "interaction_effect": {
                    "f_statistic": anova_table.loc["C(model):C(test_category)", "F"],
                    "p_value": anova_table.loc["C(model):C(test_category)", "PR(>F)"],
                    "significant": anova_table.loc["C(model):C(test_category)", "PR(>F)"] < 0.05
                }
            }
        except Exception as e:
            print(f"[WARNING] ANOVA analysis failed: {e}")

    return results

def interpret_cohens_d(d: float) -> str:
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

def calculate_confidence_intervals(df: pd.DataFrame, confidence: float = 0.95) -> Dict[str, Any]:
    """信頼区間の計算"""
    print("[STATS] Calculating confidence intervals...")

    ci_results = {}

    for model in df["model"].unique():
        model_data = df[df["model"] == "model"]["inference_time"]
        if len(model_data) > 1:
            mean = model_data.mean()
            sem = stats.sem(model_data)
            ci = stats.t.interval(confidence, len(model_data)-1, loc=mean, scale=sem)

            ci_results[model] = {
                "mean": mean,
                "confidence_interval": ci,
                "margin_of_error": (ci[1] - ci[0]) / 2,
                "confidence_level": confidence
            }

    return ci_results

def create_statistical_plots(df: pd.DataFrame, stats: Dict, output_dir: str):
    """統計分析の可視化"""
    print("[STATS] Creating statistical plots...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # スタイル設定
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # 1. 推論時間の箱ひげ図
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x="model", y="inference_time", showfliers=True)
    plt.title("Inference Time Distribution by Model", fontsize=14, fontweight='bold')
    plt.ylabel("Inference Time (seconds)")
    plt.grid(True, alpha=0.3)

    # 2. カテゴリ別推論時間
    plt.subplot(2, 2, 2)
    sns.barplot(data=df, x="test_category", y="inference_time", hue="model",
                errorbar="sd", capsize=0.1)
    plt.title("Inference Time by Category", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel("Inference Time (seconds)")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # 3. 分布比較
    plt.subplot(2, 2, 3)
    for model in df["model"].unique():
        model_data = df[df["model"] == model]["inference_time"]
        sns.kdeplot(data=model_data, label=model, fill=True, alpha=0.3)
    plt.title("Inference Time Density Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Inference Time (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Q-Qプロット（正規性チェック）
    plt.subplot(2, 2, 4)
    baseline_data = df[df["model"] == "baseline"]["inference_time"]
    aegis_data = df[df["model"] == "aegis"]["inference_time"]

    # 理論的な正規分布
    theoretical_quantiles = np.linspace(0.01, 0.99, len(baseline_data))
    normal_quantiles = stats.norm.ppf(theoretical_quantiles)

    plt.scatter(np.sort(stats.norm.cdf(baseline_data, baseline_data.mean(), baseline_data.std())),
               np.sort(baseline_data), alpha=0.6, label="Baseline", s=30)
    plt.scatter(np.sort(stats.norm.cdf(aegis_data, aegis_data.mean(), aegis_data.std())),
               np.sort(aegis_data), alpha=0.6, label="AEGIS", s=30)

    # 理想的な対角線
    min_val = min(baseline_data.min(), aegis_data.min())
    max_val = max(baseline_data.max(), aegis_data.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label="Ideal Normal")

    plt.title("Q-Q Plot (Normality Check)", fontsize=14, fontweight='bold')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "statistical_analysis_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 効果量プロット
    if "inference_time_comparison" in stats:
        plt.figure(figsize=(8, 6))
        comparison = stats["inference_time_comparison"]

        models = ["Baseline", "AEGIS"]
        means = [comparison["baseline_mean"], comparison["aegis_mean"]]
        errors = [comparison["baseline_std"], comparison["aegis_std"]]

        bars = plt.bar(models, means, yerr=errors, capsize=5,
                      color=['skyblue', 'lightcoral'], alpha=0.7)

        plt.title(f"Inference Time Comparison\nCohen's d = {comparison['cohens_d']:.3f} ({comparison['effect_size_interpretation']})",
                 fontsize=14, fontweight='bold')
        plt.ylabel("Inference Time (seconds)")
        plt.grid(True, alpha=0.3, axis='y')

        # 値ラベル追加
        for bar, mean in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_y() + 0.01,
                    f'{mean:.3f}s', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / "effect_size_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

def generate_comprehensive_report(desc_stats: Dict, stat_tests: Dict, ci_results: Dict, output_dir: str):
    """包括的なレポート生成"""
    print("[STATS] Generating comprehensive statistical report...")

    report_path = Path(output_dir) / f"comprehensive_statistical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    report = f"""# A/B Test Statistical Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive statistical analysis of the A/B testing results comparing Baseline and AEGIS models across multiple performance metrics.

## 1. Descriptive Statistics

### Overall Statistics
- **Total Tests:** {desc_stats['overall_stats']['total_tests']}
- **Models Tested:** {desc_stats['overall_stats']['models_tested']}
- **Categories Tested:** {desc_stats['overall_stats']['categories_tested']}

### Model Performance Statistics

"""

    # モデル別統計
    for model, stats in desc_stats['model_stats'].items():
        report += f"""#### {model.upper()} Model
- **Sample Size:** {stats['count']}
- **Mean Inference Time:** {stats['mean_inference_time']:.4f} seconds
- **Standard Deviation:** {stats['std_inference_time']:.4f} seconds
- **Median Inference Time:** {stats['median_inference_time']:.4f} seconds
- **Range:** {stats['min_inference_time']:.4f} - {stats['max_inference_time']:.4f} seconds

"""

    # 統計検定結果
    if "inference_time_comparison" in stat_tests:
        comp = stat_tests["inference_time_comparison"]
        report += f"""## 2. Statistical Tests

### Inference Time Comparison (Independent t-test)

- **Test Type:** Independent Samples t-test
- **t-statistic:** {comp['t_statistic']:.4f}
- **p-value:** {comp['p_value']:.4f}
- **Significant Difference:** {'Yes' if comp['significant'] else 'No'} (α = 0.05)
- **Cohen's d:** {comp['cohens_d']:.4f} ({comp['effect_size_interpretation']} effect)
- **Baseline Mean:** {comp['baseline_mean']:.4f} seconds
- **AEGIS Mean:** {comp['aegis_mean']:.4f} seconds

#### Effect Size Interpretation
Cohen's d = {comp['cohens_d']:.4f} indicates a **{comp['effect_size_interpretation']}** effect size.
- Negligible: < 0.2
- Small: 0.2 - 0.5
- Medium: 0.5 - 0.8
- Large: > 0.8

"""

    # ANOVA結果
    if "anova_analysis" in stat_tests:
        anova = stat_tests["anova_analysis"]
        report += f"""### ANOVA Analysis

#### Model Effect
- **F-statistic:** {anova['model_effect']['f_statistic']:.4f}
- **p-value:** {anova['model_effect']['p_value']:.4f}
- **Significant:** {'Yes' if anova['model_effect']['significant'] else 'No'}

#### Category Effect
- **F-statistic:** {anova['category_effect']['f_statistic']:.4f}
- **p-value:** {anova['category_effect']['p_value']:.4f}
- **Significant:** {'Yes' if anova['category_effect']['significant'] else 'No'}

#### Interaction Effect
- **F-statistic:** {anova['interaction_effect']['f_statistic']:.4f}
- **p-value:** {anova['interaction_effect']['p_value']:.4f}
- **Significant:** {'Yes' if anova['interaction_effect']['significant'] else 'No'}

"""

    # 信頼区間
    if ci_results:
        report += f"""## 3. Confidence Intervals (95%)

"""
        for model, ci_data in ci_results.items():
            report += f"""### {model.upper()} Model
- **Mean:** {ci_data['mean']:.4f} seconds
- **95% CI:** [{ci_data['confidence_interval'][0]:.4f}, {ci_data['confidence_interval'][1]:.4f}] seconds
- **Margin of Error:** ±{ci_data['margin_of_error']:.4f} seconds

"""

    # カテゴリ別分析
    report += f"""## 4. Category-wise Analysis

"""
    for category, stats in desc_stats['category_stats'].items():
        report += f"""### {category.replace('_', ' ').title()}
- **Baseline Mean:** {stats['baseline_mean']:.4f} seconds
- **AEGIS Mean:** {stats['aegis_mean']:.4f} seconds
- **Performance Ratio (Baseline/AEGIS):** {stats['baseline_mean']/stats['aegis_mean']:.2f}x
- **Sample Size:** {stats['sample_size']}

"""

    # 結論
    if "inference_time_comparison" in stat_tests:
        comp = stat_tests["inference_time_comparison"]
        winner = "AEGIS" if comp['aegis_mean'] < comp['baseline_mean'] else "Baseline"
        improvement = abs(comp['baseline_mean'] - comp['aegis_mean']) / comp['baseline_mean'] * 100

        report += f"""## 5. Conclusions

### Performance Comparison
The statistical analysis reveals that **{winner}** demonstrates {'better' if winner == 'AEGIS' else 'equivalent'} performance compared to the baseline.

- **Winner:** {winner}
- **Performance Improvement:** {improvement:.1f}%
- **Effect Size:** {comp['effect_size_interpretation']} ({comp['cohens_d']:.3f})
- **Statistical Significance:** {'Significant' if comp['significant'] else 'Not significant'} (p = {comp['p_value']:.4f})

### Recommendations
1. {'Deploy AEGIS model for production use' if winner == 'AEGIS' else 'Continue with Baseline model'}
2. {'Further optimization may be beneficial' if not comp['significant'] else 'Results are statistically robust'}
3. Consider additional testing with larger sample sizes for more precise effect size estimation

### Data Quality Notes
- All statistical tests assume normality and independence of observations
- Effect sizes should be interpreted in the context of the specific use case
- Confidence intervals provide a range of plausible values for the true population parameters

---
*Report generated automatically by MOONSHOT Statistical Analysis Framework*
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"[STATS] Comprehensive report saved to {report_path}")
    return report_path

def main():
    """メイン統計分析実行関数"""
    print("📊 Starting A/B Test Statistical Analysis...")
    print("=" * 60)

    # 結果ディレクトリ
    results_dir = "results/ab_test_results"
    output_dir = "results/ab_test_results/statistics"

    # A/Bテスト結果読み込み
    df = load_ab_test_results(results_dir)

    if df.empty:
        print("[ERROR] No A/B test results to analyze")
        return 1

    # 記述統計
    desc_stats = perform_descriptive_statistics(df)

    # 統計検定
    stat_tests = perform_statistical_tests(df)

    # 信頼区間
    ci_results = calculate_confidence_intervals(df)

    # 可視化
    create_statistical_plots(df, stat_tests, output_dir)

    # 包括的レポート生成
    report_path = generate_comprehensive_report(desc_stats, stat_tests, ci_results, output_dir)

    print(f"\n🎉 Statistical analysis completed!")
    print(f"📊 Results saved to {output_dir}")
    print(f"📋 Comprehensive report: {report_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())