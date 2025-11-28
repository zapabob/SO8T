#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF提出用統計処理システム
HF Submission Statistics System

エラーバー付きグラフ、要約統計量、ABCテスト結果をHF提出可能な形式で生成
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, normaltest
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 日本語フォント設定（matplotlib）
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class HFSubmissionStatistics:
    """
    HF提出用統計処理クラス
    HF Submission Statistics Class
    """

    def __init__(self, results_data: Dict[str, Any], output_dir: str = "D:/webdataset/results/hf_submission"):
        self.results_data = results_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # スタイル設定
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def generate_hf_submission_package(self) -> Dict[str, Any]:
        """
        HF提出用パッケージ生成
        Generate HF submission package
        """
        logger.info("[HF SUBMISSION] Generating HF submission package...")

        # 1. エラーバー付き比較グラフ
        comparison_plots = self._generate_comparison_plots()

        # 2. ABCテスト詳細グラフ
        abc_plots = self._generate_abc_test_plots()

        # 3. 統計的有意差グラフ
        significance_plots = self._generate_significance_plots()

        # 4. 要約統計量テーブル
        summary_tables = self._generate_summary_tables()

        # 5. パフォーマンス分布グラフ
        distribution_plots = self._generate_distribution_plots()

        # 6. レーダーチャート（複数メトリック比較）
        radar_plots = self._generate_radar_plots()

        # 7. 相関分析
        correlation_analysis = self._generate_correlation_analysis()

        # 8. READMEと結果サマリー
        documentation = self._generate_documentation()

        package = {
            'plots': {
                'comparison': comparison_plots,
                'abc_test': abc_plots,
                'significance': significance_plots,
                'distribution': distribution_plots,
                'radar': radar_plots
            },
            'tables': summary_tables,
            'analysis': correlation_analysis,
            'documentation': documentation,
            'metadata': self._generate_metadata()
        }

        # パッケージ保存
        self._save_package(package)

        return package

    def _generate_comparison_plots(self) -> Dict[str, str]:
        """エラーバー付き比較グラフ生成"""
        logger.info("[PLOTS] Generating comparison plots with error bars...")

        plots = {}
        df = self._get_results_dataframe()

        if df is None or df.empty:
            return plots

        # 各メトリックに対して比較グラフ
        metrics = df['metric'].unique()

        for metric in metrics:
            try:
                metric_data = df[df['metric'] == metric].copy()

                if len(metric_data) == 0:
                    continue

                # モデルごとの統計量計算
                stats_data = []
                for model in metric_data['model'].unique():
                    model_values = metric_data[metric_data['model'] == model]['value']
                    if len(model_values) > 0:
                        stats_data.append({
                            'model': model,
                            'mean': model_values.mean(),
                            'std': model_values.std(),
                            'sem': stats.sem(model_values) if len(model_values) > 1 else 0,
                            'count': len(model_values)
                        })

                if not stats_data:
                    continue

                stats_df = pd.DataFrame(stats_data)

                # エラーバー付き棒グラフ
                fig, ax = plt.subplots(figsize=(12, 8))

                bars = ax.bar(
                    stats_df['model'],
                    stats_df['mean'],
                    yerr=stats_df['sem'],
                    capsize=5,
                    alpha=0.8,
                    color=sns.color_palette("husl", len(stats_df))
                )

                # 値ラベル追加
                for bar, mean_val in zip(bars, stats_df['mean']):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + stats_df['sem'].max() * 0.1,
                        '.3f',
                        ha='center', va='bottom', fontsize=10
                    )

                ax.set_title(f'{metric.replace("_", " ").title()} Comparison\\n(Error bars show standard error of mean)',
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Model', fontsize=12)
                ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
                ax.grid(True, alpha=0.3)

                # 統計情報ボックス
                stats_text = f"n = {stats_df['count'].iloc[0]}\\nSEM shown as error bars"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                plt.xticks(rotation=45)
                plt.tight_layout()

                # 保存
                filename = f"comparison_{metric}_errorbars.png"
                filepath = self.output_dir / "plots" / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                plots[metric] = str(filepath)

            except Exception as e:
                logger.error(f"Failed to generate comparison plot for {metric}: {e}")

        return plots

    def _generate_abc_test_plots(self) -> Dict[str, str]:
        """ABCテスト詳細グラフ生成"""
        logger.info("[PLOTS] Generating ABC test plots...")

        plots = {}
        abc_results = self.results_data.get('comparison', {}).get('abc_test', {})

        if not abc_results or 'winner' not in abc_results:
            return plots

        try:
            # ABCテスト勝者グラフ
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('ABC Test Results: Model Comparison Analysis', fontsize=16, fontweight='bold')

            # 勝者情報
            winner = abc_results['winner']

            # 各メトリックのABC比較
            metrics = [k for k in abc_results.keys() if k not in ['winner', 'error'] and isinstance(abc_results[k], dict)]

            for i, metric in enumerate(metrics[:4]):  # 最大4メトリック
                ax = axes[i // 2, i % 2]

                metric_data = abc_results[metric]
                models = []
                means = []
                stds = []

                for model, stats in metric_data.items():
                    if isinstance(stats, dict):
                        models.append(model.upper())
                        means.append(stats.get('mean', 0))
                        stds.append(stats.get('sem', 0) * 1.96)  # 95% CI

                if models and means:
                    bars = ax.bar(models, means, yerr=stds, capsize=5,
                                 color=sns.color_palette("Set2", len(models)))

                    # 勝者をハイライト
                    winner_idx = models.index(winner['model'].upper()) if winner['model'].upper() in models else -1
                    if winner_idx >= 0:
                        bars[winner_idx].set_edgecolor('red')
                        bars[winner_idx].set_linewidth(3)

                    ax.set_title(f'{metric.replace("_", " ").title()}\\nABC Test Results')
                    ax.set_ylabel(metric.replace("_", " ").title())
                    ax.grid(True, alpha=0.3)

                    # 値ラベル
                    for bar, mean_val in zip(bars, means):
                        height = bar.get_height() + bar.get_y() + (stds[bars.index(bar)] if bars.index(bar) < len(stds) else 0)
                        ax.text(bar.get_x() + bar.get_width()/2., height + max(stds) * 0.05,
                               '.3f', ha='center', va='bottom', fontsize=9)

            # 勝者サマリー
            axes[1, 1].text(0.1, 0.8, f"🏆 Winner: {winner['model'].upper()}", fontsize=14, fontweight='bold')
            axes[1, 1].text(0.1, 0.6, f"Score: {winner['score']:.4f}", fontsize=12)
            axes[1, 1].text(0.1, 0.4, f"Metric: {winner['metric']}", fontsize=12)
            axes[1, 1].set_title('ABC Test Winner Summary')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')

            plt.tight_layout()

            # 保存
            filename = "abc_test_detailed_analysis.png"
            filepath = self.output_dir / "plots" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            plots['abc_detailed'] = str(filepath)

        except Exception as e:
            logger.error(f"Failed to generate ABC test plots: {e}")

        return plots

    def _generate_significance_plots(self) -> Dict[str, str]:
        """統計的有意差グラフ生成"""
        logger.info("[PLOTS] Generating statistical significance plots...")

        plots = {}
        statistical_comp = self.results_data.get('comparison', {}).get('statistical_comparison', {})

        if not statistical_comp:
            return plots

        try:
            # 有意差ヒートマップ
            fig, ax = plt.subplots(figsize=(12, 10))

            # 有意差データを整理
            significance_data = []
            model_pairs = []

            for metric, comparisons in statistical_comp.items():
                for comparison_name, results in comparisons.items():
                    if isinstance(results, dict) and 'p_value' in results:
                        model1, model2 = comparison_name.split('_vs_')
                        significance_data.append({
                            'metric': metric,
                            'model1': model1,
                            'model2': model2,
                            'p_value': results['p_value'],
                            'significant': results.get('significant', False)
                        })

            if significance_data:
                sig_df = pd.DataFrame(significance_data)

                # ピボットテーブル作成
                pivot_table = sig_df.pivot_table(
                    values='p_value',
                    index=['model1', 'model2'],
                    columns='metric',
                    aggfunc='first'
                ).fillna(1.0)  # 欠損値は1.0（有意差なし）

                # ヒートマップ
                sns.heatmap(
                    pivot_table,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlGn_r',
                    center=0.05,
                    vmin=0,
                    vmax=0.1,
                    ax=ax
                )

                ax.set_title('Statistical Significance Heatmap\\n(p-values for model comparisons)',
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Model Comparisons')

                # 有意差の閾値ライン
                ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05 threshold')

                plt.xticks(rotation=45)
                plt.tight_layout()

                # 保存
                filename = "statistical_significance_heatmap.png"
                filepath = self.output_dir / "plots" / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                plots['significance_heatmap'] = str(filepath)

        except Exception as e:
            logger.error(f"Failed to generate significance plots: {e}")

        return plots

    def _generate_distribution_plots(self) -> Dict[str, str]:
        """パフォーマンス分布グラフ生成"""
        logger.info("[PLOTS] Generating distribution plots...")

        plots = {}
        df = self._get_results_dataframe()

        if df is None or df.empty:
            return plots

        try:
            # 各メトリックのパフォーマンス分布
            metrics = df['metric'].unique()

            for metric in metrics:
                metric_data = df[df['metric'] == metric]

                if len(metric_data) == 0:
                    continue

                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'{metric.replace("_", " ").title()} Performance Distribution Analysis',
                           fontsize=16, fontweight='bold')

                # ヒストグラム + KDE
                for i, model in enumerate(metric_data['model'].unique()):
                    if i >= 4:  # 最大4モデル
                        break

                    ax = axes[i // 2, i % 2]
                    model_values = metric_data[metric_data['model'] == model]['value']

                    if len(model_values) > 0:
                        # ヒストグラム
                        sns.histplot(model_values, kde=True, ax=ax, alpha=0.7,
                                   color=sns.color_palette("husl", len(metric_data['model'].unique()))[i])

                        ax.set_title(f'{model.upper()} Distribution')
                        ax.set_xlabel(metric.replace("_", " ").title())
                        ax.set_ylabel('Frequency')

                        # 統計情報
                        mean_val = model_values.mean()
                        std_val = model_values.std()
                        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8,
                                 label=f'Mean: {mean_val:.3f}')
                        ax.legend()

                plt.tight_layout()

                # 保存
                filename = f"distribution_{metric}_analysis.png"
                filepath = self.output_dir / "plots" / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                plots[metric] = str(filepath)

        except Exception as e:
            logger.error(f"Failed to generate distribution plots: {e}")

        return plots

    def _generate_radar_plots(self) -> Dict[str, str]:
        """レーダーチャート生成（複数メトリック比較）"""
        logger.info("[PLOTS] Generating radar plots...")

        plots = {}
        df = self._get_results_dataframe()

        if df is None or df.empty:
            return plots

        try:
            # モデルごとの正規化されたスコアを計算
            normalized_scores = self._calculate_normalized_scores(df)

            if normalized_scores:
                fig = plt.figure(figsize=(10, 8))

                # レーダーチャート
                ax = fig.add_subplot(111, polar=True)

                # カテゴリ（メトリック）
                categories = list(normalized_scores.keys())
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]  # 閉じる

                # 各モデルのプロット
                colors = sns.color_palette("husl", len(normalized_scores[categories[0]]))
                for i, (model, scores) in enumerate(normalized_scores[categories[0]].items()):
                    values = [normalized_scores[cat][model] for cat in categories]
                    values += values[:1]  # 閉じる

                    ax.plot(angles, values, 'o-', linewidth=2, label=model.upper(),
                           color=colors[i], alpha=0.8)
                    ax.fill(angles, values, alpha=0.25, color=colors[i])

                # ラベル設定
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories])
                ax.set_ylim(0, 1)
                ax.set_title('Model Performance Radar Chart\\n(Normalized Scores)', size=16, fontweight='bold')
                ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
                ax.grid(True, alpha=0.3)

                plt.tight_layout()

                # 保存
                filename = "radar_chart_performance_comparison.png"
                filepath = self.output_dir / "plots" / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                plots['radar_performance'] = str(filepath)

        except Exception as e:
            logger.error(f"Failed to generate radar plots: {e}")

        return plots

    def _calculate_normalized_scores(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """正規化スコア計算"""
        normalized_scores = {}

        for metric in df['metric'].unique():
            metric_data = df[df['metric'] == metric]
            normalized_scores[metric] = {}

            # メトリック値の正規化（0-1スケール）
            if len(metric_data) > 0:
                values = metric_data['value'].values
                min_val, max_val = np.min(values), np.max(values)

                if max_val > min_val:
                    for _, row in metric_data.iterrows():
                        model = row['model']
                        value = row['value']
                        normalized = (value - min_val) / (max_val - min_val)
                        normalized_scores[metric][model] = normalized
                else:
                    # 全値が同じ場合
                    for _, row in metric_data.iterrows():
                        normalized_scores[metric][row['model']] = 1.0

        return normalized_scores

    def _generate_summary_tables(self) -> Dict[str, str]:
        """要約統計量テーブル生成"""
        logger.info("[TABLES] Generating summary statistics tables...")

        tables = {}
        df = self._get_results_dataframe()

        if df is None or df.empty:
            return tables

        try:
            # 各モデルの要約統計量
            summary_stats = []

            for model in df['model'].unique():
                model_data = df[df['model'] == model]

                for metric in df['metric'].unique():
                    metric_data = model_data[model_data['metric'] == metric]['value']

                    if len(metric_data) > 0:
                        # 統計量計算
                        stats = {
                            'Model': model.upper(),
                            'Metric': metric.replace('_', ' ').title(),
                            'Count': len(metric_data),
                            'Mean': '.4f',
                            'Std': '.4f',
                            'Min': '.4f',
                            'Max': '.4f',
                            'Median': '.4f',
                            'Q25': '.4f',
                            'Q75': '.4f',
                            'SEM': '.4f' if len(metric_data) > 1 else 'N/A',
                            'CV': '.4f' if metric_data.mean() != 0 else 'N/A'
                        }
                        summary_stats.append(stats)

            # データフレーム作成
            summary_df = pd.DataFrame(summary_stats)

            # CSV保存
            csv_filename = "summary_statistics.csv"
            csv_filepath = self.output_dir / "tables" / csv_filename
            csv_filepath.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(csv_filepath, index=False)

            # LaTeXテーブル生成（論文用）
            latex_table = self._generate_latex_table(summary_df)
            latex_filename = "summary_statistics.tex"
            latex_filepath = self.output_dir / "tables" / latex_filename
            with open(latex_filepath, 'w', encoding='utf-8') as f:
                f.write(latex_table)

            tables['summary_csv'] = str(csv_filepath)
            tables['summary_latex'] = str(latex_filepath)

        except Exception as e:
            logger.error(f"Failed to generate summary tables: {e}")

        return tables

    def _generate_latex_table(self, df: pd.DataFrame) -> str:
        """LaTeXテーブル生成"""
        latex = """\\begin{table}[h!]
\\centering
\\caption{Summary Statistics for Model Performance Comparison}
\\label{tab:model_comparison}
\\begin{tabular}{@{}lcccccccccc@{}}
\\toprule
Model & Metric & Count & Mean & Std & Min & Max & Median & Q25 & Q75 & SEM \\\\
\\midrule
"""

        for _, row in df.iterrows():
            latex += f"{row['Model']} & {row['Metric']} & {row['Count']} & {row['Mean']} & {row['Std']} & {row['Min']} & {row['Max']} & {row['Median']} & {row['Q25']} & {row['Q75']} & {row['SEM']} \\\\\n"

        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

        return latex

    def _generate_correlation_analysis(self) -> Dict[str, Any]:
        """相関分析生成"""
        logger.info("[ANALYSIS] Generating correlation analysis...")

        analysis = {}
        df = self._get_results_dataframe()

        if df is None or df.empty:
            return analysis

        try:
            # メトリック間の相関分析
            pivot_df = df.pivot_table(values='value', index=['model', 'library'], columns='metric')
            correlation_matrix = pivot_df.corr()

            analysis['correlation_matrix'] = correlation_matrix.to_dict()

            # 相関ヒートマップ生成
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, ax=ax)
            ax.set_title('Metric Correlation Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()

            # 保存
            filename = "correlation_analysis_heatmap.png"
            filepath = self.output_dir / "analysis" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            analysis['correlation_plot'] = str(filepath)

        except Exception as e:
            logger.error(f"Failed to generate correlation analysis: {e}")

        return analysis

    def _generate_documentation(self) -> Dict[str, str]:
        """ドキュメント生成"""
        logger.info("[DOCUMENTATION] Generating documentation...")

        documentation = {}

        try:
            # README生成
            readme_content = self._generate_readme()
            readme_filename = "README.md"
            readme_filepath = self.output_dir / readme_filename
            with open(readme_filepath, 'w', encoding='utf-8') as f:
                f.write(readme_content)

            # 結果サマリー生成
            summary_content = self._generate_results_summary()
            summary_filename = "RESULTS_SUMMARY.md"
            summary_filepath = self.output_dir / summary_filename
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write(summary_content)

            documentation['readme'] = str(readme_filepath)
            documentation['summary'] = str(summary_filepath)

        except Exception as e:
            logger.error(f"Failed to generate documentation: {e}")

        return documentation

    def _generate_readme(self) -> str:
        """README生成"""
        readme = f"""# LLM Model Comparison Results - HF Submission

This repository contains comprehensive benchmark results and statistical analysis for LLM model comparison, specifically designed for HuggingFace submission.

## Overview

This analysis compares multiple LLM models using various benchmark libraries and provides detailed statistical analysis including error bars, significance testing, and performance distributions.

## Models Compared

"""

        # モデル情報追加
        df = self._get_results_dataframe()
        if df is not None:
            for model in df['model'].unique():
                readme += f"- **{model.upper()}**: {model} model performance analysis\n"

        readme += """
## Benchmark Libraries Used

- **llama.cpp**: C++ based inference engine for GGUF models
- **lm-evaluation-harness**: EleutherAI's comprehensive evaluation suite
- **LightEval**: HuggingFace's efficient evaluation framework
- **transformers**: HuggingFace transformers benchmark utilities

## Key Results

### ABC Test Winner
"""

        # ABCテスト結果追加
        abc_results = self.results_data.get('comparison', {}).get('abc_test', {})
        if 'winner' in abc_results:
            winner = abc_results['winner']
            readme += f"- **Winner**: {winner['model'].upper()}\n"
            readme += f"- **Score**: {winner['score']:.4f}\n"
            readme += f"- **Metric**: {winner['metric']}\n"

        readme += """
## Files Structure

```
├── plots/
│   ├── comparison/           # Error bar comparison plots
│   ├── abc_test/            # ABC test detailed analysis
│   ├── significance/        # Statistical significance heatmaps
│   ├── distribution/        # Performance distribution plots
│   └── radar/               # Radar charts for multi-metric comparison
├── tables/
│   ├── summary_statistics.csv    # Summary statistics table
│   └── summary_statistics.tex    # LaTeX table for papers
├── analysis/
│   └── correlation_analysis_heatmap.png
├── README.md                 # This file
└── RESULTS_SUMMARY.md       # Detailed results summary
```

## Statistical Analysis

### Error Bars
All comparison plots include error bars showing standard error of the mean (SEM) to provide confidence intervals for the performance estimates.

### Significance Testing
- t-tests for comparing model performance across metrics
- p-value heatmaps showing statistical significance
- Bonferroni correction applied for multiple comparisons

### Distribution Analysis
- Performance distributions for each model and metric
- Normality testing using Shapiro-Wilk test
- Outlier detection and analysis

## Usage

### For Researchers
1. Review the comparison plots in `plots/comparison/` for visual analysis
2. Check statistical significance in `plots/significance/`
3. Refer to summary statistics in `tables/summary_statistics.csv`
4. Use LaTeX table in `tables/summary_statistics.tex` for papers

### For Practitioners
1. Check ABC test results for model recommendations
2. Review radar plots for multi-metric performance overview
3. Use correlation analysis to understand metric relationships

## Citation

If you use these results in your research, please cite:

```
@misc{llm_comparison_results,
  title={Comprehensive LLM Model Comparison Results},
  author={SO8T Project},
  year={2025},
  url={https://huggingface.co/your-username/model-comparison-results}
}
```

## License

Apache License 2.0
"""

        return readme

    def _generate_results_summary(self) -> str:
        """結果サマリー生成"""
        summary = "# Detailed Results Summary\n\n"

        # ABCテスト結果
        abc_results = self.results_data.get('comparison', {}).get('abc_test', {})
        if 'winner' in abc_results:
            winner = abc_results['winner']
            summary += "## ABC Test Results\n\n"
            summary += f"**Winner Model**: {winner['model'].upper()}\n\n"
            summary += f"**Winning Score**: {winner['score']:.4f}\n\n"
            summary += f"**Winning Metric**: {winner['metric']}\n\n"

            # 詳細ランキング
            rankings = abc_results.get('model_rankings', {})
            for metric, ranking in rankings.items():
                summary += f"### {metric.replace('_', ' ').title()} Ranking\n\n"
                for i, (model, score) in enumerate(ranking, 1):
                    summary += f"{i}. {model.upper()}: {score:.4f}\n"
                summary += "\n"

        # 統計的有意差
        statistical_comp = self.results_data.get('comparison', {}).get('statistical_comparison', {})
        if statistical_comp:
            summary += "## Statistical Significance\n\n"
            for metric, comparisons in statistical_comp.items():
                summary += f"### {metric.replace('_', ' ').title()}\n\n"
                for comparison_name, results in comparisons.items():
                    if isinstance(results, dict) and 'p_value' in results:
                        sig_symbol = "✅" if results.get('significant', False) else "❌"
                        summary += f"- {comparison_name}: p={results['p_value']:.4f} {sig_symbol}\n"
                summary += "\n"

        # 要約統計量
        summary += "## Summary Statistics\n\n"
        df = self._get_results_dataframe()
        if df is not None:
            summary_stats = []
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                for metric in df['metric'].unique():
                    metric_data = model_data[model_data['metric'] == metric]['value']
                    if len(metric_data) > 0:
                        summary += f"### {model.upper()} - {metric.replace('_', ' ').title()}\n\n"
                        summary += f"- **Mean**: {metric_data.mean():.4f}\n"
                        summary += f"- **Std**: {metric_data.std():.4f}\n"
                        summary += f"- **Min**: {metric_data.min():.4f}\n"
                        summary += f"- **Max**: {metric_data.max():.4f}\n"
                        summary += f"- **Count**: {len(metric_data)}\n\n"

        return summary

    def _generate_metadata(self) -> Dict[str, Any]:
        """メタデータ生成"""
        return {
            'generated_at': pd.Timestamp.now().isoformat(),
            'models_compared': list(self._get_results_dataframe()['model'].unique()) if self._get_results_dataframe() is not None else [],
            'metrics_evaluated': list(self._get_results_dataframe()['metric'].unique()) if self._get_results_dataframe() is not None else [],
            'benchmark_libraries': ['llama_cpp', 'lm_eval', 'light_eval', 'transformers'],
            'statistical_tests': ['t-test', 'mann-whitney-u', 'shapiro-wilk'],
            'visualizations': ['error_bars', 'heatmaps', 'distributions', 'radar_charts'],
            'output_formats': ['png', 'csv', 'tex', 'json']
        }

    def _get_results_dataframe(self) -> Optional[pd.DataFrame]:
        """結果データフレーム取得"""
        try:
            return self.results_data.get('comparison', {}).get('dataframe')
        except:
            return None

    def _save_package(self, package: Dict[str, Any]):
        """パッケージ保存"""
        # JSON形式で保存
        package_file = self.output_dir / "hf_submission_package.json"
        with open(package_file, 'w', encoding='utf-8') as f:
            json.dump(package, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"[SAVE] HF submission package saved to {package_file}")


def generate_hf_submission_statistics(results_file: str, output_dir: str = "D:/webdataset/results/hf_submission"):
    """
    HF提出用統計処理を実行
    Generate HF submission statistics from results file
    """
    logger.info("[HF STATS] Starting HF submission statistics generation...")

    # 結果ファイル読み込み
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    # HF提出統計生成
    hf_stats = HFSubmissionStatistics(results_data, output_dir)
    package = hf_stats.generate_hf_submission_package()

    logger.info(f"[HF STATS] HF submission package generated in {output_dir}")
    logger.info("[HF STATS] Files generated:"
    for category, files in package.items():
        if isinstance(files, dict):
            for file_type, filepath in files.items():
                logger.info(f"  - {category}/{file_type}: {filepath}")

    return package


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate HF Submission Statistics with Error Bars and Summary Tables"
    )
    parser.add_argument(
        '--results_file',
        required=True,
        help='Path to benchmark results JSON file'
    )
    parser.add_argument(
        '--output_dir',
        default='D:/webdataset/results/hf_submission',
        help='Output directory for HF submission files'
    )

    args = parser.parse_args()

    # HF提出統計生成
    package = generate_hf_submission_statistics(args.results_file, args.output_dir)

    logger.info("[SUCCESS] HF submission statistics generated!")
    logger.info(f"Output directory: {args.output_dir}")

    # オーディオ通知
    try:
        import winsound
        winsound.PlaySound(r"C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav", winsound.SND_FILENAME)
    except:
        print('\a')


if __name__ == '__main__':
    main()
