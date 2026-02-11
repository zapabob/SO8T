#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
エラーバー付きABCベンチマーク統計可視化スクリプト
統計的分析結果を使用してエラーバー付きグラフを生成
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# 統計的分析スクリプトをインポート
import sys
sys.path.insert(0, str(Path(__file__).parent))
from statistical_abc_analysis import StatisticalABCAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# スタイル設定
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (15, 10)


class ABCBenchmarkVisualizer:
    """ABCベンチマーク統計可視化クラス"""

    def __init__(self, results_file: str, statistical_analysis_file: str = None):
        """
        初期化
        
        Args:
            results_file: ABCテスト結果JSONファイルのパス
            statistical_analysis_file: 統計的分析結果JSONファイルのパス（オプション）
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.results_dir = self.project_root / "results" / "abc_testing" / "visualizations"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 結果データ読み込み
        with open(results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        # 統計的分析結果読み込み（存在する場合）
        self.statistical_analysis = None
        if statistical_analysis_file and Path(statistical_analysis_file).exists():
            with open(statistical_analysis_file, 'r', encoding='utf-8') as f:
                self.statistical_analysis = json.load(f)
        else:
            # 統計的分析を実行
            logger.info("[VIZ] Running statistical analysis...")
            analyzer = StatisticalABCAnalyzer(results_data=self.results)
            self.statistical_analysis = analyzer.analyze_abc_results(
                apply_correction=True,
                correction_method='bonferroni'
            )
        
        self.models = ['A', 'B', 'C']
        self.model_names = {
            'A': 'Qwen2.5-7B (Base)',
            'B': 'SO8T-trained',
            'C': 'AEGIS-Phi3.5'
        }
        
        # ベンチマーク名マッピング
        self.benchmark_names = {
            'gsm8k': 'GSM8K',
            'math': 'MATH',
            'arc_easy': 'ARC-Easy',
            'arc_challenge': 'ARC-Challenge',
            'mmlu': 'MMLU',
            'bbh': 'BBH',
            'commonsenseqa': 'CommonsenseQA',
            'openbookqa': 'OpenBookQA',
            'socialiqa': 'SocialIQA',
            'piqa': 'PIQA',
            'winogrande': 'Winogrande',
            'boolq': 'BoolQ',
            'drop': 'DROP',
            'strategyqa': 'StrategyQA',
            'elyza_tasks_100': 'ELYZA-100'
        }
        
        # ベンチマークリストを取得
        if 'metadata' in self.results:
            self.benchmarks = self.results['metadata'].get('benchmarks', list(self.results.get('A', {}).keys()))
        else:
            self.benchmarks = list(self.results.get('A', {}).keys())
        
        # ベンチマーク分類
        self.industry_benchmarks = ['mmlu', 'bbh', 'commonsenseqa', 'openbookqa', 
                                   'socialiqa', 'piqa', 'winogrande', 'boolq']
        self.advanced_benchmarks = ['drop', 'strategyqa']
        self.japanese_benchmarks = ['elyza_tasks_100']

    def create_individual_benchmark_comparison(self):
        """個別ベンチマーク比較グラフ（エラーバー付き）"""
        n_benchmarks = len(self.benchmarks)
        n_cols = 3
        n_rows = (n_benchmarks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        if n_benchmarks == 1:
            axes = [axes]
        else:
            axes = axes.ravel()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, benchmark in enumerate(self.benchmarks):
            ax = axes[i]
            
            # データ準備
            means = []
            ci_lowers = []
            ci_uppers = []
            model_labels = []
            
            for model in self.models:
                if benchmark in self.statistical_analysis.get('error_bars', {}):
                    if model in self.statistical_analysis['error_bars'][benchmark]:
                        error_bars = self.statistical_analysis['error_bars'][benchmark][model]
                        means.append(error_bars['mean'] * 100)  # パーセント変換
                        ci_lowers.append(error_bars['ci_lower'] * 100)
                        ci_uppers.append(error_bars['ci_upper'] * 100)
                        model_labels.append(self.model_names[model])
            
            if means:
                x_pos = np.arange(len(means))
                errors_lower = [m - l for m, l in zip(means, ci_lowers)]
                errors_upper = [u - m for m, u in zip(means, ci_uppers)]
                
                bars = ax.bar(x_pos, means, yerr=[errors_lower, errors_upper],
                             capsize=8, color=colors[:len(means)], alpha=0.7, width=0.6)
                
                # 値ラベル追加
                for bar, mean_val, ci_l, ci_u in zip(bars, means, ci_lowers, ci_uppers):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (ci_u - mean_val) + 1,
                           f'{mean_val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                # 統計的有意性マーカー
                if benchmark in self.statistical_analysis.get('pairwise_comparisons', {}):
                    pairwise = self.statistical_analysis['pairwise_comparisons'][benchmark]
                    # A vs Cの有意性を表示
                    if 'A_vs_C' in pairwise:
                        p_value = pairwise['A_vs_C'].get('p_value', 1.0)
                        if p_value < 0.05:
                            # アスタリスクで有意性を表示
                            significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
                            ax.text(len(means) - 1, max(means) + max([u - m for m, u in zip(means, ci_uppers)]) + 2,
                                   significance, ha='center', va='bottom', fontsize=14, fontweight='bold', color='red')
                
                ax.set_title(f'{self.benchmark_names.get(benchmark, benchmark)} Performance',
                           fontsize=13, fontweight='bold', pad=15)
                ax.set_ylabel('Accuracy (%)', fontsize=11)
                ax.set_xlabel('Models', fontsize=11)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(model_labels, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim(0, max(ci_uppers) * 1.15 if ci_uppers else 100)
        
        # 空のサブプロットを非表示
        for i in range(len(self.benchmarks), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('ABC Test: Individual Benchmark Comparison (with 95% CI Error Bars)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = self.results_dir / 'abc_individual_benchmark_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[VIZ] Individual benchmark comparison saved to {output_file}")

    def create_comprehensive_benchmark_overview(self):
        """包括的ベンチマーク概要グラフ（エラーバー付き）"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # データ準備
        data_rows = []
        for benchmark in self.benchmarks:
            for model in self.models:
                if benchmark in self.statistical_analysis.get('error_bars', {}):
                    if model in self.statistical_analysis['error_bars'][benchmark]:
                        error_bars = self.statistical_analysis['error_bars'][benchmark][model]
                        data_rows.append({
                            'Benchmark': self.benchmark_names.get(benchmark, benchmark),
                            'Model': self.model_names[model],
                            'Mean': error_bars['mean'] * 100,
                            'CI_Lower': error_bars['ci_lower'] * 100,
                            'CI_Upper': error_bars['ci_upper'] * 100
                        })
        
        df = pd.DataFrame(data_rows)
        
        # データが空の場合はスキップ
        if df.empty:
            logger.warning("[VIZ] No data available for comprehensive benchmark overview")
            plt.close()
            return
        
        # グループ化バープロット
        benchmarks = df['Benchmark'].unique()
        models = df['Model'].unique()
        
        x = np.arange(len(benchmarks))
        width = 0.25
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            means = []
            errors_lower = []
            errors_upper = []
            
            for bench in benchmarks:
                row = model_data[model_data['Benchmark'] == bench]
                if len(row) > 0:
                    mean_val = row['Mean'].iloc[0]
                    ci_l = row['CI_Lower'].iloc[0]
                    ci_u = row['CI_Upper'].iloc[0]
                    means.append(mean_val)
                    errors_lower.append(mean_val - ci_l)
                    errors_upper.append(ci_u - mean_val)
                else:
                    means.append(0)
                    errors_lower.append(0)
                    errors_upper.append(0)
            
            bars = ax.bar(x + i*width, means, width, yerr=[errors_lower, errors_upper],
                         label=model, color=colors[i], alpha=0.8, capsize=5)
            
            # 値ラベル
            for bar, mean_val in zip(bars, means):
                if mean_val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + errors_upper[means.index(mean_val)] + 0.5,
                           f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Benchmarks', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('ABC Test: Comprehensive Benchmark Overview (with 95% CI Error Bars)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = self.results_dir / 'abc_comprehensive_benchmark_overview.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[VIZ] Comprehensive benchmark overview saved to {output_file}")

    def create_statistical_significance_visualization(self):
        """統計的有意性可視化グラフ"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        significance_data = []
        
        for benchmark in self.benchmarks:
            if benchmark in self.statistical_analysis.get('pairwise_comparisons', {}):
                pairwise = self.statistical_analysis['pairwise_comparisons'][benchmark]
                
                # A vs C (AEGIS vs Qwen-base)
                if 'A_vs_C' in pairwise:
                    result = pairwise['A_vs_C']
                    p_value = result.get('p_value', 1.0)
                    effect_size = result.get('effect_size', 0.0)
                    mean_diff = result.get('mean_diff', 0.0)
                    
                    if p_value < 0.05:
                        significance_data.append({
                            'Benchmark': self.benchmark_names.get(benchmark, benchmark),
                            'Comparison': 'AEGIS vs Qwen-base',
                            'Improvement': mean_diff * 100,  # パーセントポイント
                            'P_Value': p_value,
                            'Effect_Size': effect_size,
                            'Significant': True
                        })
                
                # B vs C (SO8T-trained vs AEGIS)
                if 'B_vs_C' in pairwise:
                    result = pairwise['B_vs_C']
                    p_value = result.get('p_value', 1.0)
                    effect_size = result.get('effect_size', 0.0)
                    mean_diff = result.get('mean_diff', 0.0)
                    
                    if p_value < 0.05:
                        significance_data.append({
                            'Benchmark': self.benchmark_names.get(benchmark, benchmark),
                            'Comparison': 'AEGIS vs SO8T-trained',
                            'Improvement': mean_diff * 100,
                            'P_Value': p_value,
                            'Effect_Size': effect_size,
                            'Significant': True
                        })
        
        if significance_data:
            df = pd.DataFrame(significance_data)
            
            # カラーマップ（p値に基づく）
            colors = []
            for p_val in df['P_Value']:
                if p_val < 0.001:
                    colors.append('#FF0000')  # 赤（極めて有意）
                elif p_val < 0.01:
                    colors.append('#FF6B6B')  # ピンク（非常に有意）
                else:
                    colors.append('#FFA500')  # オレンジ（有意）
            
            y_labels = [f"{row['Benchmark']} ({row['Comparison']})" for _, row in df.iterrows()]
            bars = ax.barh(y_labels, df['Improvement'], color=colors, alpha=0.7)
            
            # 値ラベル
            for bar, imp, p_val in zip(bars, df['Improvement'], df['P_Value']):
                width = bar.get_width()
                significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                ax.text(width + (0.5 if width >= 0 else -0.5), bar.get_y() + bar.get_height()/2,
                       f'{imp:+.1f}pp {significance}', ha='left' if width >= 0 else 'right',
                       va='center', fontweight='bold', fontsize=10)
            
            ax.set_xlabel('Performance Improvement (percentage points)', fontsize=12, fontweight='bold')
            ax.set_title('ABC Test: Statistically Significant Improvements\n'
                        '(*** p<0.001, ** p<0.01, * p<0.05, Bonferroni corrected)',
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='x')
            ax.axvline(x=0, color='black', linewidth=1, alpha=0.7)
            
            # 凡例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#FF0000', label='p < 0.001 (***)'),
                Patch(facecolor='#FF6B6B', label='p < 0.01 (**)'),
                Patch(facecolor='#FFA500', label='p < 0.05 (*)')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        output_file = self.results_dir / 'abc_statistical_significance.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[VIZ] Statistical significance visualization saved to {output_file}")

    def create_industry_comparison_chart(self):
        """業界標準比較グラフ"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # 業界標準データ（参考値）
        industry_data = {
            'gsm8k': {'Llama-3-8B': 75.7, 'Qwen2.5-7B': 84.1},
            'math': {'Llama-3-8B': 35.0, 'Qwen2.5-7B': 41.0},
            'arc_challenge': {'Llama-3-8B': 78.6, 'Qwen2.5-7B': 85.0},
            'mmlu': {'Llama-3-8B': 68.0, 'Qwen2.5-7B': 72.0}
        }
        
        # AEGIS (Model C)のスコアを取得
        aegis_scores = {}
        for benchmark in ['gsm8k', 'math', 'arc_challenge', 'mmlu']:
            if benchmark in self.statistical_analysis.get('error_bars', {}):
                if 'C' in self.statistical_analysis['error_bars'][benchmark]:
                    error_bars = self.statistical_analysis['error_bars'][benchmark]['C']
                    aegis_scores[benchmark] = error_bars['mean'] * 100
        
        # データ準備
        benchmarks = []
        aegis_values = []
        llama_values = []
        qwen_values = []
        
        for benchmark in ['gsm8k', 'math', 'arc_challenge', 'mmlu']:
            if benchmark in aegis_scores and benchmark in industry_data:
                benchmarks.append(self.benchmark_names.get(benchmark, benchmark))
                aegis_values.append(aegis_scores[benchmark])
                llama_values.append(industry_data[benchmark]['Llama-3-8B'])
                qwen_values.append(industry_data[benchmark]['Qwen2.5-7B'])
        
        x = np.arange(len(benchmarks))
        width = 0.25
        
        bars1 = ax.bar(x - width, aegis_values, width, label='AEGIS-Phi3.5', color='#45B7D1', alpha=0.8)
        bars2 = ax.bar(x, llama_values, width, label='Llama-3-8B', color='#95A5A6', alpha=0.8)
        bars3 = ax.bar(x + width, qwen_values, width, label='Qwen2.5-7B', color='#E74C3C', alpha=0.8)
        
        # 値ラベル
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Benchmarks', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('ABC Test: Industry Standard Comparison\n'
                    '(AEGIS-Phi3.5 vs Llama-3-8B vs Qwen2.5-7B)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = self.results_dir / 'abc_industry_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[VIZ] Industry comparison chart saved to {output_file}")

    def create_ranking_heatmap(self):
        """モデルランキングヒートマップ"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # ランキングデータ準備
        ranking_data = []
        for benchmark in self.benchmarks:
            benchmark_means = {}
            for model in self.models:
                if benchmark in self.statistical_analysis.get('error_bars', {}):
                    if model in self.statistical_analysis['error_bars'][benchmark]:
                        error_bars = self.statistical_analysis['error_bars'][benchmark][model]
                        benchmark_means[model] = error_bars['mean']
            
            # ランキング計算（1=最高, 3=最低）
            sorted_models = sorted(benchmark_means.items(), key=lambda x: x[1], reverse=True)
            for rank, (model, mean) in enumerate(sorted_models, 1):
                ranking_data.append({
                    'Benchmark': self.benchmark_names.get(benchmark, benchmark),
                    'Model': self.model_names[model],
                    'Rank': rank,
                    'Score': mean * 100
                })
        
        df = pd.DataFrame(ranking_data)
        
        # データが空の場合はスキップ
        if df.empty:
            logger.warning("[VIZ] No data available for ranking heatmap")
            plt.close()
            return
        
        # ピボットテーブル作成
        pivot_rank = df.pivot(index='Benchmark', columns='Model', values='Rank')
        pivot_score = df.pivot(index='Benchmark', columns='Model', values='Score')
        
        # ヒートマップ作成（ランキング）
        sns.heatmap(pivot_rank, annot=True, fmt='d', cmap='RdYlGn_r', 
                   cbar_kws={'label': 'Rank (1=Best, 3=Worst)'},
                   linewidths=0.5, linecolor='gray', ax=ax)
        
        ax.set_title('ABC Test: Model Ranking Heatmap\n'
                    '(1=Best Performance, 3=Worst Performance)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Benchmarks', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_file = self.results_dir / 'abc_ranking_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[VIZ] Ranking heatmap saved to {output_file}")

    def create_industry_standard_benchmark_chart(self):
        """業界標準ベンチマーク分類グラフ（MMLU含む、業界標準測定手法）"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # 業界標準ベンチマークのみをフィルタ
        industry_benchmarks_filtered = [b for b in self.benchmarks if b in self.industry_benchmarks]
        
        if not industry_benchmarks_filtered:
            logger.warning("[VIZ] No industry standard benchmarks found")
            return
        
        # データ準備
        data_rows = []
        for benchmark in industry_benchmarks_filtered:
            for model in self.models:
                if benchmark in self.statistical_analysis.get('error_bars', {}):
                    if model in self.statistical_analysis['error_bars'][benchmark]:
                        error_bars = self.statistical_analysis['error_bars'][benchmark][model]
                        data_rows.append({
                            'Benchmark': self.benchmark_names.get(benchmark, benchmark),
                            'Model': self.model_names[model],
                            'Mean': error_bars['mean'] * 100,
                            'CI_Lower': error_bars['ci_lower'] * 100,
                            'CI_Upper': error_bars['ci_upper'] * 100
                        })
        
        if not data_rows:
            logger.warning("[VIZ] No data for industry standard benchmarks")
            return
        
        df = pd.DataFrame(data_rows)
        
        # グループ化バープロット
        benchmarks = df['Benchmark'].unique()
        models = df['Model'].unique()
        
        x = np.arange(len(benchmarks))
        width = 0.25
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            means = []
            errors_lower = []
            errors_upper = []
            
            for bench in benchmarks:
                row = model_data[model_data['Benchmark'] == bench]
                if len(row) > 0:
                    mean_val = row['Mean'].iloc[0]
                    ci_l = row['CI_Lower'].iloc[0]
                    ci_u = row['CI_Upper'].iloc[0]
                    means.append(mean_val)
                    errors_lower.append(mean_val - ci_l)
                    errors_upper.append(ci_u - mean_val)
                else:
                    means.append(0)
                    errors_lower.append(0)
                    errors_upper.append(0)
            
            bars = ax.bar(x + i*width, means, width, yerr=[errors_lower, errors_upper],
                         label=model, color=colors[i], alpha=0.8, capsize=5)
            
            # 値ラベル
            for bar, mean_val in zip(bars, means):
                if mean_val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + errors_upper[means.index(mean_val)] + 0.5,
                           f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Industry Standard Benchmarks (MMLU with 5-shot protocol)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('ABC Test: Industry Standard Benchmarks Comparison\n'
                    '(MMLU, BBH, CommonsenseQA, OpenBookQA, SocialIQA, PIQA, Winogrande, BoolQ)\n'
                    'with 95% CI Error Bars (10 random seeds)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = self.results_dir / 'abc_industry_standard_benchmarks.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[VIZ] Industry standard benchmarks chart saved to {output_file}")

    def create_advanced_benchmark_chart(self):
        """高度ベンチマーク分類グラフ"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 高度ベンチマークのみをフィルタ
        advanced_benchmarks_filtered = [b for b in self.benchmarks if b in self.advanced_benchmarks]
        
        if not advanced_benchmarks_filtered:
            logger.warning("[VIZ] No advanced benchmarks found")
            return
        
        # データ準備
        data_rows = []
        for benchmark in advanced_benchmarks_filtered:
            for model in self.models:
                if benchmark in self.statistical_analysis.get('error_bars', {}):
                    if model in self.statistical_analysis['error_bars'][benchmark]:
                        error_bars = self.statistical_analysis['error_bars'][benchmark][model]
                        data_rows.append({
                            'Benchmark': self.benchmark_names.get(benchmark, benchmark),
                            'Model': self.model_names[model],
                            'Mean': error_bars['mean'] * 100,
                            'CI_Lower': error_bars['ci_lower'] * 100,
                            'CI_Upper': error_bars['ci_upper'] * 100
                        })
        
        if not data_rows:
            logger.warning("[VIZ] No data for advanced benchmarks")
            return
        
        df = pd.DataFrame(data_rows)
        
        # グループ化バープロット
        benchmarks = df['Benchmark'].unique()
        models = df['Model'].unique()
        
        x = np.arange(len(benchmarks))
        width = 0.25
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            means = []
            errors_lower = []
            errors_upper = []
            
            for bench in benchmarks:
                row = model_data[model_data['Benchmark'] == bench]
                if len(row) > 0:
                    mean_val = row['Mean'].iloc[0]
                    ci_l = row['CI_Lower'].iloc[0]
                    ci_u = row['CI_Upper'].iloc[0]
                    means.append(mean_val)
                    errors_lower.append(mean_val - ci_l)
                    errors_upper.append(ci_u - mean_val)
                else:
                    means.append(0)
                    errors_lower.append(0)
                    errors_upper.append(0)
            
            bars = ax.bar(x + i*width, means, width, yerr=[errors_lower, errors_upper],
                         label=model, color=colors[i], alpha=0.8, capsize=5)
            
            # 値ラベル
            for bar, mean_val in zip(bars, means):
                if mean_val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + errors_upper[means.index(mean_val)] + 0.5,
                           f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Advanced Benchmarks', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('ABC Test: Advanced Benchmarks Comparison\n'
                    '(DROP, StrategyQA) with 95% CI Error Bars (10 random seeds)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = self.results_dir / 'abc_advanced_benchmarks.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[VIZ] Advanced benchmarks chart saved to {output_file}")

    def create_elyza100_benchmark_chart(self):
        """ELIZA-100分類グラフ"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # ELIZA-100のみをフィルタ
        elyza_benchmarks = [b for b in self.benchmarks if b in self.japanese_benchmarks]
        
        if not elyza_benchmarks:
            logger.warning("[VIZ] No ELIZA-100 benchmarks found")
            return
        
        # データ準備
        data_rows = []
        for benchmark in elyza_benchmarks:
            for model in self.models:
                if benchmark in self.statistical_analysis.get('error_bars', {}):
                    if model in self.statistical_analysis['error_bars'][benchmark]:
                        error_bars = self.statistical_analysis['error_bars'][benchmark][model]
                        data_rows.append({
                            'Benchmark': self.benchmark_names.get(benchmark, benchmark),
                            'Model': self.model_names[model],
                            'Mean': error_bars['mean'] * 100,
                            'CI_Lower': error_bars['ci_lower'] * 100,
                            'CI_Upper': error_bars['ci_upper'] * 100
                        })
        
        if not data_rows:
            logger.warning("[VIZ] No data for ELIZA-100 benchmarks")
            return
        
        df = pd.DataFrame(data_rows)
        
        # バープロット
        models = df['Model'].unique()
        means = []
        errors_lower = []
        errors_upper = []
        
        for model in models:
            model_data = df[df['Model'] == model]
            if len(model_data) > 0:
                mean_val = model_data['Mean'].iloc[0]
                ci_l = model_data['CI_Lower'].iloc[0]
                ci_u = model_data['CI_Upper'].iloc[0]
                means.append(mean_val)
                errors_lower.append(mean_val - ci_l)
                errors_upper.append(ci_u - mean_val)
            else:
                means.append(0)
                errors_lower.append(0)
                errors_upper.append(0)
        
        x = np.arange(len(models))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars = ax.bar(x, means, yerr=[errors_lower, errors_upper],
                     color=colors[:len(models)], alpha=0.8, capsize=8)
        
        # 値ラベル
        for bar, mean_val, ci_u in zip(bars, means, errors_upper):
            if mean_val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + ci_u + 0.5,
                       f'{mean_val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('ABC Test: ELIZA-100 (Japanese Language Evaluation)\n'
                    'with 95% CI Error Bars (10 random seeds)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = self.results_dir / 'abc_elyza100_benchmark.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[VIZ] ELIZA-100 benchmark chart saved to {output_file}")

    def generate_all_visualizations(self):
        """すべての可視化を生成"""
        logger.info("[VIZ] Generating all visualizations...")
        
        self.create_individual_benchmark_comparison()
        self.create_comprehensive_benchmark_overview()
        self.create_statistical_significance_visualization()
        self.create_industry_comparison_chart()
        self.create_ranking_heatmap()
        
        # 分類グラフ（業界標準/高度/ELIZA-100）
        self.create_industry_standard_benchmark_chart()
        self.create_advanced_benchmark_chart()
        self.create_elyza100_benchmark_chart()
        
        logger.info(f"[VIZ] All visualizations saved to {self.results_dir}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ABC Benchmark Statistics Visualization')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to ABC test results JSON file')
    parser.add_argument('--statistical_analysis_file', type=str, default=None,
                       help='Path to statistical analysis results JSON file (optional)')
    
    args = parser.parse_args()
    
    # 可視化実行
    visualizer = ABCBenchmarkVisualizer(
        results_file=args.results_file,
        statistical_analysis_file=args.statistical_analysis_file
    )
    
    visualizer.generate_all_visualizations()
    
    logger.info("[VIZ] Visualization completed successfully")


if __name__ == "__main__":
    main()
