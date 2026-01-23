#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABCテスト結果をエラーバー付きグラフで図式化
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

# スタイル設定
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (15, 10)

class ABCTestVisualizer:
    """ABCテスト結果可視化クラス"""

    def __init__(self, results_file="abc_test_results.json"):
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.results = data["results"]
        self.models = list(self.results.keys())
        self.benchmarks = list(self.results[self.models[0]].keys())

        # モデル名マッピング
        self.model_names = {
            "microsoft_phi35": "Microsoft Phi-3.5",
            "boreas_phi35": "Boreas Phi-3.5",
            "aegis_v25": "AEGIS v2.5"
        }

        # ベンチマーク名マッピング
        self.benchmark_names = {
            "gsm8k": "GSM8K",
            "math": "MATH",
            "arc_challenge": "ARC-Challenge",
            "mmlu": "MMLU",
            "elyza_tasks": "ELYZA Tasks"
        }

    def create_performance_comparison_chart(self):
        """各ベンチマークごとのモデル比較グラフ"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for i, benchmark in enumerate(self.benchmarks):
            ax = axes[i]

            # データ準備
            models_data = []
            means = []
            errors = []

            for j, model_key in enumerate(self.models):
                model_data = self.results[model_key][benchmark]
                mean_val = model_data["mean"]
                std_val = model_data["std"]

                means.append(mean_val)
                errors.append(std_val)
                models_data.append({
                    'model': self.model_names[model_key],
                    'mean': mean_val,
                    'std': std_val,
                    'color': colors[j]
                })

            # バープロット作成
            bars = ax.bar(self.model_names.values(), means, yerr=errors,
                         capsize=8, color=colors, alpha=0.7, width=0.6)

            # 値ラベル追加
            for bar, mean_val in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(errors)*0.1,
                       f'{mean_val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

            # グラフ設定
            ax.set_title(f'{self.benchmark_names[benchmark]} Performance Comparison',
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_xlabel('Models', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

            # Y軸範囲設定
            max_val = max(means) + max(errors) * 1.2
            min_val = min(means) - max(errors) * 0.2
            ax.set_ylim(max(0, min_val), max_val)

            # 凡例
            if i == 0:
                ax.legend(bars, self.model_names.values(), loc='upper left', bbox_to_anchor=(0, 1.15))

        # 空のサブプロットを非表示
        if len(self.benchmarks) < 6:
            for i in range(len(self.benchmarks), 6):
                axes[i].set_visible(False)

        plt.suptitle('ABC Test Results: Model Performance Comparison (with Error Bars)',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('abc_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("[SAVE] Performance comparison chart saved as 'abc_performance_comparison.png'")

    def create_benchmark_overview_chart(self):
        """ベンチマークごとの性能概要グラフ"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # データ準備
        data = []
        for model_key, model_data in self.results.items():
            for benchmark, bench_data in model_data.items():
                data.append({
                    'Model': self.model_names[model_key],
                    'Benchmark': self.benchmark_names[benchmark],
                    'Mean': bench_data['mean'],
                    'Std': bench_data['std']
                })

        df = pd.DataFrame(data)

        # グループ化バープロット
        models = df['Model'].unique()
        benchmarks = df['Benchmark'].unique()

        x = np.arange(len(benchmarks))
        width = 0.25

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            means = []
            errors = []

            for bench in benchmarks:
                row = model_data[model_data['Benchmark'] == bench].iloc[0]
                means.append(row['Mean'])
                errors.append(row['Std'])

            bars = ax.bar(x + i*width, means, width, yerr=errors,
                         label=model, color=colors[i], alpha=0.8, capsize=5)

            # 値ラベル
            for bar, mean_val in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Benchmarks', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('ABC Test: Benchmark-wise Performance Overview',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('abc_benchmark_overview.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("[SAVE] Benchmark overview chart saved as 'abc_benchmark_overview.png'")

    def create_significance_visualization(self):
        """統計的有意性を視覚化したグラフ"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # 有意性の計算（簡易版）
        significance_data = []

        for benchmark in self.benchmarks:
            aegis_data = self.results['aegis_v25'][benchmark]
            ms_data = self.results['microsoft_phi35'][benchmark]
            boreas_data = self.results['boreas_phi35'][benchmark]

            # AEGIS vs Microsoft
            if self._calculate_significance(aegis_data, ms_data):
                significance_data.append({
                    'Benchmark': self.benchmark_names[benchmark],
                    'Comparison': 'AEGIS vs MS Phi-3.5',
                    'Improvement': aegis_data['mean'] - ms_data['mean'],
                    'Significant': True
                })

            # AEGIS vs Boreas
            if self._calculate_significance(aegis_data, boreas_data):
                significance_data.append({
                    'Benchmark': self.benchmark_names[benchmark],
                    'Comparison': 'AEGIS vs Boreas',
                    'Improvement': aegis_data['mean'] - boreas_data['mean'],
                    'Significant': True
                })

        if significance_data:
            df = pd.DataFrame(significance_data)

            # カラーマップ
            colors = ['#FF6B6B' if sig else '#cccccc' for sig in df['Significant']]

            bars = ax.barh(df['Benchmark'] + ' (' + df['Comparison'] + ')',
                          df['Improvement'], color=colors, alpha=0.7)

            # 値ラベル
            for bar, imp in zip(bars, df['Improvement']):
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                       f'{imp:+.1f}', ha='left', va='center', fontweight='bold')

            ax.set_xlabel('Performance Improvement (percentage points)', fontsize=12)
            ax.set_title('ABC Test: Statistically Significant Improvements (p < 0.05)',
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='x')
            ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.7)

            # 凡例
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#FF6B6B', label='Statistically Significant (p < 0.05)')]
            ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig('abc_significance_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("[SAVE] Significance visualization saved as 'abc_significance_visualization.png'")

    def create_industry_comparison_chart(self):
        """業界標準との比較グラフ"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # 業界標準データ
        industry_data = {
            'gsm8k': {'llama3_8b': 75.7, 'qwen2.5_7b': 84.1},
            'math': {'llama3_8b': 35.0, 'qwen2.5_7b': 41.0},
            'arc_challenge': {'llama3_8b': 78.6, 'qwen2.5_7b': 85.0},
            'mmlu': {'llama3_8b': 68.0, 'qwen2.5_7b': 72.0}
        }

        # データ準備
        benchmarks = ['gsm8k', 'math', 'arc_challenge', 'mmlu']
        x = np.arange(len(benchmarks))

        # AEGISのスコア
        aegis_scores = [self.results['aegis_v25'][b]['mean'] for b in benchmarks]
        aegis_errors = [self.results['aegis_v25'][b]['std'] for b in benchmarks]

        # 業界標準
        llama_scores = [industry_data[b]['llama3_8b'] for b in benchmarks]
        qwen_scores = [industry_data[b]['qwen2.5_7b'] for b in benchmarks]

        # プロット
        width = 0.25

        # AEGIS
        aegis_bars = ax.bar(x - width, aegis_scores, width, yerr=aegis_errors,
                           label='AEGIS v2.5', color='#45B7D1', alpha=0.8, capsize=5)

        # Llama-3-8B
        llama_bars = ax.bar(x, llama_scores, width,
                           label='Llama-3-8B', color='#FF6B6B', alpha=0.7)

        # Qwen2.5-7B
        qwen_bars = ax.bar(x + width, qwen_scores, width,
                          label='Qwen2.5-7B', color='#4ECDC4', alpha=0.7)

        # 値ラベル
        for bars in [aegis_bars, llama_bars, qwen_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('Benchmarks', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('ABC Test: AEGIS v2.5 vs Industry Leaders',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([self.benchmark_names[b] for b in benchmarks])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('abc_industry_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("[SAVE] Industry comparison chart saved as 'abc_industry_comparison.png'")

    def create_model_ranking_heatmap(self):
        """モデルランキングのヒートマップ"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # データ準備
        data = np.zeros((len(self.models), len(self.benchmarks)))

        for i, model_key in enumerate(self.models):
            for j, benchmark in enumerate(self.benchmarks):
                data[i, j] = self.results[model_key][benchmark]['mean']

        # ランキング計算（各ベンチマーク内で）
        ranked_data = np.zeros_like(data)
        for j in range(len(self.benchmarks)):
            scores = data[:, j]
            ranks = np.argsort(np.argsort(-scores)) + 1  # 降順ランキング
            ranked_data[:, j] = ranks

        # ヒートマップ
        im = ax.imshow(ranked_data, cmap='RdYlGn_r', aspect='auto', alpha=0.8)

        # テキストとグリッド
        for i in range(len(self.models)):
            for j in range(len(self.benchmarks)):
                text = ax.text(j, i, f'{ranked_data[i, j]:.0f}\\n({data[i, j]:.1f}%)',
                             ha='center', va='center', fontsize=11, fontweight='bold')

        ax.set_xticks(np.arange(len(self.benchmarks)))
        ax.set_yticks(np.arange(len(self.models)))
        ax.set_xticklabels([self.benchmark_names[b] for b in self.benchmarks])
        ax.set_yticklabels([self.model_names[m] for m in self.models])

        ax.set_title('ABC Test: Model Ranking by Benchmark\\n(1=Best, 3=Worst)',
                    fontsize=14, fontweight='bold', pad=20)

        # カラー設定
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Ranking (1=Best)', fontsize=12)

        plt.tight_layout()
        plt.savefig('abc_ranking_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("[SAVE] Ranking heatmap saved as 'abc_ranking_heatmap.png'")

    def _calculate_significance(self, data1, data2, alpha=0.05):
        """簡易有意性検定"""
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(data1['scores'], data2['scores'], equal_var=False)
        return p_value < alpha

    def create_all_charts(self):
        """全てのグラフを作成"""
        print("[CHART] Creating ABC Test Charts...")

        self.create_performance_comparison_chart()
        print("[CHART] 1/5 Performance comparison charts created")

        self.create_benchmark_overview_chart()
        print("[CHART] 2/5 Benchmark overview chart created")

        self.create_significance_visualization()
        print("[CHART] 3/5 Significance visualization created")

        self.create_industry_comparison_chart()
        print("[CHART] 4/5 Industry comparison chart created")

        self.create_model_ranking_heatmap()
        print("[CHART] 5/5 Ranking heatmap created")

        print("\\n[SUCCESS] All ABC test charts created successfully!")
        print("[FILES] Saved files:")
        print("   - abc_performance_comparison.png")
        print("   - abc_benchmark_overview.png")
        print("   - abc_significance_visualization.png")
        print("   - abc_industry_comparison.png")
        print("   - abc_ranking_heatmap.png")

def main():
    """メイン実行関数"""
    visualizer = ABCTestVisualizer()
    visualizer.create_all_charts()

if __name__ == "__main__":
    main()