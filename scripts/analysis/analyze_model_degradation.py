#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8Tモデル劣化分析スクリプト
元モデル（model-a/modela）とSO8T（AEGIS）モデルの性能比較と劣化要因分析
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

class ModelDegradationAnalyzer:
    """モデル劣化分析クラス"""
    
    def __init__(self, results_file: Optional[Path], output_dir: Path):
        self.results_file = Path(results_file) if results_file else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        
    def load_results(self):
        """結果を読み込む"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        from scripts.analysis.visualize_benchmark_summary import BenchmarkVisualizer
        
        visualizer = BenchmarkVisualizer(self.output_dir)
        
        # 結果ファイルから直接読み込む
        input_dirs = [
            Path("_docs/benchmark_results"),
            Path("benchmarks/results"),
            Path("D:/webdataset/benchmark_results")
        ]
        
        visualizer.load_all_results(input_dirs)
        self.df = pd.DataFrame(visualizer.all_results)
        
        print(f"[INFO] Loaded {len(self.df)} results")
        return len(self.df) > 0
    
    def identify_baseline_and_so8t_models(self) -> Dict[str, List[str]]:
        """ベースラインモデルとSO8Tモデルを識別"""
        models = self.df['model'].unique()
        
        # ベースラインモデル（元モデル）
        baseline_models = [
            'model-a', 'model_a', 'modela',
            'Borea-Phi3.5-instinct-jp', 'borea-phi35-instinct-jp'
        ]
        
        # SO8T/AEGISモデル
        so8t_models = [
            'aegis', 'AEGIS', 'agiasi',
            'AEGIS-phi35-golden-sigmoid',
            'aegis_alpha_0_6', 'aegis-alpha-0.6',
            'aegis-q4km', 'aegis-alpha-0.6-q4km'
        ]
        
        identified = {
            'baseline': [],
            'so8t': []
        }
        
        for model in models:
            model_lower = str(model).lower()
            
            # ベースライン判定
            if any(bm.lower() in model_lower for bm in baseline_models):
                identified['baseline'].append(model)
            
            # SO8T判定
            if any(sm.lower() in model_lower for sm in so8t_models):
                identified['so8t'].append(model)
        
        print(f"[INFO] Baseline models: {identified['baseline']}")
        print(f"[INFO] SO8T models: {identified['so8t']}")
        
        return identified
    
    def calculate_degradation_metrics(self, model_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """劣化メトリクスを計算"""
        degradation_data = []
        
        # ベースラインとSO8Tのペアを作成
        for baseline in model_groups['baseline']:
            baseline_df = self.df[self.df['model'] == baseline]
            
            if len(baseline_df) == 0:
                continue
            
            baseline_mean = baseline_df['score'].mean()
            baseline_std = baseline_df['score'].std()
            baseline_count = len(baseline_df)
            
            for so8t in model_groups['so8t']:
                so8t_df = self.df[self.df['model'] == so8t]
                
                if len(so8t_df) == 0:
                    continue
                
                so8t_mean = so8t_df['score'].mean()
                so8t_std = so8t_df['score'].std()
                so8t_count = len(so8t_df)
                
                # 劣化率（パーセンテージ）
                degradation_rate = ((baseline_mean - so8t_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
                
                # 絶対劣化
                absolute_degradation = baseline_mean - so8t_mean
                
                # 統計的有意差検定
                if baseline_count > 1 and so8t_count > 1:
                    try:
                        t_stat, p_value = stats.ttest_ind(baseline_df['score'], so8t_df['score'])
                        significant = p_value < 0.05
                    except:
                        t_stat, p_value = np.nan, np.nan
                        significant = False
                else:
                    t_stat, p_value = np.nan, np.nan
                    significant = False
                
                degradation_data.append({
                    'baseline_model': baseline,
                    'so8t_model': so8t,
                    'baseline_mean': baseline_mean,
                    'baseline_std': baseline_std,
                    'baseline_count': baseline_count,
                    'so8t_mean': so8t_mean,
                    'so8t_std': so8t_std,
                    'so8t_count': so8t_count,
                    'absolute_degradation': absolute_degradation,
                    'degradation_rate_pct': degradation_rate,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'statistically_significant': significant
                })
        
        return pd.DataFrame(degradation_data)
    
    def analyze_category_degradation(self, model_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """カテゴリ別劣化分析"""
        category_data = []
        
        categories = self.df['category'].unique()
        
        for category in categories:
            cat_df = self.df[self.df['category'] == category]
            
            for baseline in model_groups['baseline']:
                baseline_cat_df = cat_df[cat_df['model'] == baseline]
                
                if len(baseline_cat_df) == 0:
                    continue
                
                baseline_mean = baseline_cat_df['score'].mean()
                baseline_count = len(baseline_cat_df)
                
                for so8t in model_groups['so8t']:
                    so8t_cat_df = cat_df[cat_df['model'] == so8t]
                    
                    if len(so8t_cat_df) == 0:
                        continue
                    
                    so8t_mean = so8t_cat_df['score'].mean()
                    so8t_count = len(so8t_cat_df)
                    
                    degradation_rate = ((baseline_mean - so8t_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
                    
                    category_data.append({
                        'category': category,
                        'baseline_model': baseline,
                        'so8t_model': so8t,
                        'baseline_mean': baseline_mean,
                        'so8t_mean': so8t_mean,
                        'degradation_rate_pct': degradation_rate,
                        'baseline_count': baseline_count,
                        'so8t_count': so8t_count
                    })
        
        return pd.DataFrame(category_data)
    
    def create_degradation_visualization(self, degradation_df: pd.DataFrame, category_df: pd.DataFrame):
        """劣化可視化を作成"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SO8T Model Degradation Analysis', fontsize=16, fontweight='bold')
        
        # 1. 全体劣化率比較
        ax1 = axes[0, 0]
        if len(degradation_df) > 0:
            degradation_df_sorted = degradation_df.sort_values('degradation_rate_pct', ascending=False)
            
            x_pos = np.arange(len(degradation_df_sorted))
            colors = ['red' if d > 0 else 'green' for d in degradation_df_sorted['degradation_rate_pct']]
            
            ax1.barh(x_pos, degradation_df_sorted['degradation_rate_pct'], color=colors, alpha=0.7)
            ax1.set_yticks(x_pos)
            ax1.set_yticklabels([f"{row['baseline_model']} vs {row['so8t_model']}" 
                                 for _, row in degradation_df_sorted.iterrows()], fontsize=8)
            ax1.set_xlabel('Degradation Rate (%)', fontsize=12)
            ax1.set_title('Overall Degradation Rate Comparison', fontsize=14, fontweight='bold')
            ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax1.grid(True, alpha=0.3)
        
        # 2. カテゴリ別劣化ヒートマップ
        ax2 = axes[0, 1]
        if len(category_df) > 0:
            pivot_data = category_df.pivot_table(
                values='degradation_rate_pct',
                index='category',
                columns='so8t_model',
                aggfunc='mean'
            )
            
            if len(pivot_data) > 0:
                sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='Reds', 
                           center=0, cbar_kws={'label': 'Degradation Rate (%)'}, 
                           ax=ax2, linewidths=0.5)
                ax2.set_title('Category-wise Degradation Heatmap', fontsize=14, fontweight='bold')
                ax2.set_xlabel('SO8T Model', fontsize=12)
                ax2.set_ylabel('Category', fontsize=12)
        
        # 3. ベースライン vs SO8T スコア比較
        ax3 = axes[1, 0]
        if len(degradation_df) > 0:
            x = degradation_df['baseline_mean']
            y = degradation_df['so8t_mean']
            
            ax3.scatter(x, y, alpha=0.7, s=100)
            
            # 対角線（y=x）を描画
            max_val = max(x.max(), y.max()) if len(x) > 0 else 1.0
            ax3.plot([0, max_val], [0, max_val], 'r--', label='No degradation', linewidth=2)
            
            # モデル名をラベル
            for idx, row in degradation_df.iterrows():
                ax3.annotate(f"{row['so8t_model']}", 
                           (row['baseline_mean'], row['so8t_mean']),
                           fontsize=8, alpha=0.7)
            
            ax3.set_xlabel('Baseline Model Mean Score', fontsize=12)
            ax3.set_ylabel('SO8T Model Mean Score', fontsize=12)
            ax3.set_title('Baseline vs SO8T Score Comparison', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 統計的有意差の表示
        ax4 = axes[1, 1]
        if len(degradation_df) > 0:
            significant_df = degradation_df[degradation_df['statistically_significant'] == True]
            
            if len(significant_df) > 0:
                x_pos = np.arange(len(significant_df))
                colors = ['red' if d > 0 else 'green' for d in significant_df['degradation_rate_pct']]
                
                ax4.barh(x_pos, significant_df['degradation_rate_pct'], color=colors, alpha=0.7)
                ax4.set_yticks(x_pos)
                ax4.set_yticklabels([f"{row['baseline_model']} vs {row['so8t_model']}" 
                                     for _, row in significant_df.iterrows()], fontsize=8)
                ax4.set_xlabel('Degradation Rate (%)', fontsize=12)
                ax4.set_title('Statistically Significant Degradations (p < 0.05)', fontsize=14, fontweight='bold')
                ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No statistically significant\ndegradations found', 
                        ha='center', va='center', fontsize=12, transform=ax4.transAxes)
                ax4.set_title('Statistically Significant Degradations', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'model_degradation_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved degradation visualization to {output_path}")
        
        plt.close()
    
    def generate_degradation_report(self, degradation_df: pd.DataFrame, category_df: pd.DataFrame):
        """劣化レポートを生成"""
        report_path = self.output_dir / 'model_degradation_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# SO8T Model Degradation Analysis Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            if len(degradation_df) > 0:
                avg_degradation = degradation_df['degradation_rate_pct'].mean()
                max_degradation = degradation_df['degradation_rate_pct'].max()
                min_degradation = degradation_df['degradation_rate_pct'].min()
                
                f.write(f"- **Average Degradation Rate**: {avg_degradation:.2f}%\n")
                f.write(f"- **Maximum Degradation Rate**: {max_degradation:.2f}%\n")
                f.write(f"- **Minimum Degradation Rate**: {min_degradation:.2f}%\n")
                f.write(f"- **Total Model Pairs Analyzed**: {len(degradation_df)}\n\n")
                
                significant_count = degradation_df['statistically_significant'].sum()
                f.write(f"- **Statistically Significant Degradations**: {significant_count}/{len(degradation_df)}\n\n")
            
            f.write("## Overall Degradation Metrics\n\n")
            f.write("| Baseline Model | SO8T Model | Baseline Mean | SO8T Mean | ")
            f.write("Absolute Degradation | Degradation Rate (%) | p-value | Significant |\n")
            f.write("|----------------|------------|---------------|-----------|")
            f.write("-------------------|----------------------|---------|-------------|\n")
            
            for _, row in degradation_df.iterrows():
                p_val_str = f"{row['p_value']:.4f}" if not pd.isna(row['p_value']) else "N/A"
                sig_str = "Yes" if row['statistically_significant'] else "No"
                
                f.write(f"| {row['baseline_model']} | {row['so8t_model']} | "
                       f"{row['baseline_mean']:.3f} | {row['so8t_mean']:.3f} | "
                       f"{row['absolute_degradation']:.3f} | {row['degradation_rate_pct']:.2f}% | "
                       f"{p_val_str} | {sig_str} |\n")
            
            f.write("\n## Category-wise Degradation Analysis\n\n")
            
            if len(category_df) > 0:
                category_summary = category_df.groupby('category').agg({
                    'degradation_rate_pct': ['mean', 'max', 'min', 'count']
                }).reset_index()
                
                category_summary.columns = ['Category', 'Mean Degradation (%)', 
                                          'Max Degradation (%)', 'Min Degradation (%)', 'Count']
                
                f.write("| Category | Mean Degradation (%) | Max Degradation (%) | ")
                f.write("Min Degradation (%) | Count |\n")
                f.write("|----------|---------------------|---------------------|")
                f.write("---------------------|-------|\n")
                
                for _, row in category_summary.iterrows():
                    f.write(f"| {row['Category']} | {row['Mean Degradation (%)']:.2f} | "
                           f"{row['Max Degradation (%)']:.2f} | {row['Min Degradation (%)']:.2f} | "
                           f"{int(row['Count'])} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            if len(degradation_df) > 0:
                worst_degradation = degradation_df.loc[degradation_df['degradation_rate_pct'].idxmax()]
                f.write(f"### Worst Degradation\n")
                f.write(f"- **Baseline**: {worst_degradation['baseline_model']}\n")
                f.write(f"- **SO8T**: {worst_degradation['so8t_model']}\n")
                f.write(f"- **Degradation Rate**: {worst_degradation['degradation_rate_pct']:.2f}%\n")
                f.write(f"- **Absolute Degradation**: {worst_degradation['absolute_degradation']:.3f}\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **Investigate Architecture Changes**: Review SO(8) rotation gate implementation\n")
            f.write("2. **Check Training Process**: Verify fine-tuning parameters and regularization\n")
            f.write("3. **Analyze Category-specific Issues**: Focus on categories with highest degradation\n")
            f.write("4. **Compare Model Sizes**: Check if quantization affects performance\n")
            f.write("5. **Review Evaluation Metrics**: Ensure fair comparison between models\n")
        
        print(f"[INFO] Saved degradation report to {report_path}")
    
    def run(self):
        """分析を実行"""
        if not self.load_results():
            print("[ERROR] Failed to load results")
            return
        
        # モデルグループを識別
        model_groups = self.identify_baseline_and_so8t_models()
        
        if len(model_groups['baseline']) == 0 or len(model_groups['so8t']) == 0:
            print("[WARNING] Could not identify baseline or SO8T models")
            return
        
        # 劣化メトリクスを計算
        degradation_df = self.calculate_degradation_metrics(model_groups)
        
        # カテゴリ別劣化分析
        category_df = self.analyze_category_degradation(model_groups)
        
        # 可視化
        self.create_degradation_visualization(degradation_df, category_df)
        
        # レポート生成
        self.generate_degradation_report(degradation_df, category_df)
        
        print(f"[INFO] Degradation analysis complete. Output directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze SO8T model degradation')
    parser.add_argument('--output-dir', default='_docs/benchmark_results/degradation_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    analyzer = ModelDegradationAnalyzer(None, output_dir)
    analyzer.run()


if __name__ == '__main__':
    main()

