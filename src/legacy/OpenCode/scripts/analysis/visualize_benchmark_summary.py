#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマークテスト結果の統合可視化スクリプト
エラーバー付きグラフと要約統計量を生成
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # バックエンドを設定
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

class BenchmarkVisualizer:
    """ベンチマーク結果の可視化クラス"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_results = []
        
    def load_results_from_file(self, file_path: Path) -> List[Dict]:
        """結果ファイルを読み込む"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = []
            
            # 形式1: metadata + results 構造
            if isinstance(data, dict) and 'results' in data:
                for item in data['results']:
                    if 'model' in item and 'score' in item:
                        results.append({
                            'model': item['model'],
                            'category': item.get('category', 'unknown'),
                            'test_name': item.get('test_name', 'unknown'),
                            'score': float(item.get('score', 0.0)),
                            'response_time': float(item.get('response_time', 0.0)),
                            'source_file': str(file_path),
                            'timestamp': item.get('timestamp', '')
                        })
            
            # 形式2: タスク別結果配列（benchmarks/results/*.json）
            elif isinstance(data, list):
                # ファイル名からモデル名を抽出
                model_name = file_path.stem.replace('_results', '').replace('_q8_0', '').replace('_q4_k_m', '')
                
                for item in data:
                    if 'is_correct' in item:
                        score = 1.0 if item['is_correct'] else 0.0
                        task = item.get('task', 'unknown')
                        
                        results.append({
                            'model': model_name,
                            'category': task,
                            'test_name': task,
                            'score': score,
                            'response_time': float(item.get('wall_time', 0.0)),
                            'tokens_per_sec': float(item.get('tokens_per_sec', 0.0)),
                            'source_file': str(file_path),
                            'timestamp': ''
                        })
            
            return results
            
        except Exception as e:
            print(f"[WARNING] Failed to load {file_path}: {e}")
            return []
    
    def load_all_results(self, input_dirs: List[Path]):
        """すべての結果ファイルを読み込む"""
        self.all_results = []
        
        for input_dir in input_dirs:
            input_path = Path(input_dir)
            
            if not input_path.exists():
                print(f"[WARNING] Directory not found: {input_path}")
                continue
            
            # JSONファイルを検索
            json_files = list(input_path.rglob('*.json'))
            
            for json_file in json_files:
                # 結果ファイルのみを読み込む（設定ファイルは除外）
                if 'config' in json_file.name.lower() or 'model' in json_file.name.lower() and 'config' in json_file.name.lower():
                    continue
                
                results = self.load_results_from_file(json_file)
                self.all_results.extend(results)
                print(f"[INFO] Loaded {len(results)} results from {json_file}")
        
        print(f"[INFO] Total results loaded: {len(self.all_results)}")
        return len(self.all_results) > 0
    
    def calculate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """要約統計量を計算"""
        stats_dict = {}
        
        # モデル別統計
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            
            stats_dict[f'{model}_mean'] = model_df['score'].mean()
            stats_dict[f'{model}_std'] = model_df['score'].std()
            stats_dict[f'{model}_median'] = model_df['score'].median()
            stats_dict[f'{model}_min'] = model_df['score'].min()
            stats_dict[f'{model}_max'] = model_df['score'].max()
            stats_dict[f'{model}_count'] = len(model_df)
            
            # 95%信頼区間
            if len(model_df) > 1:
                sem = stats.sem(model_df['score'])
                ci = stats.t.interval(0.95, len(model_df) - 1, loc=model_df['score'].mean(), scale=sem)
                stats_dict[f'{model}_ci_lower'] = ci[0]
                stats_dict[f'{model}_ci_upper'] = ci[1]
            else:
                stats_dict[f'{model}_ci_lower'] = model_df['score'].mean()
                stats_dict[f'{model}_ci_upper'] = model_df['score'].mean()
        
        # カテゴリ別統計
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            
            for model in cat_df['model'].unique():
                model_cat_df = cat_df[cat_df['model'] == model]
                
                if len(model_cat_df) > 0:
                    stats_dict[f'{model}_{category}_mean'] = model_cat_df['score'].mean()
                    stats_dict[f'{model}_{category}_std'] = model_cat_df['score'].std()
                    stats_dict[f'{model}_{category}_count'] = len(model_cat_df)
        
        return stats_dict
    
    def create_errorbar_chart(self, df: pd.DataFrame, stats_dict: Dict):
        """エラーバー付きグラフを作成"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Benchmark Test Results Summary with Error Bars', fontsize=16, fontweight='bold')
        
        # 1. モデル別全体比較（エラーバー付き）
        ax1 = axes[0, 0]
        models = df['model'].unique()
        means = []
        stds = []
        ci_lowers = []
        ci_uppers = []
        
        for model in models:
            model_df = df[df['model'] == model]
            mean = model_df['score'].mean()
            std = model_df['score'].std()
            
            means.append(mean)
            stds.append(std)
            
            # 95%信頼区間
            if len(model_df) > 1:
                sem = stats.sem(model_df['score'])
                ci = stats.t.interval(0.95, len(model_df) - 1, loc=mean, scale=sem)
                ci_lowers.append(ci[0])
                ci_uppers.append(ci[1])
            else:
                ci_lowers.append(mean)
                ci_uppers.append(mean)
        
        x_pos = np.arange(len(models))
        ax1.bar(x_pos, means, yerr=[np.array(means) - np.array(ci_lowers), 
                                      np.array(ci_uppers) - np.array(means)],
                capsize=5, alpha=0.7, color=sns.color_palette("husl", len(models)))
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Mean Score', fontsize=12)
        ax1.set_title('Overall Performance Comparison (95% CI)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, max(ci_uppers) * 1.2 if ci_uppers else 1.0])
        
        # 統計量をテキストで表示
        for i, model in enumerate(models):
            mean_val = means[i]
            std_val = stds[i]
            count_val = len(df[df['model'] == model])
            ax1.text(i, mean_val + (ci_uppers[i] - means[i]) * 1.1, 
                    f'n={count_val}\nμ={mean_val:.3f}\nσ={std_val:.3f}',
                    ha='center', va='bottom', fontsize=8)
        
        # 2. カテゴリ別比較（エラーバー付き）
        ax2 = axes[0, 1]
        categories = df['category'].unique()
        category_means = {model: [] for model in models}
        category_stds = {model: [] for model in models}
        category_cis = {model: {'lower': [], 'upper': []} for model in models}
        
        for category in categories:
            for model in models:
                cat_model_df = df[(df['category'] == category) & (df['model'] == model)]
                if len(cat_model_df) > 0:
                    mean = cat_model_df['score'].mean()
                    std = cat_model_df['score'].std()
                    category_means[model].append(mean)
                    category_stds[model].append(std)
                    
                    if len(cat_model_df) > 1:
                        sem = stats.sem(cat_model_df['score'])
                        ci = stats.t.interval(0.95, len(cat_model_df) - 1, loc=mean, scale=sem)
                        category_cis[model]['lower'].append(ci[0])
                        category_cis[model]['upper'].append(ci[1])
                    else:
                        category_cis[model]['lower'].append(mean)
                        category_cis[model]['upper'].append(mean)
                else:
                    category_means[model].append(0)
                    category_stds[model].append(0)
                    category_cis[model]['lower'].append(0)
                    category_cis[model]['upper'].append(0)
        
        x_pos_cat = np.arange(len(categories))
        width = 0.35
        
        for i, model in enumerate(models):
            means_cat = category_means[model]
            ci_lowers_cat = category_cis[model]['lower']
            ci_uppers_cat = category_cis[model]['upper']
            
            offset = (i - len(models) / 2 + 0.5) * width / len(models)
            ax2.bar(x_pos_cat + offset, means_cat,
                   yerr=[np.array(means_cat) - np.array(ci_lowers_cat),
                         np.array(ci_uppers_cat) - np.array(means_cat)],
                   width=width/len(models), label=model, alpha=0.7, capsize=3)
        
        ax2.set_xlabel('Category', fontsize=12)
        ax2.set_ylabel('Mean Score', fontsize=12)
        ax2.set_title('Category-wise Performance Comparison (95% CI)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos_cat)
        ax2.set_xticklabels(categories, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 箱ひげ図（分布可視化）
        ax3 = axes[1, 0]
        data_for_box = [df[df['model'] == model]['score'].values for model in models]
        bp = ax3.boxplot(data_for_box, labels=models, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(models))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_xlabel('Model', fontsize=12)
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 4. 要約統計量テーブル
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # 統計量テーブルを作成
        table_data = []
        table_data.append(['Model', 'Mean', 'Std', 'Median', 'Min', 'Max', 'Count', '95% CI'])
        
        for model in models:
            model_df = df[df['model'] == model]
            mean = model_df['score'].mean()
            std = model_df['score'].std()
            median = model_df['score'].median()
            min_val = model_df['score'].min()
            max_val = model_df['score'].max()
            count = len(model_df)
            
            if len(model_df) > 1:
                sem = stats.sem(model_df['score'])
                ci = stats.t.interval(0.95, len(model_df) - 1, loc=mean, scale=sem)
                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            else:
                ci_str = f"[{mean:.3f}, {mean:.3f}]"
            
            table_data.append([
                model,
                f"{mean:.3f}",
                f"{std:.3f}",
                f"{median:.3f}",
                f"{min_val:.3f}",
                f"{max_val:.3f}",
                str(count),
                ci_str
            ])
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.1, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # ヘッダー行を強調
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Summary Statistics Table', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # 保存
        output_path = self.output_dir / 'benchmark_summary_with_errorbars.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved errorbar chart to {output_path}")
        
        plt.close()
    
    def create_category_heatmap(self, df: pd.DataFrame):
        """カテゴリ別ヒートマップを作成"""
        # カテゴリ×モデルの平均スコアを計算
        pivot_data = df.pivot_table(values='score', index='category', columns='model', aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Mean Score'}, ax=ax, linewidths=0.5)
        
        ax.set_title('Category-wise Performance Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Category', fontsize=12)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'category_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved heatmap to {output_path}")
        
        plt.close()
    
    def generate_summary_report(self, df: pd.DataFrame, stats_dict: Dict):
        """要約レポートを生成"""
        report_path = self.output_dir / 'benchmark_summary_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Benchmark Test Results Summary Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Results**: {len(df)}\n\n")
            
            f.write("## Summary Statistics\n\n")
            f.write("| Model | Mean | Std | Median | Min | Max | Count | 95% CI |\n")
            f.write("|-------|------|-----|--------|-----|-----|-------|--------|\n")
            
            models = df['model'].unique()
            for model in models:
                model_df = df[df['model'] == model]
                mean = model_df['score'].mean()
                std = model_df['score'].std()
                median = model_df['score'].median()
                min_val = model_df['score'].min()
                max_val = model_df['score'].max()
                count = len(model_df)
                
                if len(model_df) > 1:
                    sem = stats.sem(model_df['score'])
                    ci = stats.t.interval(0.95, len(model_df) - 1, loc=mean, scale=sem)
                    ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                else:
                    ci_str = f"[{mean:.3f}, {mean:.3f}]"
                
                f.write(f"| {model} | {mean:.3f} | {std:.3f} | {median:.3f} | "
                       f"{min_val:.3f} | {max_val:.3f} | {count} | {ci_str} |\n")
            
            f.write("\n## Category-wise Statistics\n\n")
            
            categories = sorted(df['category'].unique())
            for category in categories:
                f.write(f"### {category}\n\n")
                f.write("| Model | Mean | Std | Count |\n")
                f.write("|-------|------|-----|-------|\n")
                
                for model in models:
                    cat_model_df = df[(df['category'] == category) & (df['model'] == model)]
                    if len(cat_model_df) > 0:
                        mean = cat_model_df['score'].mean()
                        std = cat_model_df['score'].std()
                        count = len(cat_model_df)
                        f.write(f"| {model} | {mean:.3f} | {std:.3f} | {count} |\n")
                
                f.write("\n")
            
            f.write("## Visualizations\n\n")
            f.write("- `benchmark_summary_with_errorbars.png`: Error bar charts and summary statistics\n")
            f.write("- `category_heatmap.png`: Category-wise performance heatmap\n")
        
        print(f"[INFO] Saved summary report to {report_path}")
    
    def run(self, input_dirs: List[Path]):
        """可視化を実行"""
        # 結果を読み込む
        if not self.load_all_results(input_dirs):
            print("[ERROR] No results loaded")
            return
        
        # DataFrameに変換
        df = pd.DataFrame(self.all_results)
        
        if len(df) == 0:
            print("[ERROR] No data to visualize")
            return
        
        print(f"[INFO] Processing {len(df)} results")
        print(f"[INFO] Models: {df['model'].unique()}")
        print(f"[INFO] Categories: {df['category'].unique()}")
        
        # 統計量を計算
        stats_dict = self.calculate_summary_statistics(df)
        
        # グラフを作成
        self.create_errorbar_chart(df, stats_dict)
        self.create_category_heatmap(df)
        
        # レポートを生成
        self.generate_summary_report(df, stats_dict)
        
        print(f"[INFO] Visualization complete. Output directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark test results')
    parser.add_argument('--input-dirs', nargs='+', required=True,
                       help='Input directories containing benchmark results')
    parser.add_argument('--output-dir', default='_docs/benchmark_results/visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    input_dirs = [Path(d) for d in args.input_dirs]
    output_dir = Path(args.output_dir)
    
    visualizer = BenchmarkVisualizer(output_dir)
    visualizer.run(input_dirs)


if __name__ == '__main__':
    main()







