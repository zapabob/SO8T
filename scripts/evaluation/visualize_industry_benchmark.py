#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
業界標準ベンチマーク結果可視化スクリプト
エラーバー付きグラフと統計的可視化を生成
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 日本語フォント設定（Windows）
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

RESULTS_DIR = Path("D:/webdataset/benchmark_results/industry_standard")
FIGURES_DIR = RESULTS_DIR / "figures"


class BenchmarkVisualizer:
    """ベンチマーク結果可視化クラス"""

    def __init__(self, results_path: Path, output_dir: Path = FIGURES_DIR):
        self.results_path = results_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 結果読み込み
        with open(results_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.comparison = self.data.get("comparison", {})
        self.statistics = self.data.get("statistics", {})
        self.significance = self.data.get("significance", {})

    def create_model_comparison_errorbars(self):
        """エラーバー付きモデル比較グラフ"""
        if not self.comparison:
            logger.warning("No comparison data available")
            return None
        
        tasks = list(self.comparison.keys())
        model_a_scores = [self.comparison[t]["modelA"] * 100 for t in tasks]
        aegis_scores = [self.comparison[t]["AEGIS"] * 100 for t in tasks]
        
        # エラーバー（95%信頼区間）
        model_a_errors = []
        aegis_errors = []
        
        for task in tasks:
            stats_data = self.statistics.get(task, {})
            ci_95 = stats_data.get("ci_95_upper", 0.0) - stats_data.get("ci_95_lower", 0.0)
            model_a_errors.append(ci_95 * 50)  # 簡易版
            aegis_errors.append(ci_95 * 50)
        
        x = np.arange(len(tasks))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, model_a_scores, width, label='Model A (Borea-Phi3.5)', 
                      yerr=model_a_errors, capsize=5, alpha=0.8)
        bars2 = ax.bar(x + width/2, aegis_scores, width, label='AEGIS',
                      yerr=aegis_errors, capsize=5, alpha=0.8)
        
        ax.set_xlabel('Benchmark Task', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Model Comparison with Error Bars (95% CI)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in tasks], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "model_comparison_errorbars.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_path}")
        return output_path

    def create_task_breakdown_comparison(self):
        """タスク別詳細比較グラフ"""
        if not self.comparison:
            logger.warning("No comparison data available")
            return None
        
        tasks = list(self.comparison.keys())
        differences = [self.comparison[t]["difference"] * 100 for t in tasks]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if d > 0 else 'red' for d in differences]
        bars = ax.barh(range(len(tasks)), differences, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([t.replace('_', ' ').title() for t in tasks])
        ax.set_xlabel('Score Difference (%) [AEGIS - Model A]', fontsize=12)
        ax.set_title('Task Breakdown Comparison', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 値のラベル追加
        for i, (bar, diff) in enumerate(zip(bars, differences)):
            ax.text(diff + (0.5 if diff > 0 else -0.5), i, f'{diff:+.2f}%',
                   va='center', ha='left' if diff > 0 else 'right', fontsize=10)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "task_breakdown_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_path}")
        return output_path

    def create_statistical_significance_heatmap(self):
        """統計的有意差検定結果ヒートマップ"""
        if not self.significance:
            logger.warning("No significance data available")
            return None
        
        tasks = list(self.significance.keys())
        differences = [self.significance[t]["difference"] * 100 for t in tasks]
        significant = [self.significance[t]["significant"] for t in tasks]
        
        # データフレーム作成
        df = pd.DataFrame({
            'Task': [t.replace('_', ' ').title() for t in tasks],
            'Difference (%)': differences,
            'Significant': significant
        })
        
        # ヒートマップ用データ準備
        heatmap_data = np.array([[d] for d in differences])
        
        fig, ax = plt.subplots(figsize=(8, len(tasks) * 0.6))
        
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Difference (%)'},
            yticklabels=df['Task'],
            xticklabels=['AEGIS - Model A'],
            ax=ax
        )
        
        ax.set_title('Statistical Significance Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "statistical_significance_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_path}")
        return output_path

    def create_elyza_100_comparison(self):
        """ELYZA-100専用比較グラフ（データがある場合）"""
        # ELYZA-100の結果があるか確認
        elyza_results = [r for r in self.data.get("results", []) if "elyza" in r.get("task", "").lower()]
        
        if not elyza_results:
            logger.info("No ELYZA-100 results found, skipping ELYZA-100 comparison")
            return None
        
        # ELYZA-100の可視化実装（データ構造に応じて調整）
        logger.info("ELYZA-100 comparison graph creation (placeholder)")
        return None

    def create_agi_tests_breakdown(self):
        """AGIテストカテゴリ別比較グラフ"""
        agi_tasks = [t for t in self.comparison.keys() if any(x in t.lower() for x in ['arc', 'hellaswag', 'winogrande'])]
        
        if not agi_tasks:
            logger.warning("No AGI test results available")
            return None
        
        tasks = agi_tasks
        model_a_scores = [self.comparison[t]["modelA"] * 100 for t in tasks]
        aegis_scores = [self.comparison[t]["AEGIS"] * 100 for t in tasks]
        
        x = np.arange(len(tasks))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, model_a_scores, width, label='Model A (Borea-Phi3.5)', alpha=0.8)
        bars2 = ax.bar(x + width/2, aegis_scores, width, label='AEGIS', alpha=0.8)
        
        ax.set_xlabel('AGI Test Category', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('AGI Tests Breakdown Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in tasks], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "agi_tests_breakdown.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_path}")
        return output_path

    def create_all_visualizations(self):
        """全可視化グラフを生成"""
        logger.info("Creating all visualizations...")
        
        figures = {}
        
        try:
            figures["comparison_errorbars"] = self.create_model_comparison_errorbars()
        except Exception as e:
            logger.error(f"Failed to create comparison errorbars: {e}")
        
        try:
            figures["task_breakdown"] = self.create_task_breakdown_comparison()
        except Exception as e:
            logger.error(f"Failed to create task breakdown: {e}")
        
        try:
            figures["significance_heatmap"] = self.create_statistical_significance_heatmap()
        except Exception as e:
            logger.error(f"Failed to create significance heatmap: {e}")
        
        try:
            figures["agi_breakdown"] = self.create_agi_tests_breakdown()
        except Exception as e:
            logger.error(f"Failed to create AGI breakdown: {e}")
        
        try:
            figures["elyza_100"] = self.create_elyza_100_comparison()
        except Exception as e:
            logger.error(f"Failed to create ELYZA-100 comparison: {e}")
        
        logger.info(f"Created {len([f for f in figures.values() if f])} visualizations")
        return figures


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Industry Standard Benchmark Results"
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to benchmark results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Output directory for figures"
    )
    
    args = parser.parse_args()
    
    if not args.results.exists():
        logger.error(f"Results file not found: {args.results}")
        sys.exit(1)
    
    visualizer = BenchmarkVisualizer(args.results, args.output_dir)
    figures = visualizer.create_all_visualizations()
    
    logger.info("Visualization completed!")
    print(f"\nFigures saved to: {args.output_dir}")
    for name, path in figures.items():
        if path:
            print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()

































