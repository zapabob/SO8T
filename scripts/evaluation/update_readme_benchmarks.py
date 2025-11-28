#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
README更新スクリプト
ベンチマーク結果をREADMEに自動挿入
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReadmeUpdater:
    """README更新クラス"""

    def __init__(self, readme_path: Path, results_path: Path, figures_dir: Path):
        self.readme_path = readme_path
        self.results_path = results_path
        self.figures_dir = figures_dir
        
        # 結果読み込み
        with open(results_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.comparison = self.data.get("comparison", {})
        self.statistics = self.data.get("statistics", {})
        self.significance = self.data.get("significance", {})

    def format_percentage(self, value: float) -> str:
        """パーセンテージフォーマット"""
        return f"{value * 100:.2f}%"

    def format_difference(self, diff: float) -> str:
        """差分フォーマット"""
        return f"{diff * 100:+.2f}%"

    def format_significance(self, task_key: str) -> str:
        """有意差フォーマット"""
        sig_data = self.significance.get(task_key, {})
        if sig_data.get("significant", False):
            return "p < 0.05"
        else:
            return "ns"

    def generate_benchmark_table(self) -> str:
        """ベンチマーク結果テーブル生成"""
        if not self.comparison:
            return "No benchmark results available."
        
        lines = [
            "## 📊 Industry Standard Benchmark Results",
            "",
            "### Model Comparison (lm-evaluation-harness)",
            "",
            "| Task | Model A (Borea-Phi3.5) | AEGIS | Difference | Significance |",
            "|------|------------------------|-------|------------|--------------|"
        ]
        
        # タスク順序定義
        task_order = [
            "mmlu", "gsm8k", "arc_challenge", "arc_easy", 
            "hellaswag", "winogrande"
        ]
        
        for task_key in task_order:
            if task_key not in self.comparison:
                continue
            
            comp_data = self.comparison[task_key]
            task_display = task_key.replace('_', ' ').title()
            
            model_a_score = self.format_percentage(comp_data["modelA"])
            aegis_score = self.format_percentage(comp_data["AEGIS"])
            difference = self.format_difference(comp_data["difference"])
            significance = self.format_significance(task_key)
            
            lines.append(
                f"| {task_display} | {model_a_score} | {aegis_score} | {difference} | {significance} |"
            )
        
        return "\n".join(lines)

    def generate_detailed_statistics(self) -> str:
        """詳細統計セクション生成"""
        if not self.statistics:
            return ""
        
        lines = [
            "",
            "### Detailed Statistics",
            ""
        ]
        
        for task_key, stats_data in self.statistics.items():
            task_display = task_key.replace('_', ' ').title()
            model_a_mean = self.format_percentage(stats_data["modelA_mean"])
            aegis_mean = self.format_percentage(stats_data["AEGIS_mean"])
            diff_mean = self.format_difference(stats_data["difference_mean"])
            ci_lower = self.format_difference(stats_data["ci_95_lower"])
            ci_upper = self.format_difference(stats_data["ci_95_upper"])
            
            lines.extend([
                f"#### {task_display}",
                "",
                f"- **Model A Mean**: {model_a_mean}",
                f"- **AEGIS Mean**: {aegis_mean}",
                f"- **Difference**: {diff_mean}",
                f"- **95% CI**: [{ci_lower}, {ci_upper}]",
                ""
            ])
        
        return "\n".join(lines)

    def generate_visualizations_section(self) -> str:
        """可視化セクション生成"""
        figures = {
            "model_comparison_errorbars.png": "Model Comparison with Error Bars",
            "task_breakdown_comparison.png": "Task Breakdown Comparison",
            "statistical_significance_heatmap.png": "Statistical Significance Heatmap",
            "agi_tests_breakdown.png": "AGI Tests Breakdown",
            "elyza_100_comparison.png": "ELYZA-100 Comparison"
        }
        
        lines = [
            "",
            "### Visualizations",
            ""
        ]
        
        for filename, description in figures.items():
            figure_path = self.figures_dir / filename
            if figure_path.exists():
                # 相対パス計算（READMEからの相対パス）
                rel_path = figure_path.relative_to(self.readme_path.parent)
                lines.append(f"- **{description}**: ![{description}]({rel_path})")
        
        if len(lines) == 3:  # ヘッダーのみ
            lines.append("No visualizations available.")
        
        return "\n".join(lines)

    def generate_benchmark_section(self) -> str:
        """完全なベンチマークセクション生成"""
        sections = [
            self.generate_benchmark_table(),
            self.generate_detailed_statistics(),
            self.generate_visualizations_section()
        ]
        
        return "\n".join(sections)

    def update_readme(self, insert_after: str = "## 🔬 Benchmark Method"):
        """READMEを更新"""
        if not self.readme_path.exists():
            logger.error(f"README not found: {self.readme_path}")
            return False
        
        # README読み込み
        with open(self.readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 既存のベンチマーク結果セクションを検索・削除
        pattern = r"## 📊 Industry Standard Benchmark Results.*?(?=## |\Z)"
        content = re.sub(pattern, "", content, flags=re.DOTALL)
        
        # 挿入位置を検索
        insert_pattern = re.escape(insert_after)
        match = re.search(insert_pattern, content)
        
        if not match:
            logger.warning(f"Insert marker '{insert_after}' not found, appending to end")
            new_section = self.generate_benchmark_section()
            content = content.rstrip() + "\n\n" + new_section + "\n"
        else:
            # マーカーの後に挿入
            insert_pos = match.end()
            new_section = self.generate_benchmark_section()
            content = content[:insert_pos] + "\n\n" + new_section + "\n" + content[insert_pos:]
        
        # バックアップ作成
        backup_path = self.readme_path.with_suffix('.md.bak')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # README更新
        with open(self.readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"README updated: {self.readme_path}")
        logger.info(f"Backup saved: {backup_path}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Update README with Benchmark Results"
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="Path to README.md"
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to benchmark results JSON file"
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("D:/webdataset/benchmark_results/industry_standard/figures"),
        help="Directory containing figure images"
    )
    parser.add_argument(
        "--insert-after",
        type=str,
        default="## 🔬 Benchmark Method",
        help="Insert benchmark section after this marker"
    )
    
    args = parser.parse_args()
    
    if not args.results.exists():
        logger.error(f"Results file not found: {args.results}")
        sys.exit(1)
    
    updater = ReadmeUpdater(args.readme, args.results, args.figures_dir)
    success = updater.update_readme(args.insert_after)
    
    if success:
        logger.info("README update completed!")
        print(f"\nREADME updated: {args.readme}")
    else:
        logger.error("README update failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
























