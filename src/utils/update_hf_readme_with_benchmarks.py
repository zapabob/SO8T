#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace README更新スクリプト
統計的分析結果とグラフ画像をREADMEに埋め込む
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFReadmeUpdater:
    """HuggingFace README更新クラス"""

    def __init__(self, repo_id: str, results_file: str, 
                 statistical_analysis_file: str, visualization_dir: str):
        """
        初期化
        
        Args:
            repo_id: HuggingFaceリポジトリID
            results_file: ABCテスト結果JSONファイルのパス
            statistical_analysis_file: 統計的分析結果JSONファイルのパス
            visualization_dir: 可視化グラフのディレクトリ
        """
        self.repo_id = repo_id
        self.api = HfApi()
        self.project_root = Path(__file__).parent.parent.parent
        
        # ファイル読み込み
        with open(results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        with open(statistical_analysis_file, 'r', encoding='utf-8') as f:
            self.statistical_analysis = json.load(f)
        
        self.visualization_dir = Path(visualization_dir)
        
        # モデル名マッピング
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

    def generate_benchmark_table(self) -> str:
        """ベンチマーク結果テーブルを生成（エラーバー付き）"""
        table_lines = [
            "## [STATS] Benchmark Results / ベンチマーク結果",
            "",
            "### Industry Standard Benchmarks / 業界標準ベンチマーク",
            "",
            "| Benchmark | Qwen2.5-7B (Base) | SO8T-trained | AEGIS-Phi3.5 | Improvement |",
            "|-----------|-------------------|--------------|--------------|-------------|"
        ]
        
        # 業界標準ベンチマーク
        industry_benchmarks = ['mmlu', 'bbh', 'commonsenseqa', 'openbookqa', 
                              'socialiqa', 'piqa', 'winogrande', 'boolq']
        
        for benchmark in industry_benchmarks:
            if benchmark in self.statistical_analysis.get('summary_statistics', {}):
                stats = self.statistical_analysis['summary_statistics'][benchmark]
                
                # 各モデルのスコア（エラーバー付き）
                scores = {}
                for model in ['A', 'B', 'C']:
                    if model in stats:
                        mean = stats[model]['mean'] * 100
                        ci_lower = stats[model]['ci_95_lower'] * 100
                        ci_upper = stats[model]['ci_95_upper'] * 100
                        scores[model] = f"{mean:.1f} [{ci_lower:.1f}, {ci_upper:.1f}]"
                    else:
                        scores[model] = "N/A"
                
                # 改善度（A vs C）
                improvement = ""
                if 'A' in stats and 'C' in stats:
                    mean_a = stats['A']['mean'] * 100
                    mean_c = stats['C']['mean'] * 100
                    diff = mean_c - mean_a
                    improvement = f"+{diff:.1f}pp" if diff > 0 else f"{diff:.1f}pp"
                    
                    # 統計的有意性マーカー
                    if benchmark in self.statistical_analysis.get('pairwise_comparisons', {}):
                        if 'A_vs_C' in self.statistical_analysis['pairwise_comparisons'][benchmark]:
                            p_value = self.statistical_analysis['pairwise_comparisons'][benchmark]['A_vs_C'].get('p_value', 1.0)
                            if p_value < 0.001:
                                improvement += " ***"
                            elif p_value < 0.01:
                                improvement += " **"
                            elif p_value < 0.05:
                                improvement += " *"
                
                benchmark_name = self.benchmark_names.get(benchmark, benchmark)
                table_lines.append(
                    f"| {benchmark_name} | {scores.get('A', 'N/A')} | "
                    f"{scores.get('B', 'N/A')} | {scores.get('C', 'N/A')} | {improvement} |"
                )
        
        table_lines.extend([
            "",
            "### Advanced Benchmarks / 高度ベンチマーク",
            "",
            "| Benchmark | Qwen2.5-7B (Base) | SO8T-trained | AEGIS-Phi3.5 | Improvement |",
            "|-----------|-------------------|--------------|--------------|-------------|"
        ])
        
        # 高度ベンチマーク
        advanced_benchmarks = ['drop', 'strategyqa']
        
        for benchmark in advanced_benchmarks:
            if benchmark in self.statistical_analysis.get('summary_statistics', {}):
                stats = self.statistical_analysis['summary_statistics'][benchmark]
                
                scores = {}
                for model in ['A', 'B', 'C']:
                    if model in stats:
                        mean = stats[model]['mean'] * 100
                        ci_lower = stats[model]['ci_95_lower'] * 100
                        ci_upper = stats[model]['ci_95_upper'] * 100
                        scores[model] = f"{mean:.1f} [{ci_lower:.1f}, {ci_upper:.1f}]"
                    else:
                        scores[model] = "N/A"
                
                improvement = ""
                if 'A' in stats and 'C' in stats:
                    mean_a = stats['A']['mean'] * 100
                    mean_c = stats['C']['mean'] * 100
                    diff = mean_c - mean_a
                    improvement = f"+{diff:.1f}pp" if diff > 0 else f"{diff:.1f}pp"
                    
                    if benchmark in self.statistical_analysis.get('pairwise_comparisons', {}):
                        if 'A_vs_C' in self.statistical_analysis['pairwise_comparisons'][benchmark]:
                            p_value = self.statistical_analysis['pairwise_comparisons'][benchmark]['A_vs_C'].get('p_value', 1.0)
                            if p_value < 0.001:
                                improvement += " ***"
                            elif p_value < 0.01:
                                improvement += " **"
                            elif p_value < 0.05:
                                improvement += " *"
                
                benchmark_name = self.benchmark_names.get(benchmark, benchmark)
                table_lines.append(
                    f"| {benchmark_name} | {scores.get('A', 'N/A')} | "
                    f"{scores.get('B', 'N/A')} | {scores.get('C', 'N/A')} | {improvement} |"
                )
        
        table_lines.extend([
            "",
            "### Japanese Benchmarks / 日本語ベンチマーク",
            "",
            "| Benchmark | Qwen2.5-7B (Base) | SO8T-trained | AEGIS-Phi3.5 | Improvement |",
            "|-----------|-------------------|--------------|--------------|-------------|"
        ])
        
        # 日本語ベンチマーク
        japanese_benchmarks = ['elyza_tasks_100']
        
        for benchmark in japanese_benchmarks:
            if benchmark in self.statistical_analysis.get('summary_statistics', {}):
                stats = self.statistical_analysis['summary_statistics'][benchmark]
                
                scores = {}
                for model in ['A', 'B', 'C']:
                    if model in stats:
                        mean = stats[model]['mean'] * 100
                        ci_lower = stats[model]['ci_95_lower'] * 100
                        ci_upper = stats[model]['ci_95_upper'] * 100
                        scores[model] = f"{mean:.1f} [{ci_lower:.1f}, {ci_upper:.1f}]"
                    else:
                        scores[model] = "N/A"
                
                improvement = ""
                if 'A' in stats and 'C' in stats:
                    mean_a = stats['A']['mean'] * 100
                    mean_c = stats['C']['mean'] * 100
                    diff = mean_c - mean_a
                    improvement = f"+{diff:.1f}pp" if diff > 0 else f"{diff:.1f}pp"
                    
                    if benchmark in self.statistical_analysis.get('pairwise_comparisons', {}):
                        if 'A_vs_C' in self.statistical_analysis['pairwise_comparisons'][benchmark]:
                            p_value = self.statistical_analysis['pairwise_comparisons'][benchmark]['A_vs_C'].get('p_value', 1.0)
                            if p_value < 0.001:
                                improvement += " ***"
                            elif p_value < 0.01:
                                improvement += " **"
                            elif p_value < 0.05:
                                improvement += " *"
                
                benchmark_name = self.benchmark_names.get(benchmark, benchmark)
                table_lines.append(
                    f"| {benchmark_name} | {scores.get('A', 'N/A')} | "
                    f"{scores.get('B', 'N/A')} | {scores.get('C', 'N/A')} | {improvement} |"
                )
        
        table_lines.extend([
            "",
            "**Note**: Scores shown as Mean [95% CI Lower, 95% CI Upper]. "
            "Improvement shows AEGIS-Phi3.5 vs Qwen2.5-7B (Base). "
            "Statistical significance: *** p<0.001, ** p<0.01, * p<0.05 (Bonferroni corrected).",
            "",
            "**注意**: スコアは平均値 [95%信頼区間下限, 95%信頼区間上限] で表示。"
            "改善度はAEGIS-Phi3.5 vs Qwen2.5-7B (Base)を示す。"
            "統計的有意性: *** p<0.001, ** p<0.01, * p<0.05（Bonferroni補正済み）。"
        ])
        
        return "\n".join(table_lines)

    def generate_visualization_section(self) -> str:
        """可視化セクションを生成"""
        chart_files = [
            ('abc_individual_benchmark_comparison.png', 
             'Individual Benchmark Comparison', 
             '個別ベンチマーク比較',
             'Error bars show 95% confidence intervals across 10 random seeds. Higher bars indicate better performance.'),
            ('abc_comprehensive_benchmark_overview.png',
             'Comprehensive Benchmark Overview',
             '包括的ベンチマーク概要',
             'Comprehensive view of all models across all benchmarks with 95% CI error bars.'),
            ('abc_statistical_significance.png',
             'Statistical Significance',
             '統計的有意性',
             'Performance improvements with statistical significance. Red bars indicate p<0.001, pink p<0.01, orange p<0.05 (Bonferroni corrected).'),
            ('abc_industry_comparison.png',
             'Industry Standard Comparison',
             '業界標準比較',
             'AEGIS-Phi3.5 performance compared to industry leaders (Llama-3-8B, Qwen2.5-7B).'),
            ('abc_ranking_heatmap.png',
             'Model Ranking Heatmap',
             'モデルランキングヒートマップ',
             'Ranking visualization (1=Best, 3=Worst) across benchmarks. Darker green indicates better ranking.'),
            ('abc_industry_standard_benchmarks.png',
             'Industry Standard Benchmarks (MMLU with 5-shot protocol)',
             '業界標準ベンチマーク（MMLU 5-shotプロトコル）',
             'Industry standard benchmarks (MMLU, BBH, CommonsenseQA, OpenBookQA, SocialIQA, PIQA, Winogrande, BoolQ) evaluated with industry-standard measurement protocols. MMLU uses 5-shot few-shot evaluation.'),
            ('abc_advanced_benchmarks.png',
             'Advanced Benchmarks',
             '高度ベンチマーク',
             'Advanced reasoning benchmarks (DROP, StrategyQA) with 95% CI error bars across 10 random seeds.'),
            ('abc_elyza100_benchmark.png',
             'ELIZA-100 (Japanese Language Evaluation)',
             'ELIZA-100（日本語評価）',
             'ELIZA-100 comprehensive Japanese language understanding and reasoning evaluation with 95% CI error bars across 10 random seeds.')
        ]
        
        section_lines = [
            "## [STATS] ABC Test Visualizations / ABCテスト可視化",
            "",
            "### Performance Comparison Charts / 性能比較チャート",
            ""
        ]
        
        for i, (chart_file, title_en, title_jp, description) in enumerate(chart_files, 1):
            section_lines.extend([
                f"#### {i}. {title_en} / {title_jp}",
                f"![{title_en}](abc_test_charts/{chart_file})",
                "",
                f"**Description**: {description}",
                "",
                f"**説明**: {description}",
                ""
            ])
        
        section_lines.extend([
            "### Benchmark Categories / ベンチマークカテゴリ",
            "",
            "#### Industry Standard Benchmarks / 業界標準ベンチマーク",
            "",
            "![Industry Standard Benchmarks](abc_test_charts/abc_industry_standard_benchmarks.png)",
            "",
            "**Description**: Industry standard benchmarks (MMLU with 5-shot protocol, BBH, CommonsenseQA, OpenBookQA, SocialIQA, PIQA, Winogrande, BoolQ) evaluated using industry-standard measurement protocols.",
            "",
            "**説明**: 業界標準ベンチマーク（MMLU 5-shotプロトコル、BBH、CommonsenseQA、OpenBookQA、SocialIQA、PIQA、Winogrande、BoolQ）を業界標準測定手法で評価。",
            "",
            "#### Advanced Benchmarks / 高度ベンチマーク",
            "",
            "![Advanced Benchmarks](abc_test_charts/abc_advanced_benchmarks.png)",
            "",
            "**Description**: Advanced reasoning benchmarks (DROP, StrategyQA) with 95% CI error bars across 10 random seeds.",
            "",
            "**説明**: 高度推論ベンチマーク（DROP、StrategyQA）を10ランダムシードで評価、95%信頼区間エラーバー付き。",
            "",
            "#### ELIZA-100 (Japanese Language Evaluation) / ELIZA-100（日本語評価）",
            "",
            "![ELIZA-100 Benchmark](abc_test_charts/abc_elyza100_benchmark.png)",
            "",
            "**Description**: ELIZA-100 comprehensive Japanese language understanding and reasoning evaluation with 95% CI error bars across 10 random seeds.",
            "",
            "**説明**: ELIZA-100包括的日本語理解・推論評価を10ランダムシードで評価、95%信頼区間エラーバー付き。",
            "",
            "### Statistical Methodology / 統計的手法",
            "",
            "- **Multiple Comparison Correction**: Bonferroni correction (α = 0.05 / N benchmarks)",
            "- **Error Bars**: 95% confidence intervals using t-distribution",
            "- **Random Seeds**: 10 random seeds for statistical robustness",
            "- **Significance Testing**: t-test or Mann-Whitney U test based on normality",
            "- **Effect Size**: Cohen's d for practical significance",
            "- **MMLU Protocol**: 5-shot few-shot evaluation (industry standard)",
            "",
            "- **多重比較補正**: Bonferroni補正（α = 0.05 / Nベンチマーク）",
            "- **エラーバー**: t分布を使用した95%信頼区間",
            "- **ランダムシード**: 統計的堅牢性のための10ランダムシード",
            "- **有意性検定**: 正規性に基づくt検定またはMann-Whitney U検定",
            "- **効果量**: 実用的意義のためのCohen's d",
            "- **MMLUプロトコル**: 5-shot few-shot評価（業界標準）",
            ""
        ])
        
        return "\n".join(section_lines)

    def upload_charts_to_hf(self):
        """グラフ画像をHuggingFace Hubにアップロード"""
        chart_files = [
            'abc_individual_benchmark_comparison.png',
            'abc_comprehensive_benchmark_overview.png',
            'abc_statistical_significance.png',
            'abc_industry_comparison.png',
            'abc_ranking_heatmap.png',
            'abc_industry_standard_benchmarks.png',
            'abc_advanced_benchmarks.png',
            'abc_elyza100_benchmark.png'
        ]
        
        logger.info(f"[UPLOAD] Uploading charts to {self.repo_id}...")
        
        for chart_file in chart_files:
            chart_path = self.visualization_dir / chart_file
            if chart_path.exists():
                try:
                    self.api.upload_file(
                        path_or_fileobj=str(chart_path),
                        path_in_repo=f"abc_test_charts/{chart_file}",
                        repo_id=self.repo_id,
                        commit_message=f"Upload ABC test visualization: {chart_file}"
                    )
                    logger.info(f"[UPLOAD] Chart uploaded: {chart_file}")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to upload {chart_file}: {e}")
            else:
                logger.warning(f"[MISSING] Chart file not found: {chart_path}")

    def update_readme(self):
        """READMEを更新"""
        try:
            # 既存のREADMEを取得
            try:
                readme_path = self.api.hf_hub_download(
                    repo_id=self.repo_id,
                    filename="README.md",
                    local_dir="."
                )
                with open("README.md", "r", encoding="utf-8") as f:
                    current_content = f.read()
            except:
                logger.warning("[WARNING] Could not download existing README, creating new one")
                current_content = f"# {self.repo_id.split('/')[-1]}\n\n"
            
            # ベンチマークテーブルセクション生成
            benchmark_table = self.generate_benchmark_table()
            
            # 可視化セクション生成
            visualization_section = self.generate_visualization_section()
            
            # 既存のセクションを置き換えまたは追加
            if "## [STATS] Benchmark Results" in current_content:
                # 既存セクションを置き換え
                import re
                pattern = r"## [STATS] Benchmark Results.*?(?=## |$)"
                current_content = re.sub(pattern, benchmark_table + "\n\n", current_content, flags=re.DOTALL)
            else:
                # 新規追加
                current_content += "\n\n" + benchmark_table + "\n\n"
            
            if "## [STATS] ABC Test Visualizations" in current_content:
                # 既存セクションを置き換え
                import re
                pattern = r"## [STATS] ABC Test Visualizations.*?(?=## |$)"
                current_content = re.sub(pattern, visualization_section + "\n\n", current_content, flags=re.DOTALL)
            else:
                # 新規追加
                current_content += "\n\n" + visualization_section + "\n\n"
            
            # 一時ファイルに保存
            temp_readme = "temp_readme.md"
            with open(temp_readme, "w", encoding="utf-8") as f:
                f.write(current_content)
            
            # HuggingFace Hubにアップロード
            self.api.upload_file(
                path_or_fileobj=temp_readme,
                path_in_repo="README.md",
                repo_id=self.repo_id,
                commit_message="Update README with comprehensive ABC benchmark results and visualizations"
            )
            
            logger.info("[UPDATE] README updated successfully")
            
            # クリーンアップ
            if os.path.exists(temp_readme):
                os.remove(temp_readme)
            if os.path.exists("README.md"):
                os.remove("README.md")
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to update README: {e}")
            raise

    def run(self):
        """実行"""
        logger.info(f"[HF] Updating README for {self.repo_id}...")
        
        # グラフ画像アップロード
        self.upload_charts_to_hf()
        
        # README更新
        self.update_readme()
        
        logger.info("[SUCCESS] HF README update completed!")
        logger.info(f"[REPO] https://huggingface.co/{self.repo_id}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update HuggingFace README with ABC Benchmark Results')
    parser.add_argument('--repo_id', type=str, required=True,
                       help='HuggingFace repository ID (e.g., username/model-name)')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to ABC test results JSON file')
    parser.add_argument('--statistical_analysis_file', type=str, required=True,
                       help='Path to statistical analysis results JSON file')
    parser.add_argument('--visualization_dir', type=str,
                       default='results/abc_testing/visualizations',
                       help='Directory containing visualization charts')
    
    args = parser.parse_args()
    
    # README更新実行
    updater = HFReadmeUpdater(
        repo_id=args.repo_id,
        results_file=args.results_file,
        statistical_analysis_file=args.statistical_analysis_file,
        visualization_dir=args.visualization_dir
    )
    
    updater.run()


if __name__ == "__main__":
    main()
