#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Industry Standard + AGI ABC Test Report
Markdown report with embedded graphs and summary statistics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def load_analysis(analysis_dir: Path) -> Dict[str, Any]:
    """分析結果をロード"""
    analysis_file = analysis_dir / "statistical_analysis.json"
    if analysis_file.exists():
        with analysis_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def format_statistics_table(stats_dict: Dict[str, Any]) -> str:
    """統計量を表形式でフォーマット"""
    lines = [
        "| Metric | Value |",
        "|--------|-------|"
    ]
    
    metrics = {
        'mean': 'Mean',
        'std': 'Std Dev',
        'median': 'Median',
        'min': 'Min',
        'max': 'Max',
        'q25': 'Q25',
        'q75': 'Q75',
        'count': 'Count',
    }
    
    for key, label in metrics.items():
        value = stats_dict.get(key, 0)
        if isinstance(value, float):
            lines.append(f"| {label} | {value:.3f} |")
        else:
            lines.append(f"| {label} | {value} |")
    
    return "\n".join(lines)


def generate_markdown_report(analysis: Dict[str, Any], results_dir: Path, analysis_dir: Path, output_path: Path):
    """Markdownレポートを生成"""
    lines = [
        "# Industry Standard + AGI ABC Test Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
    ]
    
    # 全体統計
    if 'overall_comparison' in analysis and 'overall_stats' in analysis['overall_comparison']:
        overall_stats = analysis['overall_comparison']['overall_stats']
        lines.append("### Overall Performance Statistics")
        lines.append("")
        
        for model_name, stats_dict in overall_stats.items():
            mean = stats_dict.get('mean', 0)
            std = stats_dict.get('std', 0)
            lines.append(f"**{model_name}**: Mean = {mean:.3f} ± {std:.3f} (n={stats_dict.get('count', 0)})")
        
        lines.append("")
    
    # モデル比較
    if 'overall_comparison' in analysis and 'comparisons' in analysis['overall_comparison']:
        comparisons = analysis['overall_comparison']['comparisons']
        if comparisons:
            lines.append("### Model Comparisons")
            lines.append("")
            lines.append("| Model 1 | Model 2 | Effect Size (Cohen's d) | t-test p-value | Significant |")
            lines.append("|---------|---------|------------------------|----------------|-------------|")
            
            for comp in comparisons:
                model1 = comp['model1']
                model2 = comp['model2']
                effect = comp['effect_size']
                tests = comp['statistical_tests']
                
                cohens_d = effect.get('cohens_d', 0)
                interpretation = effect.get('interpretation', 'N/A')
                
                p_value = "N/A"
                significant = False
                if 't_test' in tests and tests['t_test'] and 'p_value' in tests['t_test']:
                    p_value = f"{tests['t_test']['p_value']:.4f}"
                    significant = tests['t_test'].get('significant', False)
                
                sig_mark = "[OK]" if significant else "[NG]"
                lines.append(f"| {model1} | {model2} | {cohens_d:.3f} ({interpretation}) | {p_value} | {sig_mark} |")
            
            lines.append("")
    
    # カテゴリ別分析
    if 'category_analysis' in analysis:
        lines.append("## Category-wise Analysis")
        lines.append("")
        
        category_data = analysis['category_analysis']
        categories = list(category_data.keys())
        
        for category in categories:
            lines.append(f"### {category.replace('_', ' ').title()}")
            lines.append("")
            
            cat_data = category_data[category]
            model_names = sorted(cat_data.keys())
            
            # カテゴリ別統計表
            lines.append("| Model | Mean | Std Dev | 95% CI Lower | 95% CI Upper | Count |")
            lines.append("|-------|------|---------|--------------|--------------|-------|")
            
            for model_name in model_names:
                stats_dict = cat_data[model_name].get('stats', {})
                ci = cat_data[model_name].get('ci', (0, 0))
                
                mean = stats_dict.get('mean', 0)
                std = stats_dict.get('std', 0)
                ci_lower, ci_upper = ci
                count = stats_dict.get('count', 0)
                
                lines.append(f"| {model_name} | {mean:.3f} | {std:.3f} | {ci_lower:.3f} | {ci_upper:.3f} | {count} |")
            
            lines.append("")
    
    # 四重推論分析
    if 'quadruple_reasoning_analysis' in analysis:
        lines.append("## Quadruple Reasoning Analysis (AEGIS)")
        lines.append("")
        
        quad_analysis = analysis['quadruple_reasoning_analysis']
        
        for model_name, quad_data in quad_analysis.items():
            lines.append(f"### {model_name}")
            lines.append("")
            
            axis_stats = quad_data.get('axis_stats', {})
            total = quad_data.get('total_responses', 0)
            
            lines.append(f"**Total responses with quadruple reasoning**: {total}")
            lines.append("")
            
            lines.append("| Axis | Mean Length | Std Dev | Median |")
            lines.append("|------|-------------|---------|--------|")
            
            for axis, stats_dict in axis_stats.items():
                mean = stats_dict.get('mean', 0)
                std = stats_dict.get('std', 0)
                median = stats_dict.get('median', 0)
                lines.append(f"| {axis} | {mean:.1f} | {std:.1f} | {median:.1f} |")
            
            lines.append("")
    
    # グラフ埋め込み
    graphs_dir = analysis_dir / "graphs"
    if graphs_dir.exists():
        lines.append("## Visualizations")
        lines.append("")
        
        graph_files = sorted(graphs_dir.glob("*.png"))
        for graph_file in graph_files:
            graph_name = graph_file.stem
            # 相対パスで埋め込み
            rel_path = graph_file.relative_to(output_path.parent)
            lines.append(f"### {graph_name.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"![{graph_name}]({rel_path})")
            lines.append("")
    
    # 結論
    lines.append("## Conclusions")
    lines.append("")
    
    if 'overall_comparison' in analysis and 'comparisons' in analysis['overall_comparison']:
        comparisons = analysis['overall_comparison']['comparisons']
        if comparisons:
            comp = comparisons[0]
            model1 = comp['model1']
            model2 = comp['model2']
            effect = comp['effect_size']
            tests = comp['statistical_tests']
            
            cohens_d = effect.get('cohens_d', 0)
            interpretation = effect.get('interpretation', 'N/A')
            
            if 't_test' in tests and tests['t_test'] and 'p_value' in tests['t_test']:
                p_value = tests['t_test']['p_value']
                significant = tests['t_test'].get('significant', False)
                
                if significant:
                    if cohens_d > 0:
                        winner = model1
                    else:
                        winner = model2
                    lines.append(f"- **Statistical significance**: Significant difference found (p < 0.05)")
                    lines.append(f"- **Effect size**: {abs(cohens_d):.3f} ({interpretation})")
                    lines.append(f"- **Winner**: {winner} shows better performance")
                else:
                    lines.append(f"- **Statistical significance**: No significant difference found (p = {p_value:.4f})")
                    lines.append(f"- **Effect size**: {abs(cohens_d):.3f} ({interpretation})")
                    lines.append(f"- **Conclusion**: Models perform similarly")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by Industry Standard + AGI ABC Test System*")
    
    # ファイル保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[REPORT] Markdown report saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Industry Standard + AGI ABC Test Report"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="結果ディレクトリ",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="分析結果ディレクトリ（デフォルト: results-dir/analysis）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="レポート出力先（デフォルト: _docs/benchmark_results/industry_standard_agi/report.md）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 分析ディレクトリ設定
    if args.analysis_dir is None:
        analysis_dir = args.results_dir / "analysis"
    else:
        analysis_dir = args.analysis_dir
    
    # 出力先設定
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("_docs/benchmark_results/industry_standard_agi") / f"report_{timestamp}.md"
    else:
        output_path = args.output
    
    print("=" * 80)
    print("Industry Standard + AGI ABC Test Report Generator")
    print("=" * 80)
    print(f"Results directory: {args.results_dir}")
    print(f"Analysis directory: {analysis_dir}")
    print(f"Output: {output_path}")
    print()
    
    # 分析結果ロード
    print("[LOAD] Loading analysis results...")
    analysis = load_analysis(analysis_dir)
    
    if not analysis:
        print("[ERROR] No analysis results found")
        return
    
    # レポート生成
    print("[GENERATE] Generating markdown report...")
    generate_markdown_report(analysis, args.results_dir, analysis_dir, output_path)
    
    print(f"\n[COMPLETE] Report generated: {output_path}")


if __name__ == "__main__":
    main()




