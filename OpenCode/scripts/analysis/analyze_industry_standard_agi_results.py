#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Industry Standard + AGI Results Analysis
Statistical analysis with error bars and summary statistics
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy import stats
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_results(results_dir: Path) -> Dict[str, Any]:
    """結果をロード"""
    summary_file = results_dir / "summary.json"
    if summary_file.exists():
        with summary_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    # 個別ファイルからロード
    results = {}
    for model_dir in results_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            agi_file = model_dir / f"{model_name}_agi_results.json"
            if agi_file.exists():
                with agi_file.open("r", encoding="utf-8") as f:
                    results[model_name] = {
                        'agi_results': json.load(f),
                        'lm_eval_results': None,
                    }
    return {'results': results}


def calculate_summary_statistics(scores: List[float]) -> Dict[str, float]:
    """要約統計量を計算"""
    if not scores:
        return {}
    
    scores_array = np.array(scores)
    
    return {
        'mean': float(np.mean(scores_array)),
        'std': float(np.std(scores_array)),
        'median': float(np.median(scores_array)),
        'min': float(np.min(scores_array)),
        'max': float(np.max(scores_array)),
        'q25': float(np.percentile(scores_array, 25)),
        'q75': float(np.percentile(scores_array, 75)),
        'count': len(scores_array),
    }


def calculate_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """信頼区間を計算"""
    if len(scores) < 2:
        return (0.0, 0.0)
    
    scores_array = np.array(scores)
    mean = np.mean(scores_array)
    sem = stats.sem(scores_array)
    h = sem * stats.t.ppf((1 + confidence) / 2, len(scores_array) - 1)
    
    return (float(mean - h), float(mean + h))


def calculate_effect_size(scores1: List[float], scores2: List[float]) -> Dict[str, float]:
    """効果量（Cohen's d）を計算"""
    if len(scores1) < 2 or len(scores2) < 2:
        return {'cohens_d': 0.0, 'interpretation': 'insufficient_data'}
    
    scores1_array = np.array(scores1)
    scores2_array = np.array(scores2)
    
    mean1 = np.mean(scores1_array)
    mean2 = np.mean(scores2_array)
    std1 = np.std(scores1_array, ddof=1)
    std2 = np.std(scores2_array, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    
    if pooled_std == 0:
        return {'cohens_d': 0.0, 'interpretation': 'no_variance'}
    
    cohens_d = (mean1 - mean2) / pooled_std
    
    # 解釈
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        interpretation = 'negligible'
    elif abs_d < 0.5:
        interpretation = 'small'
    elif abs_d < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    
    return {
        'cohens_d': float(cohens_d),
        'interpretation': interpretation,
    }


def perform_statistical_tests(scores1: List[float], scores2: List[float]) -> Dict[str, Any]:
    """統計的有意差検定を実行"""
    if len(scores1) < 2 or len(scores2) < 2:
        return {
            't_test': None,
            'mann_whitney': None,
            'error': 'insufficient_data'
        }
    
    scores1_array = np.array(scores1)
    scores2_array = np.array(scores2)
    
    results = {}
    
    # t検定（正規性を仮定）
    try:
        t_stat, p_value_t = stats.ttest_ind(scores1_array, scores2_array)
        results['t_test'] = {
            'statistic': float(t_stat),
            'p_value': float(p_value_t),
            'significant': p_value_t < 0.05,
        }
    except Exception as e:
        results['t_test'] = {'error': str(e)}
    
    # Mann-Whitney U検定（ノンパラメトリック）
    try:
        u_stat, p_value_mw = stats.mannwhitneyu(scores1_array, scores2_array, alternative='two-sided')
        results['mann_whitney'] = {
            'statistic': float(u_stat),
            'p_value': float(p_value_mw),
            'significant': p_value_mw < 0.05,
        }
    except Exception as e:
        results['mann_whitney'] = {'error': str(e)}
    
    return results


def analyze_by_category(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """カテゴリ別に分析"""
    category_analysis = {}
    
    # カテゴリを取得
    categories = set()
    for model_name, model_data in results.items():
        agi_results = model_data.get('agi_results', [])
        for result in agi_results:
            categories.add(result.get('category', 'unknown'))
    
    # カテゴリ別に統計
    for category in categories:
        category_scores = {}
        
        for model_name, model_data in results.items():
            agi_results = model_data.get('agi_results', [])
            category_results = [
                r for r in agi_results
                if r.get('category') == category
            ]
            
            if category_results:
                scores = [r['scores']['overall'] for r in category_results]
                category_scores[model_name] = {
                    'scores': scores,
                    'stats': calculate_summary_statistics(scores),
                    'ci': calculate_confidence_interval(scores),
                }
        
        category_analysis[category] = category_scores
    
    return category_analysis


def analyze_quadruple_reasoning(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """四重推論の分析（AEGISのみ）"""
    quadruple_analysis = {}
    
    for model_name, model_data in results.items():
        agi_results = model_data.get('agi_results', [])
        
        # 四重推論があるモデルのみ
        quadruple_results = [
            r for r in agi_results
            if r.get('quadruple_reasoning') is not None
        ]
        
        if not quadruple_results:
            continue
        
        # 各軸の長さを分析
        axis_lengths = {
            'logic': [],
            'ethics': [],
            'practical': [],
            'creative': [],
            'final': [],
        }
        
        for result in quadruple_results:
            quad = result.get('quadruple_reasoning', {})
            for axis in axis_lengths.keys():
                axis_text = quad.get(axis, '')
                axis_lengths[axis].append(len(axis_text))
        
        # 統計量計算
        axis_stats = {}
        for axis, lengths in axis_lengths.items():
            if lengths:
                axis_stats[axis] = calculate_summary_statistics(lengths)
        
        quadruple_analysis[model_name] = {
            'axis_lengths': axis_lengths,
            'axis_stats': axis_stats,
            'total_responses': len(quadruple_results),
        }
    
    return quadruple_analysis


def generate_comparison_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """モデル間比較統計を生成"""
    model_names = list(results.keys())
    
    if len(model_names) < 2:
        return {'error': 'need_at_least_2_models'}
    
    # 全スコアを取得
    all_scores = {}
    for model_name in model_names:
        agi_results = results[model_name].get('agi_results', [])
        all_scores[model_name] = [r['scores']['overall'] for r in agi_results]
    
    # ペアワイズ比較
    comparisons = []
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            scores1 = all_scores[model1]
            scores2 = all_scores[model2]
            
            if len(scores1) > 0 and len(scores2) > 0:
                comparison = {
                    'model1': model1,
                    'model2': model2,
                    'model1_stats': calculate_summary_statistics(scores1),
                    'model2_stats': calculate_summary_statistics(scores2),
                    'effect_size': calculate_effect_size(scores1, scores2),
                    'statistical_tests': perform_statistical_tests(scores1, scores2),
                }
                comparisons.append(comparison)
    
    return {
        'comparisons': comparisons,
        'overall_stats': {
            model_name: calculate_summary_statistics(scores)
            for model_name, scores in all_scores.items()
        },
    }


def create_visualizations(analysis: Dict[str, Any], output_dir: Path):
    """エラーバー付きグラフを生成"""
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    graphs_created = []
    
    # 1. カテゴリ別スコア比較（エラーバー付き）
    if 'category_analysis' in analysis:
        fig, ax = plt.subplots(figsize=(14, 8))
        category_data = analysis['category_analysis']
        
        categories = list(category_data.keys())
        model_names = set()
        for cat_data in category_data.values():
            model_names.update(cat_data.keys())
        model_names = sorted(list(model_names))
        
        x = np.arange(len(categories))
        width = 0.35
        
        for i, model_name in enumerate(model_names):
            means = []
            errors = []
            for category in categories:
                cat_stats = category_data[category].get(model_name, {}).get('stats', {})
                mean = cat_stats.get('mean', 0)
                std = cat_stats.get('std', 0)
                means.append(mean)
                # 標準誤差として使用
                n = cat_stats.get('count', 1)
                se = std / np.sqrt(n) if n > 1 else std
                errors.append(se)
            
            offset = (i - len(model_names) / 2 + 0.5) * width / len(model_names)
            ax.bar(x + offset, means, width / len(model_names), 
                   label=model_name, yerr=errors, capsize=5, alpha=0.8)
        
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Mean Score', fontsize=12)
        ax.set_title('Category-wise Performance Comparison (with Error Bars)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        graph_path = graphs_dir / "category_comparison_errorbars.png"
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        graphs_created.append(str(graph_path))
        print(f"[GRAPH] Created category comparison: {graph_path}")
    
    # 2. 箱ひげ図（分布の可視化）
    if 'overall_comparison' in analysis and 'overall_stats' in analysis['overall_comparison']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # スコアデータを準備（実際のデータが必要）
        # ここでは統計量から箱ひげ図を近似
        overall_stats = analysis['overall_comparison']['overall_stats']
        
        box_data = []
        labels = []
        for model_name, stats_dict in overall_stats.items():
            # 統計量から分布を近似（実際のデータがあればそれを使用）
            mean = stats_dict.get('mean', 0)
            std = stats_dict.get('std', 0)
            q25 = stats_dict.get('q25', mean - std)
            q75 = stats_dict.get('q75', mean + std)
            median = stats_dict.get('median', mean)
            
            # 箱ひげ図用データ（簡易版）
            box_data.append({
                'med': median,
                'q1': q25,
                'q3': q75,
                'whislo': stats_dict.get('min', q25 - 1.5 * (q75 - q25)),
                'whishi': stats_dict.get('max', q75 + 1.5 * (q75 - q25)),
            })
            labels.append(model_name)
        
        bp = ax.bxp(box_data, positions=range(len(labels)), labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Score Distribution Comparison (Box Plot)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        graph_path = graphs_dir / "score_distribution_boxplot.png"
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        graphs_created.append(str(graph_path))
        print(f"[GRAPH] Created box plot: {graph_path}")
    
    # 3. ヒートマップ（カテゴリ×モデル）
    if 'category_analysis' in analysis:
        fig, ax = plt.subplots(figsize=(12, 8))
        category_data = analysis['category_analysis']
        
        categories = list(category_data.keys())
        model_names = set()
        for cat_data in category_data.values():
            model_names.update(cat_data.keys())
        model_names = sorted(list(model_names))
        
        heatmap_data = []
        for category in categories:
            row = []
            for model_name in model_names:
                cat_stats = category_data[category].get(model_name, {}).get('stats', {})
                mean = cat_stats.get('mean', 0)
                row.append(mean)
            heatmap_data.append(row)
        
        heatmap_array = np.array(heatmap_data)
        sns.heatmap(heatmap_array, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=model_names, yticklabels=categories,
                   cbar_kws={'label': 'Mean Score'}, ax=ax)
        
        ax.set_title('Performance Heatmap (Category × Model)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        graph_path = graphs_dir / "performance_heatmap.png"
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        graphs_created.append(str(graph_path))
        print(f"[GRAPH] Created heatmap: {graph_path}")
    
    # 4. 四重推論分析（AEGISのみ）
    if 'quadruple_reasoning_analysis' in analysis:
        quad_analysis = analysis['quadruple_reasoning_analysis']
        
        for model_name, quad_data in quad_analysis.items():
            fig, ax = plt.subplots(figsize=(12, 6))
            
            axis_stats = quad_data.get('axis_stats', {})
            axes = list(axis_stats.keys())
            means = [axis_stats[axis].get('mean', 0) for axis in axes]
            stds = [axis_stats[axis].get('std', 0) for axis in axes]
            
            x = np.arange(len(axes))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            
            ax.set_xlabel('Reasoning Axis', fontsize=12)
            ax.set_ylabel('Mean Response Length (characters)', fontsize=12)
            ax.set_title(f'Quadruple Reasoning Analysis: {model_name}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(axes)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            graph_path = graphs_dir / f"quadruple_reasoning_{model_name}.png"
            plt.savefig(graph_path, dpi=300, bbox_inches='tight')
            plt.close()
            graphs_created.append(str(graph_path))
            print(f"[GRAPH] Created quadruple reasoning analysis: {graph_path}")
    
    return graphs_created


def save_analysis(analysis: Dict[str, Any], output_dir: Path):
    """分析結果を保存"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_file = output_dir / "statistical_analysis.json"
    with analysis_file.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"[SAVE] Statistical analysis saved to {analysis_file}")
    
    # 可視化生成
    graphs = create_visualizations(analysis, output_dir)
    analysis['graphs_created'] = graphs
    
    return analysis_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Industry Standard + AGI test results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="結果ディレクトリ（summary.jsonまたは個別結果ファイルを含む）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="分析結果保存先（デフォルト: results-dir/analysis）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 出力ディレクトリ設定
    if args.output_dir is None:
        output_dir = args.results_dir / "analysis"
    else:
        output_dir = args.output_dir
    
    print("=" * 80)
    print("Industry Standard + AGI Results Analysis")
    print("=" * 80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # 結果ロード
    print("[LOAD] Loading results...")
    data = load_results(args.results_dir)
    results = data.get('results', {})
    
    if not results:
        print("[ERROR] No results found")
        return
    
    print(f"[INFO] Loaded results for {len(results)} models")
    
    # 分析実行
    print("\n[ANALYZE] Performing statistical analysis...")
    
    analysis = {
        'timestamp': data.get('timestamp', ''),
        'overall_comparison': generate_comparison_statistics(results),
        'category_analysis': analyze_by_category(results),
        'quadruple_reasoning_analysis': analyze_quadruple_reasoning(results),
    }
    
    # 保存
    save_analysis(analysis, output_dir)
    
    # サマリー表示
    print("\n[SUMMARY] Statistical Analysis Complete")
    print("-" * 80)
    
    if 'overall_comparison' in analysis:
        comp = analysis['overall_comparison']
        if 'overall_stats' in comp:
            print("\nOverall Statistics:")
            for model_name, stats_dict in comp['overall_stats'].items():
                print(f"  {model_name}:")
                print(f"    Mean: {stats_dict.get('mean', 0):.3f} ± {stats_dict.get('std', 0):.3f}")
                print(f"    Median: {stats_dict.get('median', 0):.3f}")
                print(f"    Count: {stats_dict.get('count', 0)}")
        
        if 'comparisons' in comp:
            print("\nModel Comparisons:")
            for comp_item in comp['comparisons']:
                model1 = comp_item['model1']
                model2 = comp_item['model2']
                effect = comp_item['effect_size']
                tests = comp_item['statistical_tests']
                
                print(f"\n  {model1} vs {model2}:")
                print(f"    Effect size (Cohen's d): {effect.get('cohens_d', 0):.3f} ({effect.get('interpretation', 'N/A')})")
                
                if 't_test' in tests and tests['t_test'] and 'p_value' in tests['t_test']:
                    p_val = tests['t_test']['p_value']
                    sig = tests['t_test'].get('significant', False)
                    print(f"    t-test p-value: {p_val:.4f} {'[SIGNIFICANT]' if sig else '[NOT SIGNIFICANT]'}")
    
    print(f"\n[COMPLETE] Analysis saved to {output_dir}")


if __name__ == "__main__":
    main()

