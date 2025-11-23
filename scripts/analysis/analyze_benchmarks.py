#!/usr/bin/env python3
"""
Benchmark Analysis & Visualization
Generates statistical analysis and error-bar graphs from benchmark results
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(results_file: Path) -> List[Dict]:
    """Load benchmark results from JSON"""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_statistics(results: List[Dict]) -> pd.DataFrame:
    """Calculate mean, std, and 95% CI for each model"""
    # Group by model
    model_data = {}
    for result in results:
        model = result['model_name']
        if model not in model_data:
            model_data[model] = []
        model_data[model].append(result['accuracy'])
    
    stats_data = []
    for model, accuracies in model_data.items():
        mean = np.mean(accuracies)
        std = np.std(accuracies, ddof=1)
        ci = 1.96 * (std / np.sqrt(len(accuracies)))  # 95% CI
        
        stats_data.append({
            'Model': model,
            'Mean': mean,
            'Std': std,
            'CI_95': ci,
            'Min': np.min(accuracies),
            'Max': np.max(accuracies),
            'Trials': len(accuracies)
        })
    
    return pd.DataFrame(stats_data)

def plot_error_bars(mmlu_stats: pd.DataFrame, elyza_stats: pd.DataFrame, output_dir: Path):
    """Generate error-bar line graphs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # MMLU Plot
    models = mmlu_stats['Model'].tolist()
    x_pos = np.arange(len(models))
    
    ax1.errorbar(x_pos, mmlu_stats['Mean'], yerr=mmlu_stats['CI_95'],
                 fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2)
    ax1.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('MMLU-Style Benchmark Results\n(30 Questions, 3 Trials per Model)', 
                  fontsize=16, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for i, (mean, ci) in enumerate(zip(mmlu_stats['Mean'], mmlu_stats['CI_95'])):
        ax1.text(i, mean + ci + 2, f'{mean:.1f}±{ci:.1f}%', 
                ha='center', fontsize=10, fontweight='bold')
    
    # ELYZA Plot
    ax2.errorbar(x_pos, elyza_stats['Mean'], yerr=elyza_stats['CI_95'],
                 fmt='s-', linewidth=2, markersize=8, capsize=5, capthick=2, color='orange')
    ax2.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_title('ELYZA-100 Benchmark Results\n(20 Questions, 3 Trials per Model)', 
                  fontsize=16, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Add value labels
    for i, (mean, ci) in enumerate(zip(elyza_stats['Mean'], elyza_stats['CI_95'])):
        ax2.text(i, mean + ci + 2, f'{mean:.1f}±{ci:.1f}%', 
                ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved error-bar graph: {output_dir / 'benchmark_comparison.png'}")

def plot_combined_performance(mmlu_stats: pd.DataFrame, elyza_stats: pd.DataFrame, output_dir: Path):
    """Generate combined performance graph"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = mmlu_stats['Model'].tolist()
    x_pos = np.arange(len(models))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x_pos - width/2, mmlu_stats['Mean'], width, 
                   yerr=mmlu_stats['CI_95'], label='MMLU-Style (English)',
                   capsize=5, alpha=0.8, color='steelblue')
    bars2 = ax.bar(x_pos + width/2, elyza_stats['Mean'], width,
                   yerr=elyza_stats['CI_95'], label='ELYZA-100 (Japanese)',
                   capsize=5, alpha=0.8, color='coral')
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Comprehensive Benchmark Comparison\n(Error Bars: 95% Confidence Interval)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_benchmark.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined graph: {output_dir / 'combined_benchmark.png'}")

def generate_summary_tables(mmlu_stats: pd.DataFrame, elyza_stats: pd.DataFrame, 
                            mmlu_results: List[Dict], elyza_results: List[Dict],
                            output_dir: Path):
    """Generate CSV summary tables"""
    # Overall statistics
    mmlu_stats.to_csv(output_dir / 'mmlu_summary_statistics.csv', index=False)
    elyza_stats.to_csv(output_dir / 'elyza_summary_statistics.csv', index=False)
    print(f"✓ Saved MMLU statistics: {output_dir / 'mmlu_summary_statistics.csv'}")
    print(f"✓ Saved ELYZA statistics: {output_dir / 'elyza_summary_statistics.csv'}")
    
    # Domain/Category breakdown
    mmlu_domain_data = []
    for result in mmlu_results:
        for domain, stats in result['domain_breakdown'].items():
            mmlu_domain_data.append({
                'Model': result['model_name'],
                'Trial': result['trial'],
                'Domain': domain,
                'Accuracy': stats['accuracy']
            })
    pd.DataFrame(mmlu_domain_data).to_csv(output_dir / 'mmlu_domain_breakdown.csv', index=False)
    print(f"✓ Saved MMLU domain breakdown: {output_dir / 'mmlu_domain_breakdown.csv'}")
    
    elyza_category_data = []
    for result in elyza_results:
        for category, stats in result['category_breakdown'].items():
            elyza_category_data.append({
                'Model': result['model_name'],
                'Trial': result['trial'],
                'Category': category,
                'Accuracy': stats['accuracy']
            })
    pd.DataFrame(elyza_category_data).to_csv(output_dir / 'elyza_category_breakdown.csv', index=False)
    print(f"✓ Saved ELYZA category breakdown: {output_dir / 'elyza_category_breakdown.csv'}")

def perform_significance_tests(mmlu_results: List[Dict], elyza_results: List[Dict], output_dir: Path):
    """Perform t-tests between models"""
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TESTS (t-test)")
    print("="*60)
    
    # Group results by model
    def group_by_model(results):
        grouped = {}
        for r in results:
            model = r['model_name']
            if model not in grouped:
                grouped[model] = []
            grouped[model].append(r['accuracy'])
        return grouped
    
    mmlu_grouped = group_by_model(mmlu_results)
    elyza_grouped = group_by_model(elyza_results)
    
    models = list(mmlu_grouped.keys())
    
    significance_data = []
    
    print("\nMMLU Comparisons:")
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1, model2 = models[i], models[j]
            t_stat, p_value = stats.ttest_ind(mmlu_grouped[model1], mmlu_grouped[model2])
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"  {model1} vs {model2}: t={t_stat:.3f}, p={p_value:.4f} {sig}")
            significance_data.append({
                'Benchmark': 'MMLU',
                'Model_1': model1,
                'Model_2': model2,
                't_statistic': t_stat,
                'p_value': p_value,
                'Significant': sig
            })
    
    print("\nELYZA Comparisons:")
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1, model2 = models[i], models[j]
            t_stat, p_value = stats.ttest_ind(elyza_grouped[model1], elyza_grouped[model2])
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"  {model1} vs {model2}: t={t_stat:.3f}, p={p_value:.4f} {sig}")
            significance_data.append({
                'Benchmark': 'ELYZA',
                'Model_1': model1,
                'Model_2': model2,
                't_statistic': t_stat,
                'p_value': p_value,
                'Significant': sig
            })
    
    pd.DataFrame(significance_data).to_csv(output_dir / 'significance_tests.csv', index=False)
    print(f"\n✓ Saved significance tests: {output_dir / 'significance_tests.csv'}")
    print("\nKey: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

def main():
    """Main analysis function"""
    print("="*60)
    print("BENCHMARK RESULTS ANALYSIS")
    print("="*60)
    
    results_dir = Path("_docs/benchmark_results")
    
    # Load results
    print("\n📊 Loading benchmark results...")
    mmlu_results = load_results(results_dir / "mmlu_results.json")
    elyza_results = load_results(results_dir / "elyza_results.json")
    print(f"  ✓ Loaded {len(mmlu_results)} MMLU results")
    print(f"  ✓ Loaded {len(elyza_results)} ELYZA results")
    
    # Calculate statistics
    print("\n📈 Calculating statistics...")
    mmlu_stats = calculate_statistics(mmlu_results)
    elyza_stats = calculate_statistics(elyza_results)
    
    print("\nMMLU Summary Statistics:")
    print(mmlu_stats.to_string(index=False))
    
    print("\nELYZA Summary Statistics:")
    print(elyza_stats.to_string(index=False))
    
    # Generate visualizations
    print("\n📉 Generating visualizations...")
    plot_error_bars(mmlu_stats, elyza_stats, results_dir)
    plot_combined_performance(mmlu_stats, elyza_stats, results_dir)
    
    # Generate summary tables
    print("\n📋 Generating summary tables...")
    generate_summary_tables(mmlu_stats, elyza_stats, mmlu_results, elyza_results, results_dir)
    
    # Significance tests
    perform_significance_tests(mmlu_results, elyza_results, results_dir)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
