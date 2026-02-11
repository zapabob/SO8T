#!/usr/bin/env python3
"""
Generate A/B/C Test Report with Statistical Analysis and Visualization
"""

import json
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd

def load_test_results(results_dir: str):
    """Load test results from directory"""

    results_dir = Path(results_dir)
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"[ERROR] No JSON files found in {results_dir}")
        return None

    # Load the latest results
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"[INFO] Loading results from: {latest_file}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def analyze_abc_results(data):
    """Analyze A/B/C test results with statistical methods"""

    baseline_results = data['baseline']['results']
    improved_results = data['improved']['results']
    baseline_evals = data['baseline']['evaluations']
    improved_evals = data['improved']['evaluations']

    analysis = {
        'accuracy': {},
        'response_time': {},
        'reasoning_quality': {},
        'hallucination_rate': {},
        'statistical_tests': {}
    }

    # Accuracy analysis
    baseline_correct = sum(1 for e in baseline_evals if e['correct'])
    improved_correct = sum(1 for e in improved_evals if e['correct'])
    total_tests = len(baseline_evals)

    analysis['accuracy'] = {
        'baseline': baseline_correct / total_tests,
        'improved': improved_correct / total_tests,
        'improvement': (improved_correct - baseline_correct) / total_tests,
        'baseline_count': baseline_correct,
        'improved_count': improved_correct,
        'total': total_tests
    }

    # Response time analysis
    baseline_times = [r['response_time'] for r in baseline_results]
    improved_times = [r['response_time'] for r in improved_results]

    analysis['response_time'] = {
        'baseline_mean': np.mean(baseline_times),
        'improved_mean': np.mean(improved_times),
        'baseline_std': np.std(baseline_times),
        'improved_std': np.std(improved_times),
        'speed_improvement': np.mean(baseline_times) - np.mean(improved_times)
    }

    # Reasoning quality analysis (ordinal scale)
    quality_map = {'poor': 1, 'fair': 2, 'good': 3, 'excellent': 4}
    baseline_qualities = [quality_map[e['reasoning_quality']] for e in baseline_evals]
    improved_qualities = [quality_map[e['reasoning_quality']] for e in improved_evals]

    analysis['reasoning_quality'] = {
        'baseline_mean': np.mean(baseline_qualities),
        'improved_mean': np.mean(improved_qualities),
        'baseline_std': np.std(baseline_qualities),
        'improved_std': np.std(improved_qualities),
        'improvement': np.mean(improved_qualities) - np.mean(baseline_qualities)
    }

    # Hallucination rate
    baseline_hallucinations = sum(1 for e in baseline_evals if e['hallucination_detected'])
    improved_hallucinations = sum(1 for e in improved_evals if e['hallucination_detected'])

    analysis['hallucination_rate'] = {
        'baseline': baseline_hallucinations / total_tests,
        'improved': improved_hallucinations / total_tests,
        'reduction': (baseline_hallucinations - improved_hallucinations) / total_tests,
        'baseline_count': baseline_hallucinations,
        'improved_count': improved_hallucinations
    }

    # Statistical tests
    try:
        # Accuracy: Proportions test
        from statsmodels.stats.proportion import proportions_ztest
        success = np.array([baseline_correct, improved_correct])
        nobs = np.array([total_tests, total_tests])
        stat, p_value = proportions_ztest(success, nobs)
        analysis['statistical_tests']['accuracy'] = {
            'z_statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        # Response time: t-test
        t_stat, p_value = stats.ttest_ind(baseline_times, improved_times)
        analysis['statistical_tests']['response_time'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        # Reasoning quality: Mann-Whitney U test (non-parametric for ordinal)
        u_stat, p_value = stats.mannwhitneyu(baseline_qualities, improved_qualities, alternative='two-sided')
        analysis['statistical_tests']['reasoning_quality'] = {
            'u_statistic': u_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    except Exception as e:
        print(f"[WARNING] Statistical tests failed: {e}")
        analysis['statistical_tests'] = {'error': str(e)}

    return analysis

def create_performance_comparison_chart(analysis, output_path: str):
    """Create performance comparison chart with error bars"""

    metrics = ['Accuracy', 'Response Time', 'Reasoning Quality', 'Hallucination Rate']
    baseline_values = [
        analysis['accuracy']['baseline'],
        analysis['response_time']['baseline_mean'],
        analysis['reasoning_quality']['baseline_mean'],
        analysis['hallucination_rate']['baseline']
    ]
    improved_values = [
        analysis['accuracy']['improved'],
        analysis['response_time']['improved_mean'],
        analysis['reasoning_quality']['improved_mean'],
        analysis['hallucination_rate']['improved']
    ]
    baseline_errors = [
        0,  # Accuracy: no error bar
        analysis['response_time']['baseline_std'],
        analysis['reasoning_quality']['baseline_std'],
        0   # Hallucination rate: no error bar
    ]
    improved_errors = [
        0,  # Accuracy: no error bar
        analysis['response_time']['improved_std'],
        analysis['reasoning_quality']['improved_std'],
        0   # Hallucination rate: no error bar
    ]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    axes = [ax1, ax2, ax3, ax4]

    colors = ['#FF6B6B', '#4ECDC4']  # Red for baseline, teal for improved

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        x = np.array([0, 1])  # Baseline, Improved
        y = np.array([baseline_values[i], improved_values[i]])
        yerr = np.array([baseline_errors[i], improved_errors[i]])

        bars = ax.bar(x, y, color=colors, alpha=0.7, capsize=5,
                      yerr=yerr if yerr.max() > 0 else None)

        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Model A\n(Baseline)', 'AEGIS\n(Improved)'])

        # Add value labels on bars
        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (yerr[i] if yerr[i] > 0 else 0.01),
                   '.3f', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Special formatting for different metrics
        if 'Time' in metric:
            ax.set_ylabel('Time (seconds)')
        elif 'Rate' in metric:
            ax.set_ylabel('Rate (0-1)')
            ax.set_ylim(0, 1)
        elif 'Quality' in metric:
            ax.set_ylabel('Quality Score (1-4)')
            ax.set_ylim(1, 4)
        else:
            ax.set_ylabel('Accuracy (0-1)')
            ax.set_ylim(0, 1)

        # Add improvement annotation
        if i == 0:  # Accuracy
            improvement = analysis['accuracy']['improvement']
            ax.text(0.5, max(y) + 0.05, '+.1%',
                   ha='center', fontsize=11, fontweight='bold',
                   color='green' if improvement > 0 else 'red')
        elif i == 1:  # Response time
            improvement = analysis['response_time']['speed_improvement']
            ax.text(0.5, min(y) - 0.1, '+.2f',
                   ha='center', fontsize=11, fontweight='bold',
                   color='green' if improvement > 0 else 'red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[CHART] Performance comparison chart saved: {output_path}")

def create_detailed_report(data, analysis, output_path: str):
    """Create detailed report with statistical analysis"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# A/B/C Test Results: Model A vs AEGIS Analysis Report\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        f.write("**Models Compared:**\n")
        f.write(f"- **Model A (Baseline):** {data['baseline']['model']}\n")
        f.write(f"- **AEGIS (Improved):** {data['improved']['model']}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        acc_improvement = analysis['accuracy']['improvement']
        time_improvement = analysis['response_time']['speed_improvement']
        hallucination_reduction = analysis['hallucination_rate']['reduction']

        f.write("### Key Findings\n\n")
        f.write("#### Accuracy Performance\n")
        f.write(".1%")
        f.write(".1%")
        if acc_improvement >= 0:
            f.write(" (+.1%)\n\n")
        else:
            f.write(" (-.1%)\n\n")

        f.write("#### Response Speed\n")
        f.write(".2f")
        f.write(".2f")
        if time_improvement > 0:
            f.write(" (+.2f faster)\n\n")
        else:
            f.write(" (-.2f slower)\n\n")

        f.write("#### Hallucination Control\n")
        f.write(".1%")
        f.write(".1%")
        if hallucination_reduction > 0:
            f.write(" (-.1% reduction)\n\n")
        else:
            f.write(" (+.1% increase)\n\n")

        # Detailed Results
        f.write("## Detailed Results\n\n")

        # Performance Metrics Table
        f.write("### Performance Metrics\n\n")
        f.write("| Metric | Model A (Baseline) | AEGIS (Improved) | Improvement |\n")
        f.write("|--------|-------------------|------------------|-------------|\n")
        f.write(".1%")
        f.write(".2f")
        f.write("+.1%")
        f.write(".2f")
        f.write("+.2f")
        f.write(".1%")
        f.write("+.1%")
        f.write(".1%")
        f.write("+.1%")
        f.write("\n\n")

        # Statistical Significance
        f.write("### Statistical Significance\n\n")
        if 'statistical_tests' in analysis and 'error' not in analysis['statistical_tests']:
            f.write("#### Accuracy Test\n")
            acc_test = analysis['statistical_tests']['accuracy']
            f.write(".3f")
            f.write(".4f")
            f.write("**Significant**" if acc_test['significant'] else "**Not significant**")
            f.write("\n\n")

            f.write("#### Response Time Test\n")
            time_test = analysis['statistical_tests']['response_time']
            f.write(".3f")
            f.write(".4f")
            f.write("**Significant**" if time_test['significant'] else "**Not significant**")
            f.write("\n\n")

            f.write("#### Reasoning Quality Test\n")
            qual_test = analysis['statistical_tests']['reasoning_quality']
            f.write(".3f")
            f.write(".4f")
            f.write("**Significant**" if qual_test['significant'] else "**Not significant**")
            f.write("\n\n")
        else:
            f.write("*Statistical tests could not be performed due to missing dependencies.*\n\n")

        # Individual Test Case Analysis
        f.write("## Individual Test Case Analysis\n\n")

        baseline_results = data['baseline']['results']
        improved_results = data['improved']['results']
        baseline_evals = data['baseline']['evaluations']
        improved_evals = data['improved']['evaluations']

        for i, (b_res, i_res, b_eval, i_eval) in enumerate(zip(
            baseline_results, improved_results, baseline_evals, improved_evals), 1):

            f.write(f"### Test Case {i}\n\n")
            f.write(f"**Question:** {b_res['question']}\n\n")

            f.write("#### Model A (Baseline)\n")
            f.write(f"- **Correct:** {'✅' if b_eval['correct'] else '❌'}\n")
            f.write(".2f")
            f.write(f"- **Reasoning Quality:** {b_eval['reasoning_quality']}\n")
            f.write(f"- **Hallucination:** {'⚠️' if b_eval['hallucination_detected'] else '✅'}\n")
            f.write(f"- **Response:** {b_res['response'][:300]}{'...' if len(b_res['response']) > 300 else ''}\n\n")

            f.write("#### AEGIS (Improved)\n")
            f.write(f"- **Correct:** {'✅' if i_eval['correct'] else '❌'}\n")
            f.write(".2f")
            f.write(f"- **Reasoning Quality:** {i_eval['reasoning_quality']}\n")
            f.write(f"- **Hallucination:** {'⚠️' if i_eval['hallucination_detected'] else '✅'}\n")
            f.write(f"- **Response:** {i_res['response'][:300]}{'...' if len(i_res['response']) > 300 else ''}\n\n")

        # Descriptive Statistics
        f.write("## Descriptive Statistics\n\n")

        f.write("### Response Time Statistics\n")
        f.write("| Model | Mean | Std Dev | Min | Max |\n")
        f.write("|-------|------|---------|-----|-----|\n")
        baseline_times = [r['response_time'] for r in baseline_results]
        improved_times = [r['response_time'] for r in improved_results]
        f.write(".2f")
        f.write(".2f")
        f.write(".2f")
        f.write(".2f")
        f.write("\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        if acc_improvement > 0.1:
            f.write("### 🎯 **Strong Positive Results**\n")
            f.write("- Current improvements are significant and promising\n")
            f.write("- Consider deploying AEGIS for production testing\n")
            f.write("- Further optimization may yield additional gains\n\n")
        elif acc_improvement > 0:
            f.write("### 📈 **Moderate Improvement**\n")
            f.write("- Improvements detected but may need further tuning\n")
            f.write("- Consider adjusting Alpha Gate scale factor or training parameters\n")
            f.write("- Additional logic tuning cycles may be beneficial\n\n")
        else:
            f.write("### 🔄 **Needs Further Investigation**\n")
            f.write("- No significant improvement detected\n")
            f.write("- Consider different approaches:\n")
            f.write("  - Reduce Alpha Gate scale factor further (e.g., 0.6)\n")
            f.write("  - Increase logic training epochs\n")
            f.write("  - Use different dataset combinations\n")
            f.write("  - Review SO(8) transformation parameters\n\n")

        f.write("## Technical Details\n\n")
        f.write("### Test Configuration\n")
        f.write(f"- **Test Cases:** {len(baseline_results)}\n")
        f.write("- **Timeout:** 30 seconds per query\n")
        f.write("- **Evaluation Metrics:** Accuracy, Response Time, Reasoning Quality, Hallucination Detection\n")
        f.write("- **Statistical Tests:** Proportions Z-test, T-test, Mann-Whitney U test\n\n")

        f.write("### Model Specifications\n")
        f.write("- **Model A:** Standard Phi-3.5 fine-tuned model\n")
        f.write("- **AEGIS:** SO(8) geometric intelligence with Alpha Gate (φ = 1.618)\n")
        f.write("- **Quantization:** Q8_0 for both models\n\n")

    print(f"[REPORT] Detailed analysis report saved: {output_path}")

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate A/B/C Test Report")
    parser.add_argument("--results-dir", default="_docs/benchmark_results/abc_test_initial",
                       help="Directory containing test results")
    parser.add_argument("--output-dir", default="D:/abc_test_reports",
                       help="Output directory for reports and charts")

    args = parser.parse_args()

    print("=" * 60)
    print("A/B/C TEST REPORT GENERATOR")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test results
    data = load_test_results(args.results_dir)
    if not data:
        print("[ERROR] Could not load test results")
        return

    # Analyze results
    print("[ANALYSIS] Analyzing test results...")
    analysis = analyze_abc_results(data)

    # Generate chart
    chart_path = output_dir / "abc_performance_comparison.png"
    print("[CHART] Creating performance comparison chart...")
    create_performance_comparison_chart(analysis, str(chart_path))

    # Generate detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"abc_test_analysis_report_{timestamp}.md"
    print("[REPORT] Generating detailed analysis report...")
    create_detailed_report(data, analysis, str(report_path))

    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETED!")
    print("=" * 60)
    print(f"Chart: {chart_path}")
    print(f"Report: {report_path}")
    print("\nKey Findings:")
    print(".1%")
    print(".2f")
    print(".1%")

if __name__ == "__main__":
    main()
