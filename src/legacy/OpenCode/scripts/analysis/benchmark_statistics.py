#!/usr/bin/env python3
"""
Benchmark Statistics Analysis for ABC Testing
Analyzes results from modela, AEGIS, AEGISalpha0.6 benchmarks
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class BenchmarkStatisticsAnalyzer:
    """Analyzes benchmark results with statistical methods"""

    def __init__(self):
        self.data_dir = Path("_docs/benchmark_results")
        self.output_dir = Path("_docs/benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['font.family'] = 'DejaVu Sans'

    def load_latest_results(self) -> Dict[str, Any]:
        """Load the most recent benchmark results"""
        json_files = list(self.data_dir.glob("*_results.json"))
        if not json_files:
            raise FileNotFoundError("No benchmark result files found")

        # Sort by modification time (newest first)
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_file = json_files[0]

        print(f"Loading results from: {latest_file}")

        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_dataframe(self, results_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        records = []

        for result in results_data['results']:
            record = {
                'model': result['model'],
                'benchmark': result.get('benchmark', result.get('category', 'unknown')),
                'task_index': result.get('task_index', 0),
                'score': result['score'],
                'response_time': result['response_time'],
                'timestamp': result['timestamp']
            }
            records.append(record)

        df = pd.DataFrame(records)

        # Standardize model names
        model_mapping = {
            'model_a': 'modela',
            'model-a': 'modela',
            'agiasi': 'aegis',
            'aegis-adjusted': 'aegis',
            'aegis-adjusted-0.6': 'aegis_alpha_0_6'
        }
        df['model'] = df['model'].map(model_mapping).fillna(df['model'])

        return df

    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        stats_results = {}

        # Overall statistics by model
        model_stats = df.groupby('model').agg({
            'score': ['mean', 'std', 'count', 'min', 'max'],
            'response_time': ['mean', 'std', 'count', 'min', 'max']
        }).round(3)

        stats_results['model_overall'] = model_stats

        # Statistics by model and benchmark
        benchmark_stats = df.groupby(['model', 'benchmark']).agg({
            'score': ['mean', 'std', 'count'],
            'response_time': ['mean', 'std', 'count']
        }).round(3)

        stats_results['model_benchmark'] = benchmark_stats

        # Performance comparison
        if len(df['model'].unique()) >= 2:
            models = df['model'].unique()
            comparison_data = []

            for benchmark in df['benchmark'].unique():
                benchmark_data = df[df['benchmark'] == benchmark]

                for model in models:
                    model_scores = benchmark_data[benchmark_data['model'] == model]['score']
                    if len(model_scores) > 0:
                        comparison_data.append({
                            'benchmark': benchmark,
                            'model': model,
                            'mean_score': model_scores.mean(),
                            'std_score': model_scores.std() if len(model_scores) > 1 else 0,
                            'count': len(model_scores)
                        })

            stats_results['performance_comparison'] = pd.DataFrame(comparison_data)

        # Statistical tests (if applicable)
        if len(df['model'].unique()) >= 2:
            stats_results['statistical_tests'] = self._perform_statistical_tests(df)

        return stats_results

    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        test_results = {}

        models = df['model'].unique()
        if len(models) < 2:
            return test_results

        # Overall score comparison
        model_groups = [df[df['model'] == model]['score'] for model in models]

        # ANOVA test (if more than 2 groups)
        if len(models) > 2:
            try:
                f_stat, p_value = stats.f_oneway(*model_groups)
                test_results['anova_overall'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except:
                pass

        # T-tests for pairs
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                model1, model2 = models[i], models[j]
                scores1 = df[df['model'] == model1]['score']
                scores2 = df[df['model'] == model2]['score']

                if len(scores1) > 1 and len(scores2) > 1:
                    try:
                        t_stat, p_value = stats.ttest_ind(scores1, scores2)
                        test_results[f'ttest_{model1}_vs_{model2}'] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except:
                        pass

        return test_results

    def create_visualizations(self, df: pd.DataFrame, stats_results: Dict[str, Any]) -> List[str]:
        """Create comprehensive visualizations with error bars"""
        charts = []

        # 1. Overall Performance Comparison with Error Bars
        charts.extend(self._create_overall_comparison_chart(df, stats_results))

        # 2. Benchmark-specific Performance
        charts.extend(self._create_benchmark_comparison_charts(df, stats_results))

        # 3. Score Distribution Analysis
        charts.extend(self._create_distribution_analysis(df))

        # 4. Response Time Analysis
        charts.extend(self._create_response_time_analysis(df))

        # 5. Statistical Significance Plot
        if 'statistical_tests' in stats_results:
            charts.extend(self._create_statistical_plot(stats_results))

        return charts

    def _create_overall_comparison_chart(self, df: pd.DataFrame, stats_results: Dict[str, Any]) -> List[str]:
        """Create overall performance comparison with error bars"""
        charts = []

        models = df['model'].unique()
        means = []
        stds = []

        for model in models:
            model_data = df[df['model'] == model]['score']
            means.append(model_data.mean())
            stds.append(model_data.std())

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, means, yerr=stds, capsize=5, alpha=0.8,
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])

        ax.set_xlabel('Model')
        ax.set_ylabel('Average Score')
        ax.set_title('ABC Benchmark: Overall Performance Comparison\n(with Standard Error Bars)')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_overall_performance_with_errors.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(str(chart_file))

        return charts

    def _create_benchmark_comparison_charts(self, df: pd.DataFrame, stats_results: Dict[str, Any]) -> List[str]:
        """Create benchmark-specific comparison charts"""
        charts = []
        benchmarks = df['benchmark'].unique()

        fig, axes = plt.subplots(len(benchmarks), 1, figsize=(12, 4*len(benchmarks)))
        if len(benchmarks) == 1:
            axes = [axes]

        models = df['model'].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for idx, benchmark in enumerate(benchmarks):
            ax = axes[idx]
            benchmark_data = df[df['benchmark'] == benchmark]

            means = []
            stds = []

            for model in models:
                model_scores = benchmark_data[benchmark_data['model'] == model]['score']
                if len(model_scores) > 0:
                    means.append(model_scores.mean())
                    stds.append(model_scores.std() if len(model_scores) > 1 else 0)
                else:
                    means.append(0)
                    stds.append(0)

            bars = ax.bar(models, means, yerr=stds, capsize=5, alpha=0.8, color=colors[:len(models)])
            ax.set_title(f'{benchmark.upper()} Benchmark Performance')
            ax.set_ylabel('Average Score')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                            f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_benchmark_detailed_comparison.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(str(chart_file))

        return charts

    def _create_distribution_analysis(self, df: pd.DataFrame) -> List[str]:
        """Create score distribution analysis"""
        charts = []

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Box plot
        sns.boxplot(data=df, x='model', y='score', ax=ax1)
        ax1.set_title('Score Distribution by Model (Box Plot)')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3)

        # Violin plot
        sns.violinplot(data=df, x='model', y='score', ax=ax2)
        ax2.set_title('Score Distribution by Model (Violin Plot)')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_score_distributions.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(str(chart_file))

        return charts

    def _create_response_time_analysis(self, df: pd.DataFrame) -> List[str]:
        """Create response time analysis"""
        charts = []

        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter out zero response times (errors)
        valid_data = df[df['response_time'] > 0]

        if len(valid_data) > 0:
            sns.boxplot(data=valid_data, x='model', y='response_time', ax=ax)
            ax.set_title('Response Time Distribution by Model')
            ax.set_ylabel('Response Time (seconds)')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_response_time_analysis.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            charts.append(str(chart_file))

        return charts

    def _create_statistical_plot(self, stats_results: Dict[str, Any]) -> List[str]:
        """Create statistical significance visualization"""
        charts = []
        tests = stats_results.get('statistical_tests', {})

        if not tests:
            return charts

        # Extract p-values for visualization
        test_names = []
        p_values = []

        for test_name, results in tests.items():
            if 'p_value' in results:
                test_names.append(test_name.replace('_', ' ').title())
                p_values.append(results['p_value'])

        if p_values:
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(test_names, [-np.log10(p) for p in p_values])

            ax.set_xlabel('-log10(p-value)')
            ax.set_title('Statistical Significance (-log10 scale)\nHigher bars indicate stronger significance')
            ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7,
                      label='p=0.05 threshold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_statistical_significance.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            charts.append(str(chart_file))

        return charts

    def generate_comprehensive_report(self, df: pd.DataFrame, stats_results: Dict[str, Any], charts: List[str]) -> str:
        """Generate comprehensive statistical report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"{timestamp}_comprehensive_statistical_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive ABC Benchmark Statistical Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Data Points:** {len(df)}\n")
            f.write(f"**Models Tested:** {', '.join(df['model'].unique())}\n")
            f.write(f"**Benchmarks:** {', '.join(df['benchmark'].unique())}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            self._write_executive_summary(f, stats_results)

            # Detailed Statistics
            f.write("## Detailed Statistics\n\n")
            self._write_detailed_statistics(f, stats_results)

            # Performance Analysis
            f.write("## Performance Analysis\n\n")
            self._write_performance_analysis(f, df, stats_results)

            # Statistical Tests
            if 'statistical_tests' in stats_results:
                f.write("## Statistical Significance Tests\n\n")
                self._write_statistical_tests(f, stats_results['statistical_tests'])

            # Visualizations
            f.write("## Visualizations\n\n")
            for chart in charts:
                chart_name = Path(chart).name
                f.write(f"![{chart_name}]({chart_name})\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            self._write_recommendations(f, stats_results)

            # Raw Data Summary
            f.write("## Raw Data Summary\n\n")
            f.write("### Score Statistics by Model\n")
            f.write(df.groupby('model')['score'].describe().round(3).to_markdown())
            f.write("\n\n")

            f.write("### Score Statistics by Benchmark\n")
            f.write(df.groupby('benchmark')['score'].describe().round(3).to_markdown())
            f.write("\n\n")

        return str(report_file)

    def _write_executive_summary(self, f, stats_results: Dict[str, Any]):
        """Write executive summary"""
        if 'model_overall' in stats_results:
            model_stats = stats_results['model_overall']
            f.write("### Overall Performance Ranking\n\n")

            # Sort by mean score
            sorted_models = model_stats[('score', 'mean')].sort_values(ascending=False)

            for i, (model, score) in enumerate(sorted_models.items(), 1):
                std = model_stats.loc[model, ('score', 'std')]
                count = model_stats.loc[model, ('score', 'count')]
                f.write(f"{i}. **{model}**: {score:.3f} ± {std:.3f} (n={count})\n")

            f.write("\n")

            # Performance gap analysis
            if len(sorted_models) >= 2:
                best_score = sorted_models.iloc[0]
                worst_score = sorted_models.iloc[-1]
                gap = best_score - worst_score
                f.write(f"**Performance Gap:** {gap:.3f} between best and worst performers\n\n")

    def _write_detailed_statistics(self, f, stats_results: Dict[str, Any]):
        """Write detailed statistics"""
        if 'model_overall' in stats_results:
            f.write("### Model Overall Statistics\n\n")
            f.write(stats_results['model_overall'].to_markdown())
            f.write("\n\n")

        if 'model_benchmark' in stats_results:
            f.write("### Model-Benchmark Statistics\n\n")
            f.write(stats_results['model_benchmark'].to_markdown())
            f.write("\n\n")

    def _write_performance_analysis(self, f, df: pd.DataFrame, stats_results: Dict[str, Any]):
        """Write performance analysis"""
        f.write("### Benchmark Performance by Model\n\n")

        for benchmark in df['benchmark'].unique():
            f.write(f"#### {benchmark.upper()}\n\n")

            benchmark_data = df[df['benchmark'] == benchmark]
            benchmark_stats = benchmark_data.groupby('model')['score'].agg(['mean', 'std', 'count']).round(3)

            if not benchmark_stats.empty:
                f.write(benchmark_stats.to_markdown())
                f.write("\n\n")

                # Find best performer for this benchmark
                best_model = benchmark_stats['mean'].idxmax()
                best_score = benchmark_stats.loc[best_model, 'mean']
                f.write(f"**Best Performer:** {best_model} ({best_score:.3f})\n\n")

    def _write_statistical_tests(self, f, statistical_tests: Dict[str, Any]):
        """Write statistical test results"""
        if 'anova_overall' in statistical_tests:
            anova = statistical_tests['anova_overall']
            f.write("### ANOVA Test (Overall Performance)\n\n")
            f.write(f"- **F-statistic:** {anova['f_statistic']:.3f}\n")
            f.write(f"- **p-value:** {anova['p_value']:.3f}\n")
            f.write(f"- **Significant:** {'Yes' if anova['significant'] else 'No'} (α=0.05)\n\n")

        f.write("### T-Test Results (Pairwise Comparisons)\n\n")
        for test_name, results in statistical_tests.items():
            if test_name.startswith('ttest_'):
                model1, model2 = test_name.replace('ttest_', '').replace('_vs_', ' vs ').split(' vs ')
                f.write(f"#### {model1} vs {model2}\n")
                f.write(f"- **t-statistic:** {results['t_statistic']:.3f}\n")
                f.write(f"- **p-value:** {results['p_value']:.3f}\n")
                f.write(f"- **Significant:** {'Yes' if results['significant'] else 'No'} (α=0.05)\n\n")

    def _write_recommendations(self, f, stats_results: Dict[str, Any]):
        """Write recommendations based on analysis"""
        f.write("### Model Selection Guidelines\n\n")

        if 'model_overall' in stats_results:
            model_stats = stats_results['model_overall']

            # Best overall model
            best_overall = model_stats[('score', 'mean')].idxmax()
            f.write(f"1. **Overall Best Performer:** {best_overall}\n")
            f.write("   - Recommended for general-purpose applications\n\n")

            # Most consistent model (lowest std)
            most_consistent = model_stats[('score', 'std')].idxmin()
            f.write(f"2. **Most Consistent:** {most_consistent}\n")
            f.write("   - Recommended for applications requiring predictable performance\n\n")

            # Fastest model
            fastest = model_stats[('response_time', 'mean')].idxmin()
            f.write(f"3. **Fastest Response:** {fastest}\n")
            f.write("   - Recommended for real-time applications\n\n")

        f.write("### Benchmark-Specific Recommendations\n\n")
        f.write("- **Mathematical Reasoning:** Choose highest-scoring model for quantitative tasks\n")
        f.write("- **Scientific Knowledge:** Select model with strong factual knowledge\n")
        f.write("- **Japanese Language:** Use model optimized for linguistic tasks\n")
        f.write("- **Security/Ethics:** Prioritize models with safety alignment\n")
        f.write("- **AGI Tasks:** Select most advanced reasoning capabilities\n\n")

def main():
    """Main execution function"""
    print("[STATISTICS] Starting Comprehensive ABC Benchmark Analysis")
    print("=" * 80)

    analyzer = BenchmarkStatisticsAnalyzer()

    try:
        # Load latest results
        results_data = analyzer.load_latest_results()

        # Create DataFrame
        df = analyzer.create_dataframe(results_data)
        print(f"[DATA] Loaded {len(df)} data points from {len(df['model'].unique())} models")

        # Calculate statistics
        stats_results = analyzer.calculate_statistics(df)
        print("[STATS] Calculated comprehensive statistics")

        # Create visualizations
        charts = analyzer.create_visualizations(df, stats_results)
        print(f"[CHARTS] Generated {len(charts)} visualization files")

        # Generate comprehensive report
        report_file = analyzer.generate_comprehensive_report(df, stats_results, charts)
        print(f"[REPORT] Generated comprehensive statistical report")

        print("\n[RESULTS]")
        print(f"  Report: {report_file}")
        print(f"  Charts: {len(charts)} files")
        for chart in charts:
            print(f"    - {Path(chart).name}")

        # Print key findings
        if 'model_overall' in stats_results:
            model_stats = stats_results['model_overall']
            best_model = model_stats[('score', 'mean')].idxmax()
            best_score = model_stats.loc[best_model, ('score', 'mean')]
            print(f"\n[BEST PERFORMER] {best_model}: {best_score:.3f}")

        # Play completion sound
        import subprocess
        try:
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-File", "scripts/utils/play_audio_notification.ps1"
            ])
        except:
            pass

        print("\n[COMPLETE] Statistical analysis completed successfully!")

    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


