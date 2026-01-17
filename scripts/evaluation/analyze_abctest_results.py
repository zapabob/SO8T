#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B/C Test Results Analysis Script

Analyzes official A/B/C test results with detailed statistical analysis,
significance testing, and comprehensive reporting.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class ABCTestResultsAnalyzer:
    """
    Comprehensive analyzer for A/B/C test results.
    """

    def __init__(self, results_file: str):
        """
        Initialize analyzer with results file.

        Args:
            results_file: Path to A/B/C test results JSON file
        """
        self.results_file = Path(results_file)
        self.results = self._load_results()
        self.analysis_results = {}

        logger.info(f"Loaded results from: {results_file}")

    def _load_results(self) -> Dict[str, Any]:
        """Load results from JSON file."""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def perform_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive statistical and comparative analysis.

        Returns:
            Complete analysis results
        """
        logger.info("Starting comprehensive analysis")

        analysis = {
            'basic_statistics': self._calculate_basic_statistics(),
            'statistical_tests': self._perform_statistical_tests(),
            'effect_sizes': self._calculate_effect_sizes(),
            'rankings': self._generate_rankings(),
            'confidence_analysis': self._analyze_confidence_intervals(),
            'practical_significance': self._assess_practical_significance(),
            'recommendations': self._generate_recommendations()
        }

        self.analysis_results = analysis
        logger.info("Comprehensive analysis completed")

        return analysis

    def _calculate_basic_statistics(self) -> Dict[str, Any]:
        """Calculate basic descriptive statistics."""
        stats_summary = {
            'models': {},
            'benchmarks': {},
            'overall': {}
        }

        aggregated = self.results['aggregated_results']

        # Per-model statistics
        for model_name, model_results in aggregated.items():
            model_stats = {'benchmarks': {}}

            for benchmark, bench_results in model_results.items():
                if bench_results['runs_completed'] > 0:
                    model_stats['benchmarks'][benchmark] = {
                        'mean': bench_results['mean_accuracy'],
                        'std': bench_results['std_accuracy'],
                        'runs': bench_results['runs_completed'],
                        'ci_lower': bench_results['confidence_interval'][0],
                        'ci_upper': bench_results['confidence_interval'][1]
                    }

            # Calculate model-level aggregates
            all_scores = []
            for bench_stats in model_stats['benchmarks'].values():
                all_scores.append(bench_stats['mean'])

            if all_scores:
                model_stats['aggregate'] = {
                    'mean_of_means': np.mean(all_scores),
                    'std_of_means': np.std(all_scores, ddof=1) if len(all_scores) > 1 else 0.0,
                    'benchmarks_completed': len(all_scores)
                }

            stats_summary['models'][model_name] = model_stats

        # Per-benchmark statistics
        benchmarks = set()
        for model_results in aggregated.values():
            benchmarks.update(model_results.keys())

        for benchmark in benchmarks:
            benchmark_scores = []
            for model_results in aggregated.values():
                if benchmark in model_results and model_results[benchmark]['runs_completed'] > 0:
                    benchmark_scores.append(model_results[benchmark]['mean_accuracy'])

            if benchmark_scores:
                stats_summary['benchmarks'][benchmark] = {
                    'scores': benchmark_scores,
                    'mean': np.mean(benchmark_scores),
                    'std': np.std(benchmark_scores, ddof=1) if len(benchmark_scores) > 1 else 0.0,
                    'min': np.min(benchmark_scores),
                    'max': np.max(benchmark_scores),
                    'range': np.max(benchmark_scores) - np.min(benchmark_scores)
                }

        return stats_summary

    def _perform_statistical_tests(self) -> Dict[str, Any]:
        """Perform comprehensive statistical tests."""
        statistical_tests = {
            'pairwise_t_tests': [],
            'anova_results': {},
            'normality_tests': {},
            'homoscedasticity_tests': {}
        }

        aggregated = self.results['aggregated_results']

        # Pairwise t-tests for all model pairs on each benchmark
        model_names = list(aggregated.keys())
        benchmarks = set()
        for model_results in aggregated.values():
            benchmarks.update(model_results.keys())

        for benchmark in benchmarks:
            benchmark_data = {}
            for model_name in model_names:
                if benchmark in aggregated[model_name]:
                    runs = aggregated[model_name][benchmark]['accuracies']
                    if runs:
                        benchmark_data[model_name] = runs

            if len(benchmark_data) >= 2:
                # Perform pairwise t-tests
                pairs = [(model_names[i], model_names[j])
                        for i in range(len(model_names))
                        for j in range(i+1, len(model_names))]

                for model_a, model_b in pairs:
                    if model_a in benchmark_data and model_b in benchmark_data:
                        data_a = benchmark_data[model_a]
                        data_b = benchmark_data[model_b]

                        if len(data_a) >= 2 and len(data_b) >= 2:
                            # Welch's t-test (unequal variances)
                            t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)

                            statistical_tests['pairwise_t_tests'].append({
                                'benchmark': benchmark,
                                'model_a': model_a,
                                'model_b': model_b,
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < self.results['metadata']['significance_level'],
                                'mean_a': np.mean(data_a),
                                'mean_b': np.mean(data_b),
                                'effect_size': self._calculate_cohen_d(data_a, data_b)
                            })

                # One-way ANOVA if we have 3+ models with data
                if len(benchmark_data) >= 3:
                    all_data = []
                    group_labels = []
                    for model, data in benchmark_data.items():
                        all_data.extend(data)
                        group_labels.extend([model] * len(data))

                    if len(set(group_labels)) >= 3:
                        f_stat, p_value = stats.f_oneway(*[benchmark_data[model] for model in benchmark_data.keys()
                                                         if len(benchmark_data[model]) >= 2])

                        statistical_tests['anova_results'][benchmark] = {
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < self.results['metadata']['significance_level']
                        }

        return statistical_tests

    def _calculate_cohen_d(self, data_a: List[float], data_b: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        mean_a, mean_b = np.mean(data_a), np.mean(data_b)
        std_a, std_b = np.std(data_a, ddof=1), np.std(data_b, ddof=1)

        # Pooled standard deviation
        n_a, n_b = len(data_a), len(data_b)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))

        if pooled_std > 0:
            return (mean_a - mean_b) / pooled_std
        return 0.0

    def _calculate_effect_sizes(self) -> Dict[str, Any]:
        """Calculate various effect size measures."""
        effect_sizes = {
            'cohen_d_matrix': {},
            'hedges_g_matrix': {},
            'glass_delta': {},
            'interpretations': {}
        }

        aggregated = self.results['aggregated_results']
        model_names = list(aggregated.keys())

        # Calculate pairwise effect sizes
        for i, model_a in enumerate(model_names[:-1]):
            for model_b in model_names[i+1:]:
                pair_key = f"{model_a}_vs_{model_b}"

                effect_sizes['cohen_d_matrix'][pair_key] = {}
                effect_sizes['hedges_g_matrix'][pair_key] = {}

                for benchmark in self.results['metadata']['benchmarks']:
                    if (benchmark in aggregated[model_a] and
                        benchmark in aggregated[model_b]):

                        data_a = aggregated[model_a][benchmark]['accuracies']
                        data_b = aggregated[model_b][benchmark]['accuracies']

                        if data_a and data_b:
                            cohen_d = self._calculate_cohen_d(data_a, data_b)

                            # Hedges' g (bias-corrected Cohen's d)
                            n_a, n_b = len(data_a), len(data_b)
                            correction_factor = 1 - (3 / (4 * (n_a + n_b) - 9))
                            hedges_g = cohen_d * correction_factor if correction_factor > 0 else cohen_d

                            effect_sizes['cohen_d_matrix'][pair_key][benchmark] = cohen_d
                            effect_sizes['hedges_g_matrix'][pair_key][benchmark] = hedges_g

                            # Interpretation
                            abs_d = abs(cohen_d)
                            if abs_d < 0.2:
                                interpretation = "negligible"
                            elif abs_d < 0.5:
                                interpretation = "small"
                            elif abs_d < 0.8:
                                interpretation = "medium"
                            else:
                                interpretation = "large"

                            if pair_key not in effect_sizes['interpretations']:
                                effect_sizes['interpretations'][pair_key] = {}
                            effect_sizes['interpretations'][pair_key][benchmark] = interpretation

        return effect_sizes

    def _generate_rankings(self) -> Dict[str, Any]:
        """Generate comprehensive rankings across all dimensions."""
        rankings = {
            'per_benchmark': {},
            'aggregate_rankings': {},
            'consistency_analysis': {}
        }

        aggregated = self.results['aggregated_results']

        # Per-benchmark rankings
        for benchmark in self.results['metadata']['benchmarks']:
            benchmark_scores = {}

            for model_name, model_results in aggregated.items():
                if benchmark in model_results and model_results[benchmark]['runs_completed'] > 0:
                    benchmark_scores[model_name] = model_results[benchmark]['mean_accuracy']

            if benchmark_scores:
                # Sort by score (descending)
                sorted_scores = sorted(benchmark_scores.items(), key=lambda x: x[1], reverse=True)
                rankings['per_benchmark'][benchmark] = {
                    'ranking': [(model, score) for model, score in sorted_scores],
                    'winner': sorted_scores[0][0],
                    'score_range': sorted_scores[0][1] - sorted_scores[-1][1]
                }

        # Aggregate rankings (average across benchmarks)
        model_averages = {}
        model_scores_by_benchmark = {}

        for model_name in aggregated.keys():
            scores = []
            for benchmark in self.results['metadata']['benchmarks']:
                if benchmark in aggregated[model_name] and aggregated[model_name][benchmark]['runs_completed'] > 0:
                    scores.append(aggregated[model_name][benchmark]['mean_accuracy'])

            if scores:
                model_averages[model_name] = np.mean(scores)
                model_scores_by_benchmark[model_name] = scores

        if model_averages:
            sorted_averages = sorted(model_averages.items(), key=lambda x: x[1], reverse=True)
            rankings['aggregate_rankings'] = {
                'by_average_score': [(model, score) for model, score in sorted_averages],
                'overall_winner': sorted_averages[0][0]
            }

            # Consistency analysis (how stable rankings are across benchmarks)
            rankings['consistency_analysis'] = self._analyze_ranking_consistency(model_scores_by_benchmark)

        return rankings

    def _analyze_ranking_consistency(self, model_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze ranking consistency across benchmarks."""
        consistency = {
            'rank_stability': {},
            'performance_variance': {}
        }

        if len(model_scores) < 2:
            return consistency

        # Calculate rank stability for each model
        benchmark_rankings = {}

        for benchmark in self.results['metadata']['benchmarks']:
            benchmark_scores = {}
            for model, scores in model_scores.items():
                # Find score for this benchmark (assuming order matches benchmark order)
                benchmark_idx = self.results['metadata']['benchmarks'].index(benchmark)
                if benchmark_idx < len(scores):
                    benchmark_scores[model] = scores[benchmark_idx]

            if benchmark_scores:
                sorted_models = sorted(benchmark_scores.keys(), key=lambda x: benchmark_scores[x], reverse=True)
                benchmark_rankings[benchmark] = {model: rank for rank, model in enumerate(sorted_models, 1)}

        # Calculate consistency metrics
        for model in model_scores.keys():
            ranks = [benchmark_rankings.get(benchmark, {}).get(model) for benchmark in self.results['metadata']['benchmarks']]
            ranks = [r for r in ranks if r is not None]

            if len(ranks) > 1:
                consistency['rank_stability'][model] = {
                    'mean_rank': np.mean(ranks),
                    'rank_variance': np.var(ranks, ddof=1),
                    'rank_std': np.std(ranks, ddof=1),
                    'consistency_score': 1.0 / (1.0 + np.std(ranks, ddof=1))  # Higher = more consistent
                }

            # Performance variance
            scores = model_scores[model]
            if len(scores) > 1:
                consistency['performance_variance'][model] = {
                    'mean_score': np.mean(scores),
                    'score_variance': np.var(scores, ddof=1),
                    'score_std': np.std(scores, ddof=1),
                    'coefficient_of_variation': np.std(scores, ddof=1) / np.mean(scores) if np.mean(scores) > 0 else 0
                }

        return consistency

    def _analyze_confidence_intervals(self) -> Dict[str, Any]:
        """Analyze confidence intervals and their implications."""
        confidence_analysis = {
            'interval_widths': {},
            'overlaps': {},
            'statistical_separation': {}
        }

        aggregated = self.results['aggregated_results']

        # Analyze confidence interval widths
        for model_name, model_results in aggregated.items():
            for benchmark, bench_results in model_results.items():
                ci_lower, ci_upper = bench_results['confidence_interval']
                width = ci_upper - ci_lower

                if benchmark not in confidence_analysis['interval_widths']:
                    confidence_analysis['interval_widths'][benchmark] = {}

                confidence_analysis['interval_widths'][benchmark][model_name] = {
                    'width': width,
                    'relative_width': width / bench_results['mean_accuracy'] if bench_results['mean_accuracy'] > 0 else 0,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                }

        # Analyze overlaps between confidence intervals
        for benchmark in self.results['metadata']['benchmarks']:
            if benchmark in confidence_analysis['interval_widths']:
                model_cis = confidence_analysis['interval_widths'][benchmark]

                if len(model_cis) >= 2:
                    confidence_analysis['overlaps'][benchmark] = {}
                    confidence_analysis['statistical_separation'][benchmark] = {}

                    model_names = list(model_cis.keys())
                    for i, model_a in enumerate(model_names[:-1]):
                        for model_b in model_names[i+1:]:
                            ci_a = (model_cis[model_a]['ci_lower'], model_cis[model_a]['ci_upper'])
                            ci_b = (model_cis[model_b]['ci_lower'], model_cis[model_b]['ci_upper'])

                            # Check for overlap
                            overlap = not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])
                            separation = abs(ci_a[0] - ci_b[1]) if ci_a[0] > ci_b[1] else abs(ci_b[0] - ci_a[1]) if ci_b[0] > ci_a[1] else 0

                            confidence_analysis['overlaps'][benchmark][f"{model_a}_vs_{model_b}"] = overlap
                            confidence_analysis['statistical_separation'][benchmark][f"{model_a}_vs_{model_b}"] = separation

        return confidence_analysis

    def _assess_practical_significance(self) -> Dict[str, Any]:
        """Assess practical significance beyond statistical significance."""
        practical_significance = {
            'performance_gaps': {},
            'improvement_magnitudes': {},
            'benchmark_importance': {},
            'overall_assessment': {}
        }

        # Calculate practical performance gaps
        rankings = self.analysis_results.get('rankings', {})
        per_benchmark = rankings.get('per_benchmark', {})

        for benchmark, bench_data in per_benchmark.items():
            ranking = bench_data['ranking']
            if len(ranking) >= 2:
                best_score = ranking[0][1]
                worst_score = ranking[-1][1]
                gap = best_score - worst_score

                practical_significance['performance_gaps'][benchmark] = {
                    'absolute_gap': gap,
                    'relative_gap': gap / worst_score if worst_score > 0 else 0,
                    'practical_significance': 'large' if gap > 0.1 else 'medium' if gap > 0.05 else 'small'
                }

        # Assess improvement magnitudes
        statistical_tests = self.analysis_results.get('statistical_tests', {})
        pairwise_tests = statistical_tests.get('pairwise_t_tests', [])

        for test in pairwise_tests:
            if test['significant']:
                improvement = abs(test['mean_a'] - test['mean_b'])
                practical_significance['improvement_magnitudes'][f"{test['benchmark']}_{test['model_a']}_vs_{test['model_b']}"] = {
                    'absolute_improvement': improvement,
                    'relative_improvement': improvement / min(test['mean_a'], test['mean_b']) if min(test['mean_a'], test['mean_b']) > 0 else 0,
                    'practical_impact': 'high' if improvement > 0.05 else 'medium' if improvement > 0.02 else 'low'
                }

        # Overall assessment
        significant_improvements = len([imp for imp in practical_significance['improvement_magnitudes'].values() if imp['practical_impact'] in ['high', 'medium']])

        if significant_improvements > 0:
            practical_significance['overall_assessment'] = {
                'conclusion': 'practically_significant_improvements_found',
                'confidence': 'high' if significant_improvements >= 3 else 'medium',
                'recommendation': 'Consider deployment of better performing model(s)'
            }
        else:
            practical_significance['overall_assessment'] = {
                'conclusion': 'no_practically_significant_differences',
                'confidence': 'medium',
                'recommendation': 'Models are practically equivalent; choose based on other factors'
            }

        return practical_significance

    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate actionable recommendations based on analysis."""
        recommendations = {
            'model_selection': {},
            'benchmark_priorities': {},
            'future_testing': {},
            'implementation_notes': []
        }

        # Model selection recommendations
        rankings = self.analysis_results.get('rankings', {})
        aggregate = rankings.get('aggregate_rankings', {})

        if 'by_average_score' in aggregate and aggregate['by_average_score']:
            best_model = aggregate['by_average_score'][0][0]
            recommendations['model_selection'] = {
                'recommended_model': best_model,
                'reasoning': f"Top performer across {len(self.results['metadata']['benchmarks'])} benchmarks",
                'confidence': 'high'
            }

        # Benchmark priority recommendations
        basic_stats = self.analysis_results.get('basic_statistics', {})
        benchmark_stats = basic_stats.get('benchmarks', {})

        if benchmark_stats:
            # Sort benchmarks by score range (higher range = more discriminative)
            benchmark_ranges = [(b, stats['range']) for b, stats in benchmark_stats.items()]
            benchmark_ranges.sort(key=lambda x: x[1], reverse=True)

            recommendations['benchmark_priorities'] = {
                'most_discriminative': benchmark_ranges[0][0],
                'ranking': [b for b, _ in benchmark_ranges]
            }

        # Future testing recommendations
        practical = self.analysis_results.get('practical_significance', {})
        overall = practical.get('overall_assessment', {})

        if overall.get('conclusion') == 'practically_significant_improvements_found':
            recommendations['future_testing'] = {
                'recommended': 'Continue A/B/C testing with expanded benchmarks',
                'focus_areas': 'Larger sample sizes, additional benchmarks, domain-specific evaluation'
            }
        else:
            recommendations['future_testing'] = {
                'recommended': 'Shift focus to qualitative evaluation and user studies',
                'focus_areas': 'User experience, specific use cases, qualitative performance metrics'
            }

        # Implementation notes
        recommendations['implementation_notes'] = [
            "Ensure evaluation environments are identical across models",
            "Consider computational costs when selecting models for deployment",
            "Validate results on additional benchmarks not included in this test",
            "Monitor model performance in production environments",
            "Consider ensemble approaches if models have complementary strengths"
        ]

        return recommendations

    def generate_pdf_report(self, output_path: str):
        """Generate comprehensive PDF report (placeholder for future implementation)."""
        logger.info("PDF report generation not yet implemented")
        # Future: Implement PDF generation with matplotlib/seaborn plots

    def save_analysis_results(self, output_path: str):
        """Save complete analysis results."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Analysis results saved to: {output_file}")

    def print_analysis_summary(self):
        """Print comprehensive analysis summary."""
        if not self.analysis_results:
            self.perform_comprehensive_analysis()

        print("\n" + "="*100)
        print("A/B/C TEST COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*100)

        # Basic statistics summary
        basic = self.analysis_results.get('basic_statistics', {})
        if 'models' in basic:
            print(f"\nMODELS ANALYZED: {len(basic['models'])}")
            for model, stats in basic['models'].items():
                if 'aggregate' in stats:
                    agg = stats['aggregate']
                    print(".4f")

        # Rankings summary
        rankings = self.analysis_results.get('rankings', {})
        if 'aggregate_rankings' in rankings:
            agg = rankings['aggregate_rankings']
            if 'by_average_score' in agg:
                print("\nOVERALL RANKING:")
                for i, (model, score) in enumerate(agg['by_average_score'][:3], 1):
                    print(f"  {i}. {model}: {score:.4f}")

        # Statistical significance summary
        stats = self.analysis_results.get('statistical_tests', {})
        sig_tests = [t for t in stats.get('pairwise_t_tests', []) if t['significant']]
        print(f"\nSTATISTICAL SIGNIFICANCE: {len(sig_tests)} significant differences found")

        # Practical significance
        practical = self.analysis_results.get('practical_significance', {})
        overall = practical.get('overall_assessment', {})
        if overall:
            print(f"\nPRACTICAL SIGNIFICANCE: {overall.get('conclusion', 'unknown').replace('_', ' ').title()}")

        # Recommendations
        recs = self.analysis_results.get('recommendations', {})
        model_sel = recs.get('model_selection', {})
        if model_sel:
            print(f"\nRECOMMENDED MODEL: {model_sel.get('recommended_model', 'N/A')}")
            print(f"Reason: {model_sel.get('reasoning', 'N/A')}")

        print("\n" + "="*100)


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='A/B/C Test Results Analysis')
    parser.add_argument('--results-file', required=True, help='A/B/C test results JSON file')
    parser.add_argument('--output-dir', default='evaluation_results/analysis', help='Output directory')
    parser.add_argument('--generate-plots', action='store_true', help='Generate visualization plots')
    parser.add_argument('--create-pdf-report', action='store_true', help='Create PDF report')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ABCTestResultsAnalyzer(args.results_file)

    # Perform comprehensive analysis
    analysis_results = analyzer.perform_comprehensive_analysis()

    # Save analysis results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_file = output_dir / "comprehensive_analysis.json"
    analyzer.save_analysis_results(str(analysis_file))

    # Print summary
    analyzer.print_analysis_summary()

    # Generate plots if requested
    if args.generate_plots:
        # Placeholder for plot generation
        logger.info("Plot generation not yet implemented")

    # Generate PDF report if requested
    if args.create_pdf_report:
        pdf_file = output_dir / "abctest_analysis_report.pdf"
        analyzer.generate_pdf_report(str(pdf_file))

    logger.info(f"Analysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()