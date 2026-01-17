#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Official Leaderboard Compliant A/B/C Test Plan Mode

Executes standardized A/B/C testing across Phi-3.5-mini-instruct,
Borea-phi3.5-instinct-jp, and AEGIS-Phi3.5mini-jp-v2.4 using official
benchmark protocols with statistical significance validation.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm
import time
import argparse
from scipy import stats
from datetime import datetime

from standardized_benchmark_evaluator import StandardizedBenchmarkEvaluator

# Try to import GGUF evaluator, fallback to None if not available
try:
    from gguf_benchmark_evaluator import GGUFStandardizedBenchmarkEvaluator
    GGUF_AVAILABLE = True
except ImportError:
    GGUFStandardizedBenchmarkEvaluator = None
    GGUF_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OfficialABCTestPlan:
    """
    Official leaderboard compliant A/B/C testing for SO8T models.
    """

    def __init__(self, max_workers: int = 2):
        """
        Initialize A/B/C test plan.

        Args:
            max_workers: Maximum parallel workers for evaluation
        """
        self.max_workers = max_workers
        self.models = {
            "Phi-3.5-mini-instruct": {
                "path": "microsoft/Phi-3.5-mini-instruct",
                "official_scores": {
                    "gsm8k": 86.2,  # 8-shot CoT
                    "math": 48.5,   # 0-shot CoT
                    "arc_challenge": 84.6  # 10-shot
                }
            },
            "Borea-phi3.5-instinct-jp": {
                "path": "path/to/borea/model",  # Update with actual path
                "baseline_comparison": True
            },
            "AEGIS-Phi3.5mini-jp-v2.4": {
                "path": "your-username/AEGIS-Phi3.5mini-jp",  # Update with actual path
                "so8t_enhanced": True
            }
        }

        logger.info("Initialized Official A/B/C Test Plan")
        logger.info(f"GGUF support: {'Available' if GGUF_AVAILABLE else 'Not available, using Transformers only'}")

    def execute_official_abctest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute official leaderboard compliant A/B/C testing.

        Args:
            config: Test configuration

        Returns:
            Complete test results with statistical analysis
        """
        logger.info("Starting Official A/B/C Test execution")

        # Update model paths if provided
        if 'models' in config:
            for model_name, model_config in config['models'].items():
                if model_name in self.models:
                    if isinstance(model_config, str):
                        # If model_config is a string path
                        self.models[model_name]['path'] = model_config
                    elif isinstance(model_config, dict) and 'path' in model_config:
                        # If model_config is a dict with path
                        self.models[model_name]['path'] = model_config['path']

        # Prepare evaluation configuration with extended timeouts
        eval_config = {
            'benchmarks': config.get('benchmarks', ['gsm8k', 'math', 'arc_challenge']),
            'sample_sizes': config.get('sample_sizes', {'gsm8k': 1000, 'math': 500, 'arc_challenge': 1000}),
            'runs_per_model': config.get('runs_per_model', 3),
            'temperature': config.get('temperature', 0.0),
            'timeouts': {
                'gsm8k': config.get('gsm8k_timeout', 120),      # 2 minutes per GSM8K sample
                'math': config.get('math_timeout', 300),        # 5 minutes per MATH sample (complex reasoning)
                'arc_challenge': config.get('arc_timeout', 180) # 3 minutes per ARC sample
            },
            'max_new_tokens': {
                'gsm8k': config.get('gsm8k_max_tokens', 512),
                'math': config.get('math_max_tokens', 1024),    # MATH needs longer responses
                'arc_challenge': config.get('arc_max_tokens', 256)
            }
        }

        logger.info(f"Extended timeout configuration: {eval_config['timeouts']}")

        # Execute evaluations
        raw_results = self._execute_parallel_evaluations(eval_config)

        # Perform statistical analysis
        statistical_results = self._perform_statistical_analysis(raw_results, config)

        # Generate final results
        final_results = self._generate_final_results(raw_results, statistical_results, config)

        logger.info("Official A/B/C Test completed successfully")
        return final_results

    def _execute_parallel_evaluations(self, eval_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute evaluations in parallel across all models and runs.
        """
        logger.info("Executing parallel evaluations")

        results = {}
        total_jobs = len(self.models) * len(eval_config['benchmarks']) * eval_config['runs_per_model']

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            # Submit all evaluation jobs
            for model_name, model_info in self.models.items():
                for benchmark in eval_config['benchmarks']:
                    for run_id in range(eval_config['runs_per_model']):
                        future = executor.submit(
                            self._evaluate_single_run,
                            model_name,
                            model_info['path'],  # Pass only the path string
                            benchmark,
                            eval_config['sample_sizes'].get(benchmark, 100),
                            eval_config['temperature'],
                            run_id,
                            eval_config
                        )
                        futures.append((future, model_name, benchmark, run_id))

            # Collect results with progress tracking
            completed = 0
            with tqdm(total=total_jobs, desc="A/B/C Test Evaluations") as pbar:
                for future, model_name, benchmark, run_id in futures:
                    try:
                        result = future.result()
                        if model_name not in results:
                            results[model_name] = {}
                        if benchmark not in results[model_name]:
                            results[model_name][benchmark] = {}

                        results[model_name][benchmark][f'run_{run_id}'] = result
                        completed += 1
                        pbar.update(1)

                    except Exception as e:
                        logger.error(f"Evaluation failed for {model_name} {benchmark} run {run_id}: {e}")
                        completed += 1
                        pbar.update(1)

        # Aggregate results
        aggregated_results = self._aggregate_results(results, eval_config)

        return aggregated_results

    def _evaluate_single_run(self, model_name: str, model_path: str,
                           benchmark: str, sample_size: int, temperature: float,
                           run_id: int, eval_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single model on a single benchmark run with extended timeouts.
        Supports both Transformers and GGUF models.
        """
        try:
            # Set random seed for reproducibility
            np.random.seed(42 + hash(model_name + benchmark) % 1000 + run_id)

            logger.info(f"Evaluating {model_name} on {benchmark} (run {run_id})")

            start_time = time.time()

            # Check if this is a GGUF model and GGUF is available
            is_gguf = self._is_gguf_model(model_path) and GGUF_AVAILABLE

            if is_gguf and GGUFStandardizedBenchmarkEvaluator:
                # Use GGUF evaluator
                evaluator = GGUFStandardizedBenchmarkEvaluator(
                    model_path=model_path,
                    model_name=f"{model_name}_run_{run_id}",
                    n_gpu_layers=eval_config.get('gguf_settings', {}).get('n_gpu_layers', -1)
                )

                # Execute evaluation based on benchmark
                if benchmark == 'gsm8k':
                    result = evaluator.evaluate_gsm8k(num_samples=sample_size, temperature=temperature)
                elif benchmark == 'math':
                    result = evaluator.evaluate_math(num_samples=sample_size, temperature=temperature)
                elif benchmark == 'arc_challenge':
                    result = evaluator.evaluate_arc_challenge(num_samples=sample_size, temperature=temperature)
                else:
                    raise ValueError(f"Unsupported benchmark: {benchmark}")

            else:
                # Use Transformers evaluator
                import torch
                torch.manual_seed(42 + hash(model_name + benchmark) % 1000 + run_id)

                evaluator = StandardizedBenchmarkEvaluator(
                    model_path=model_path,
                    model_name=f"{model_name}_run_{run_id}",
                    device="auto"
                )

                # Get timeout and token limits for this benchmark
                timeout = eval_config['timeouts'][benchmark]
                max_tokens = eval_config['max_new_tokens'][benchmark]

                # Update evaluator with extended settings
                evaluator.timeout = timeout
                evaluator.max_new_tokens = max_tokens

                # Execute evaluation based on benchmark
                if benchmark == 'gsm8k':
                    result = evaluator.evaluate_gsm8k(num_samples=sample_size, temperature=temperature)
                elif benchmark == 'math':
                    result = evaluator.evaluate_math(num_samples=sample_size, temperature=temperature)
                elif benchmark == 'arc_challenge':
                    result = evaluator.evaluate_arc_challenge(num_samples=sample_size, temperature=temperature)
                else:
                    raise ValueError(f"Unsupported benchmark: {benchmark}")

            elapsed_time = time.time() - start_time
            logger.info(".2f")

            # Add timing information
            result['timing'] = {
                'elapsed_seconds': elapsed_time,
                'samples_per_second': sample_size / elapsed_time if elapsed_time > 0 else 0,
                'model_format': 'GGUF' if is_gguf else 'Transformers'
            }

            return result

        except Exception as e:
            logger.error(f"Single run evaluation failed for {model_name} {benchmark}: {e}")
            return {
                'error': str(e),
                'accuracy': 0.0,
                'correct': 0,
                'total': sample_size,
                'individual_results': [],
                'metadata': {
                    'model': model_name,
                    'benchmark': benchmark,
                    'run_id': run_id,
                    'sample_size': sample_size,
                    'temperature': temperature
                },
                'timing': {
                    'elapsed_seconds': 0,
                    'error_occurred': True,
                    'model_format': self._detect_model_format(model_path)
                }
            }

    def _is_gguf_model(self, model_path: str) -> bool:
        """Check if the model path points to a GGUF file."""
        return str(model_path).lower().endswith('.gguf')

    def _detect_model_format(self, model_path: str) -> str:
        """Detect model format from path."""
        if self._is_gguf_model(model_path) and GGUF_AVAILABLE:
            return 'GGUF'
        else:
            return 'Transformers'

    def _aggregate_results(self, raw_results: Dict[str, Any], eval_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate results across multiple runs.
        """
        aggregated = {}

        for model_name, model_results in raw_results.items():
            aggregated[model_name] = {}

            for benchmark, benchmark_results in model_results.items():
                accuracies = []
                all_individual_results = []

                for run_key, run_result in benchmark_results.items():
                    if 'error' not in run_result:
                        accuracies.append(run_result['accuracy'])
                        all_individual_results.extend(run_result.get('individual_results', []))

                if accuracies:
                    aggregated[model_name][benchmark] = {
                        'accuracies': accuracies,
                        'mean_accuracy': np.mean(accuracies),
                        'std_accuracy': np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0,
                        'runs_completed': len(accuracies),
                        'individual_results': all_individual_results,
                        'confidence_interval': self._calculate_confidence_interval(accuracies)
                    }
                else:
                    aggregated[model_name][benchmark] = {
                        'accuracies': [],
                        'mean_accuracy': 0.0,
                        'std_accuracy': 0.0,
                        'runs_completed': 0,
                        'individual_results': [],
                        'confidence_interval': (0.0, 0.0)
                    }

        return aggregated

    def _calculate_confidence_interval(self, accuracies: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for accuracies.
        """
        if len(accuracies) < 2:
            mean_acc = np.mean(accuracies) if accuracies else 0.0
            return (mean_acc, mean_acc)

        mean_acc = np.mean(accuracies)
        std_err = stats.sem(accuracies)
        margin = std_err * stats.t.ppf((1 + confidence) / 2, len(accuracies) - 1)

        return (max(0.0, mean_acc - margin), min(1.0, mean_acc + margin))

    def _perform_statistical_analysis(self, aggregated_results: Dict[str, Any],
                                    config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis.
        """
        logger.info("Performing statistical analysis")

        significance_level = config.get('significance_level', 0.05)
        benchmarks = config.get('benchmarks', ['gsm8k', 'math', 'arc_challenge'])

        statistical_results = {
            'pairwise_comparisons': [],
            'overall_analysis': {},
            'significance_summary': {}
        }

        # Generate all pairwise comparisons
        model_names = list(aggregated_results.keys())
        for i, model_a in enumerate(model_names[:-1]):
            for model_b in model_names[i+1:]:
                for benchmark in benchmarks:
                    if (benchmark in aggregated_results[model_a] and
                        benchmark in aggregated_results[model_b]):

                        acc_a = aggregated_results[model_a][benchmark]['accuracies']
                        acc_b = aggregated_results[model_b][benchmark]['accuracies']

                        if acc_a and acc_b:
                            comparison = self._perform_pairwise_comparison(
                                model_a, model_b, benchmark, acc_a, acc_b, significance_level
                            )
                            statistical_results['pairwise_comparisons'].append(comparison)

        # Generate significance summary
        statistical_results['significance_summary'] = self._generate_significance_summary(
            statistical_results['pairwise_comparisons'], model_names, benchmarks
        )

        return statistical_results

    def _perform_pairwise_comparison(self, model_a: str, model_b: str, benchmark: str,
                                   accuracies_a: List[float], accuracies_b: List[float],
                                   alpha: float) -> Dict[str, Any]:
        """
        Perform pairwise statistical comparison between two models.
        """
        # Perform t-test
        try:
            t_stat, p_value = stats.ttest_ind(accuracies_a, accuracies_b, equal_var=False)
        except:
            t_stat, p_value = 0.0, 1.0

        # Calculate Cohen's d
        mean_a, mean_b = np.mean(accuracies_a), np.mean(accuracies_b)
        std_a, std_b = np.std(accuracies_a, ddof=1), np.std(accuracies_b, ddof=1)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        cohen_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        # Determine significance
        significant = p_value < alpha

        # Interpret effect size
        abs_d = abs(cohen_d)
        if abs_d < 0.2:
            effect_interpretation = "negligible"
        elif abs_d < 0.5:
            effect_interpretation = "small"
        elif abs_d < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        return {
            'model_a': model_a,
            'model_b': model_b,
            'benchmark': benchmark,
            'mean_a': mean_a,
            'mean_b': mean_b,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohen_d': cohen_d,
            'significant': significant,
            'effect_size_interpretation': effect_interpretation,
            'winner': model_a if mean_a > mean_b else model_b
        }

    def _generate_significance_summary(self, comparisons: List[Dict],
                                     model_names: List[str], benchmarks: List[str]) -> Dict[str, Any]:
        """
        Generate summary of statistical significance results.
        """
        summary = {
            'significant_improvements': {},
            'model_rankings': {},
            'benchmark_winners': {}
        }

        for benchmark in benchmarks:
            benchmark_comparisons = [c for c in comparisons if c['benchmark'] == benchmark]
            benchmark_scores = {}

            # Calculate average scores for ranking
            for model in model_names:
                model_comparisons = [c for c in benchmark_comparisons
                                   if c['model_a'] == model or c['model_b'] == model]

                if model_comparisons:
                    # Use the score from comparisons (assuming symmetric)
                    score = model_comparisons[0]['mean_a'] if model_comparisons[0]['model_a'] == model else model_comparisons[0]['mean_b']
                    benchmark_scores[model] = score

            # Rank models for this benchmark
            if benchmark_scores:
                ranked = sorted(benchmark_scores.items(), key=lambda x: x[1], reverse=True)
                summary['benchmark_winners'][benchmark] = ranked[0][0]
                summary['model_rankings'][benchmark] = [model for model, _ in ranked]

        return summary

    def _generate_final_results(self, aggregated_results: Dict[str, Any],
                              statistical_results: Dict[str, Any],
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final comprehensive results.
        """
        final_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_type': 'Official Leaderboard Compliant A/B/C Test',
                'models_tested': list(aggregated_results.keys()),
                'benchmarks': config.get('benchmarks', ['gsm8k', 'math', 'arc_challenge']),
                'runs_per_model': config.get('runs_per_model', 3),
                'significance_level': config.get('significance_level', 0.05),
                'evaluation_protocols': {
                    'gsm8k': '8-shot CoT',
                    'math': '0-shot CoT',
                    'arc_challenge': '10-shot'
                }
            },
            'aggregated_results': aggregated_results,
            'statistical_analysis': statistical_results,
            'summary': self._generate_overall_summary(aggregated_results, statistical_results)
        }

        return final_results

    def _generate_overall_summary(self, aggregated_results: Dict[str, Any],
                                statistical_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overall test summary.
        """
        summary = {
            'total_models': len(aggregated_results),
            'total_benchmarks': len(statistical_results['significance_summary']['benchmark_winners']),
            'benchmark_winners': statistical_results['significance_summary']['benchmark_winners'],
            'significant_findings': []
        }

        # Determine overall winner (most benchmark wins)
        benchmark_wins = {}
        for benchmark, winner in summary['benchmark_winners'].items():
            benchmark_wins[winner] = benchmark_wins.get(winner, 0) + 1

        if benchmark_wins:
            overall_winner = max(benchmark_wins.items(), key=lambda x: x[1])
            summary['overall_winner'] = {
                'model': overall_winner[0],
                'benchmark_wins': overall_winner[1]
            }

        # Generate key findings
        significant_comparisons = [c for c in statistical_results['pairwise_comparisons']
                                 if c['significant']]

        if significant_comparisons:
            summary['significant_findings'].append(
                f"Found {len(significant_comparisons)} statistically significant differences"
            )

        return summary

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save complete test results.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Official A/B/C Test results saved to: {output_file}")

    def print_summary(self, results: Dict[str, Any]):
        """
        Print test summary.
        """
        print("\n" + "="*80)
        print("OFFICIAL LEADERBOARD COMPLIANT A/B/C TEST RESULTS")
        print("="*80)

        meta = results['metadata']
        summary = results['summary']

        print(f"Test Date: {meta['timestamp']}")
        print(f"Models Tested: {', '.join(meta['models_tested'])}")
        print(f"Benchmarks: {', '.join(meta['benchmarks'])}")
        print(f"Runs per Model: {meta['runs_per_model']}")
        print()

        print("BENCHMARK WINNERS:")
        for benchmark, winner in summary['benchmark_winners'].items():
            print(f"  {benchmark.upper()}: {winner}")
        print()

        if 'overall_winner' in summary:
            ow = summary['overall_winner']
            print(f"Overall Winner: {ow['model']} ({ow['benchmark_wins']} benchmark wins)")
        print()

        print("KEY FINDINGS:")
        for finding in summary['significant_findings']:
            print(f"  • {finding}")


def main():
    """Main A/B/C test execution"""
    parser = argparse.ArgumentParser(description='Official Leaderboard Compliant A/B/C Test')
    parser.add_argument('--models-config', help='Models configuration JSON file')
    parser.add_argument('--benchmarks', nargs='+', default=['gsm8k', 'math', 'arc_challenge'])
    parser.add_argument('--sample-sizes', help='Sample sizes as benchmark:size pairs')
    parser.add_argument('--runs-per-model', type=int, default=3)
    parser.add_argument('--significance-level', type=float, default=0.05)
    parser.add_argument('--max-workers', type=int, default=2)
    parser.add_argument('--output-path', default='evaluation_results/official_abctest_results.json')
    parser.add_argument('--temperature', type=float, default=0.0)

    # Extended timeout and token settings for complex reasoning
    parser.add_argument('--gsm8k_timeout', type=int, default=120, help='Timeout for GSM8K evaluation (seconds)')
    parser.add_argument('--math_timeout', type=int, default=300, help='Timeout for MATH evaluation (seconds)')
    parser.add_argument('--arc_timeout', type=int, default=180, help='Timeout for ARC evaluation (seconds)')
    parser.add_argument('--gsm8k_max_tokens', type=int, default=512, help='Max tokens for GSM8K generation')
    parser.add_argument('--math_max_tokens', type=int, default=1024, help='Max tokens for MATH generation')
    parser.add_argument('--arc_max_tokens', type=int, default=256, help='Max tokens for ARC generation')

    args = parser.parse_args()

    # Load models config if provided
    models_config = None
    if args.models_config:
        with open(args.models_config, 'r', encoding='utf-8') as f:
            models_config = json.load(f)

    # Parse sample sizes
    sample_sizes = {'gsm8k': 100, 'math': 50, 'arc_challenge': 100}
    if args.sample_sizes:
        for pair in args.sample_sizes.split(','):
            benchmark, size = pair.split(':')
            sample_sizes[benchmark.strip()] = int(size.strip())

    # Initialize and execute test
    abc_test = OfficialABCTestPlan(max_workers=args.max_workers)

    config = {
        'benchmarks': args.benchmarks,
        'sample_sizes': sample_sizes,
        'runs_per_model': args.runs_per_model,
        'significance_level': args.significance_level,
        'temperature': args.temperature,
        'gsm8k_timeout': args.gsm8k_timeout,
        'math_timeout': args.math_timeout,
        'arc_timeout': args.arc_timeout,
        'gsm8k_max_tokens': args.gsm8k_max_tokens,
        'math_max_tokens': args.math_max_tokens,
        'arc_max_tokens': args.arc_max_tokens
    }

    if models_config and 'models' in models_config:
        config['models'] = models_config['models']  # This should be {model_name: path_string}

    results = abc_test.execute_official_abctest(config)

    # Save and display results
    abc_test.save_results(results, args.output_path)
    abc_test.print_summary(results)


if __name__ == "__main__":
    main()