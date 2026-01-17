#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparative Model Evaluation Script

Evaluates multiple models simultaneously on standardized benchmarks
to enable direct comparison with anchor models.
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

from standardized_benchmark_evaluator import StandardizedBenchmarkEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComparativeEvaluator:
    """
    Evaluates multiple models on standardized benchmarks for comparison.
    """

    def __init__(self, models_config: Dict[str, Dict], max_workers: int = 2):
        """
        Initialize comparative evaluator.

        Args:
            models_config: Dict of model configs {model_name: config_dict}
            max_workers: Maximum parallel workers
        """
        self.models_config = models_config
        self.max_workers = max_workers
        self.results = {}

        logger.info(f"Initialized comparative evaluator with {len(models_config)} models")

    def run_comparative_evaluation(self, benchmarks: List[str],
                                 num_samples: Optional[Dict[str, int]] = None,
                                 temperature: float = 0.0) -> Dict[str, Any]:
        """
        Run comparative evaluation across all models.

        Args:
            benchmarks: List of benchmarks to evaluate
            num_samples: Sample counts per benchmark
            temperature: Generation temperature

        Returns:
            Dict with comparative results
        """
        logger.info(f"Starting comparative evaluation on benchmarks: {benchmarks}")

        if num_samples is None:
            num_samples = {bench: None for bench in benchmarks}

        # Evaluate models in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            for model_name, config in self.models_config.items():
                logger.info(f"Submitting evaluation for: {model_name}")
                future = executor.submit(
                    self._evaluate_single_model,
                    model_name,
                    config,
                    benchmarks,
                    num_samples,
                    temperature
                )
                futures[future] = model_name

            # Collect results
            for future in tqdm(as_completed(futures), total=len(futures), desc="Model Evaluations"):
                model_name = futures[future]
                try:
                    result = future.result()
                    self.results[model_name] = result
                    logger.info(f"Completed evaluation for: {model_name}")
                except Exception as e:
                    logger.error(f"Evaluation failed for {model_name}: {e}")
                    self.results[model_name] = {'error': str(e)}

        # Generate comparative analysis
        comparative_results = self._generate_comparative_analysis()

        return comparative_results

    def _evaluate_single_model(self, model_name: str, config: Dict,
                              benchmarks: List[str], num_samples: Dict[str, int],
                              temperature: float) -> Dict[str, Any]:
        """
        Evaluate a single model on specified benchmarks.
        """
        try:
            logger.info(f"Evaluating {model_name} on {benchmarks}")

            # Initialize evaluator
            evaluator = StandardizedBenchmarkEvaluator(
                model_path=config['path'],
                model_name=model_name,
                device=config.get('device', 'auto')
            )

            # Run evaluation
            results = evaluator.run_comprehensive_evaluation({
                bench: num_samples.get(bench) for bench in benchmarks
                if bench in ['gsm8k', 'math', 'arc_challenge']
            })

            return results

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return {'error': str(e), 'model': model_name}

    def _generate_comparative_analysis(self) -> Dict[str, Any]:
        """
        Generate comparative analysis across all evaluated models.
        """
        analysis = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models_evaluated': list(self.results.keys()),
            'benchmark_comparison': {},
            'rankings': {},
            'summary': {}
        }

        # Extract benchmark results for each model
        benchmarks = ['gsm8k', 'math', 'arc_challenge']

        for benchmark in benchmarks:
            benchmark_results = {}

            for model_name, model_results in self.results.items():
                if 'error' not in model_results and benchmark in model_results:
                    accuracy = model_results[benchmark]['accuracy']
                    benchmark_results[model_name] = accuracy

            if benchmark_results:
                analysis['benchmark_comparison'][benchmark] = benchmark_results

                # Generate ranking for this benchmark
                sorted_models = sorted(benchmark_results.items(), key=lambda x: x[1], reverse=True)
                analysis['rankings'][benchmark] = [
                    {'rank': i+1, 'model': model, 'accuracy': acc}
                    for i, (model, acc) in enumerate(sorted_models)
                ]

        # Generate overall summary
        analysis['summary'] = self._generate_summary(analysis)

        return analysis

    def _generate_summary(self, analysis: Dict) -> Dict[str, Any]:
        """
        Generate overall summary of comparative evaluation.
        """
        summary = {
            'total_models': len(analysis['models_evaluated']),
            'benchmarks_evaluated': list(analysis['benchmark_comparison'].keys()),
            'top_performers': {},
            'model_profiles': {}
        }

        # Calculate average rankings
        model_avg_rankings = {}
        model_performances = {}

        for benchmark, rankings in analysis['rankings'].items():
            for rank_info in rankings:
                model = rank_info['model']
                rank = rank_info['rank']
                accuracy = rank_info['accuracy']

                if model not in model_avg_rankings:
                    model_avg_rankings[model] = []
                    model_performances[model] = []

                model_avg_rankings[model].append(rank)
                model_performances[model].append(accuracy)

        # Calculate averages
        for model in model_avg_rankings:
            avg_rank = np.mean(model_avg_rankings[model])
            avg_accuracy = np.mean(model_performances[model])

            summary['model_profiles'][model] = {
                'average_rank': avg_rank,
                'average_accuracy': avg_accuracy,
                'benchmarks_evaluated': len(model_avg_rankings[model])
            }

        # Find top performers
        if summary['model_profiles']:
            best_avg_accuracy = max(summary['model_profiles'].values(), key=lambda x: x['average_accuracy'])
            best_avg_rank = min(summary['model_profiles'].values(), key=lambda x: x['average_rank'])

            summary['top_performers'] = {
                'highest_accuracy': {
                    'model': [m for m, p in summary['model_profiles'].items() if p['average_accuracy'] == best_avg_accuracy['average_accuracy']][0],
                    'accuracy': best_avg_accuracy['average_accuracy']
                },
                'best_ranking': {
                    'model': [m for m, p in summary['model_profiles'].items() if p['average_rank'] == best_avg_rank['average_rank']][0],
                    'average_rank': best_avg_rank['average_rank']
                }
            }

        return summary

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save comparative results to JSON file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Comparative results saved to: {output_file}")

    def print_comparison_table(self, results: Dict[str, Any]):
        """
        Print formatted comparison table.
        """
        print("\n" + "="*80)
        print("COMPARATIVE MODEL EVALUATION RESULTS")
        print("="*80)

        print(f"Evaluation Date: {results['timestamp']}")
        print(f"Models Evaluated: {', '.join(results['models_evaluated'])}")
        print()

        # Print benchmark-wise rankings
        for benchmark, rankings in results['rankings'].items():
            print(f"{benchmark.upper()} RANKINGS:")
            print("-" * 40)
            for rank_info in rankings:
                print("2d")
            print()

        # Print summary
        summary = results['summary']
        print("OVERALL SUMMARY:")
        print("-" * 40)
        print(f"Total Models: {summary['total_models']}")
        print(f"Benchmarks: {', '.join(summary['benchmarks_evaluated'])}")

        if 'top_performers' in summary:
            tp = summary['top_performers']
            print(f"Top Accuracy: {tp['highest_accuracy']['model']} ({tp['highest_accuracy']['accuracy']:.4f})")
            print(".2f")

        print("\nMODEL PROFILES:")
        for model, profile in summary['model_profiles'].items():
            print(f"  {model}:")
            print(".2f")
            print(f"    Benchmarks: {profile['benchmarks_evaluated']}")


def main():
    """Main comparative evaluation function"""
    parser = argparse.ArgumentParser(description='Comparative Model Evaluation')
    parser.add_argument('--models_config', required=True, help='JSON file with model configurations')
    parser.add_argument('--benchmarks', nargs='+', default=['gsm8k', 'math', 'arc_challenge'],
                       help='Benchmarks to evaluate')
    parser.add_argument('--output_path', default='evaluation_results/comparative_evaluation_results.json',
                       help='Output path')
    parser.add_argument('--max_workers', type=int, default=2, help='Maximum parallel workers')
    parser.add_argument('--gsm8k_samples', type=int, help='Number of GSM8K samples')
    parser.add_argument('--math_samples', type=int, help='Number of MATH samples')
    parser.add_argument('--arc_samples', type=int, help='Number of ARC-Challenge samples')
    parser.add_argument('--temperature', type=float, default=0.0, help='Generation temperature')

    args = parser.parse_args()

    # Load model configurations
    with open(args.models_config, 'r', encoding='utf-8') as f:
        models_config = json.load(f)

    # Setup sample counts
    num_samples = {}
    if args.gsm8k_samples:
        num_samples['gsm8k'] = args.gsm8k_samples
    if args.math_samples:
        num_samples['math'] = args.math_samples
    if args.arc_samples:
        num_samples['arc_challenge'] = args.arc_samples

    # Initialize comparative evaluator
    evaluator = ComparativeEvaluator(models_config, max_workers=args.max_workers)

    # Run comparative evaluation
    results = evaluator.run_comparative_evaluation(
        benchmarks=args.benchmarks,
        num_samples=num_samples,
        temperature=args.temperature
    )

    # Save results
    evaluator.save_results(results, args.output_path)

    # Print comparison table
    evaluator.print_comparison_table(results)


if __name__ == "__main__":
    main()