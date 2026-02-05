#!/usr/bin/env python3
"""
Simple ABC Benchmark for SO8T Models
Tests modela, AEGIS, AEGISalpha0.6 across basic benchmarks
"""

import subprocess
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

def run_ollama_command(model: str, prompt: str, timeout: int = 180) -> Tuple[str, float]:
    """Run ollama command with timeout"""
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    try:
        start_time = time.time()
        result = subprocess.run(
            ['ollama', 'run', model, prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            env=env
        )
        response_time = time.time() - start_time

        if result.returncode == 0:
            return result.stdout.strip(), response_time
        else:
            return f"[ERROR] {result.stderr}", 0.0
    except Exception as e:
        return f"[ERROR] {e}", 0.0

class SimpleABCBenchmark:
    """Simple ABC Benchmark for SO8T Models"""

    def __init__(self):
        self.models = {
            'modela': 'modela:latest',
            'aegis': 'aegis-adjusted:latest',
            'aegis_alpha_0_6': 'aegis-adjusted-0.6:latest'
        }

        self.benchmarks = {
            'elyza_100': [
                {
                    'question': 'こんにちは。今日は良い天気ですね。何かお手伝いできることはありますか？',
                    'expected_type': 'conversation'
                },
                {
                    'question': '日本の首都はどこですか？',
                    'expected_type': 'factual'
                },
                {
                    'question': '2 + 2 × 3 = ? 計算してください。',
                    'expected_type': 'math'
                }
            ],
            'mmlu': [
                {
                    'question': 'Solve: 2x + 3 = 7',
                    'answer': 'x = 2',
                    'category': 'math'
                },
                {
                    'question': 'What is the capital of France?',
                    'answer': 'Paris',
                    'category': 'geography'
                },
                {
                    'question': 'Explain Newton\'s second law: F = ma',
                    'answer': 'Force equals mass times acceleration',
                    'category': 'physics'
                }
            ],
            'agi': [
                {
                    'question': 'What is your purpose as an AI?',
                    'category': 'self_awareness'
                },
                {
                    'question': 'Why is user privacy important?',
                    'category': 'ethics'
                },
                {
                    'question': 'How would you detect AI hallucinations?',
                    'category': 'technical'
                }
            ]
        }

        self.output_dir = Path("_docs/benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_response(self, response: str, task: Dict) -> float:
        """Simple response evaluation"""
        if not response or response.startswith('[ERROR]'):
            return 0.0

        score = 0.0

        # Basic quality checks
        if len(response.strip()) > 10:
            score += 0.4

        if len(response.split('.')) > 1:
            score += 0.3

        # Content relevance (simple keyword matching)
        if 'question' in task:
            question_lower = task['question'].lower()
            response_lower = response.lower()

            # Simple relevance check
            if any(word in response_lower for word in question_lower.split()[:3]):
                score += 0.3

        return min(score, 1.0)

    def run_benchmark(self) -> Dict[str, Any]:
        """Run complete ABC benchmark"""
        print("[BENCHMARK] Starting Simple ABC Benchmark")
        print("=" * 80)

        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'models': list(self.models.keys()),
                'benchmarks': list(self.benchmarks.keys()),
            },
            'results': [],
            'summary': {}
        }

        total_start_time = time.time()

        for model_key, model_name in tqdm(self.models.items(), desc="Models"):
            print(f"\n[MODEL] Testing {model_key}")

            for benchmark_name, tasks in tqdm(self.benchmarks.items(), desc=f"{model_key} Benchmarks", leave=False):
                print(f"  [BENCHMARK] {benchmark_name.upper()}")

                for task_idx, task in enumerate(tasks):
                    prompt = task['question']

                    # Run inference
                    response, response_time = run_ollama_command(model_name, prompt)

                    # Evaluate response
                    score = self.evaluate_response(response, task)

                    result = {
                        'model': model_key,
                        'benchmark': benchmark_name,
                        'task_index': task_idx,
                        'question': task['question'],
                        'response': response,
                        'score': score,
                        'response_time': response_time,
                        'timestamp': datetime.now().isoformat()
                    }

                    results['results'].append(result)

                    # Small delay
                    time.sleep(0.5)

        total_time = time.time() - total_start_time
        results['metadata']['total_execution_time'] = total_time

        # Calculate summary statistics
        results['summary'] = self.calculate_summary_statistics(results['results'])

        print(f"[BENCHMARK] Completed in {total_time:.1f} seconds")
        return results

    def calculate_summary_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        summary = {}

        # Group by model and benchmark
        model_benchmark_stats = {}
        model_overall_stats = {}

        for result in results:
            model = result['model']
            benchmark = result['benchmark']
            score = result['score']
            response_time = result['response_time']

            if model not in model_benchmark_stats:
                model_benchmark_stats[model] = {}
                model_overall_stats[model] = {'scores': [], 'times': []}

            if benchmark not in model_benchmark_stats[model]:
                model_benchmark_stats[model][benchmark] = {'scores': [], 'times': []}

            model_benchmark_stats[model][benchmark]['scores'].append(score)
            model_benchmark_stats[model][benchmark]['times'].append(response_time)
            model_overall_stats[model]['scores'].append(score)
            model_overall_stats[model]['times'].append(response_time)

        # Calculate averages
        for model, benchmarks in model_benchmark_stats.items():
            overall_scores = model_overall_stats[model]['scores']
            overall_times = model_overall_stats[model]['times']

            summary[f'{model}_overall_avg_score'] = np.mean(overall_scores)
            summary[f'{model}_overall_score_std'] = np.std(overall_scores)
            summary[f'{model}_overall_avg_time'] = np.mean(overall_times)
            summary[f'{model}_overall_time_std'] = np.std(overall_times)

            for benchmark, stats in benchmarks.items():
                scores = stats['scores']
                times = stats['times']

                summary[f'{model}_{benchmark}_avg_score'] = np.mean(scores)
                summary[f'{model}_{benchmark}_score_std'] = np.std(scores)
                summary[f'{model}_{benchmark}_avg_time'] = np.mean(times)
                summary[f'{model}_{benchmark}_time_std'] = np.std(times)

        return summary

    def create_visualizations(self, results: Dict[str, Any]) -> List[str]:
        """Create visualizations"""
        plt.style.use('default')
        sns.set_palette("husl")

        charts = []

        # Overall comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        models = list(self.models.keys())
        scores = [results['summary'].get(f'{model}_overall_avg_score', 0) for model in models]
        score_stds = [results['summary'].get(f'{model}_overall_score_std', 0) for model in models]

        bars = ax.bar(models, scores, yerr=score_stds, capsize=5, alpha=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel('Average Score')
        ax.set_title('ABC Benchmark Overall Performance Comparison')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_abc_overall.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(str(chart_file))

        # Benchmark-specific comparison
        benchmarks = list(self.benchmarks.keys())
        fig, axes = plt.subplots(len(benchmarks), 1, figsize=(12, 4*len(benchmarks)))
        if len(benchmarks) == 1:
            axes = [axes]

        for idx, benchmark in enumerate(benchmarks):
            ax = axes[idx]
            scores = [results['summary'].get(f'{model}_{benchmark}_avg_score', 0) for model in models]
            score_stds = [results['summary'].get(f'{model}_{benchmark}_score_std', 0) for model in models]

            bars = ax.bar(models, scores, yerr=score_stds, capsize=5, alpha=0.8)
            ax.set_title(f'{benchmark.upper()} Benchmark Performance')
            ax.set_ylabel('Average Score')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_abc_benchmarks.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(str(chart_file))

        return charts

    def save_results(self, results: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        """Save results and create report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_file = self.output_dir / f"{timestamp}_simple_abc_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Create visualizations
        chart_files = self.create_visualizations(results)

        # Create report
        report_file = self.output_dir / f"{timestamp}_simple_abc_report.md"
        self._create_report(results, report_file, chart_files)

        return str(json_file), str(report_file), chart_files

    def _create_report(self, results: Dict[str, Any], report_file: Path, chart_files: List[str]):
        """Create comprehensive report"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# ABC Benchmark Report (modela vs AEGIS vs AEGISalpha0.6)\n\n")
            f.write(f"**実行日時:** {results['metadata']['timestamp']}\n")
            f.write(f"**総実行時間:** {results['metadata']['total_execution_time']:.1f}秒\n\n")

            # Overall Performance
            f.write("## Overall Performance\n\n")
            summary = results['summary']
            models = list(self.models.keys())

            f.write("| Model | Average Score | Score Std | Average Time | Time Std |\n")
            f.write("|-------|---------------|-----------|--------------|----------|\n")

            for model in models:
                avg_score = summary.get(f'{model}_overall_avg_score', 0)
                score_std = summary.get(f'{model}_overall_score_std', 0)
                avg_time = summary.get(f'{model}_overall_avg_time', 0)
                time_std = summary.get(f'{model}_overall_time_std', 0)

                f.write(f"| {model} | {avg_score:.3f} | {score_std:.3f} | {avg_time:.2f}s | {time_std:.2f}s |\n")

            f.write("\n")

            # Benchmark-specific Performance
            f.write("## Benchmark-specific Performance\n\n")
            benchmarks = list(self.benchmarks.keys())

            for benchmark in benchmarks:
                f.write(f"### {benchmark.upper()}\n\n")
                f.write("| Model | Average Score | Score Std | Average Time | Time Std |\n")
                f.write("|-------|---------------|-----------|--------------|----------|\n")

                for model in models:
                    avg_score = summary.get(f'{model}_{benchmark}_avg_score', 0)
                    score_std = summary.get(f'{model}_{benchmark}_score_std', 0)
                    avg_time = summary.get(f'{model}_{benchmark}_avg_time', 0)
                    time_std = summary.get(f'{model}_{benchmark}_time_std', 0)

                    f.write(f"| {model} | {avg_score:.3f} | {score_std:.3f} | {avg_time:.2f}s | {time_std:.2f}s |\n")

                f.write("\n")

            # Statistical Analysis
            f.write("## Statistical Analysis\n\n")
            self._write_statistical_analysis(f, results)

            # Visualizations
            f.write("## Visualizations\n\n")
            for chart_file in chart_files:
                chart_name = Path(chart_file).name
                f.write(f"![{chart_name}]({chart_name})\n\n")

            # Key Findings
            f.write("## Key Findings\n\n")
            self._write_key_findings(f, results)

    def _write_statistical_analysis(self, f, results: Dict[str, Any]):
        """Write statistical analysis"""
        summary = results['summary']
        models = list(self.models.keys())

        # Calculate rankings
        model_scores = [(model, summary.get(f'{model}_overall_avg_score', 0)) for model in models]
        model_scores.sort(key=lambda x: x[1], reverse=True)

        f.write("### Performance Ranking\n")
        for i, (model, score) in enumerate(model_scores, 1):
            f.write(f"{i}. **{model}**: {score:.3f}\n")
        f.write("\n")

        # Best and worst performers
        best_model = model_scores[0][0]
        worst_model = model_scores[-1][0]

        f.write(f"**Best Overall Performer:** {best_model}\n")
        f.write(f"**Performance Gap:** {model_scores[0][1] - model_scores[-1][1]:.3f}\n\n")

    def _write_key_findings(self, f, results: Dict[str, Any]):
        """Write key findings"""
        summary = results['summary']
        models = list(self.models.keys())

        # Find best model per benchmark
        benchmarks = list(self.benchmarks.keys())

        f.write("### Benchmark Winners\n\n")
        for benchmark in benchmarks:
            benchmark_scores = [(model, summary.get(f'{model}_{benchmark}_avg_score', 0)) for model in models]
            benchmark_scores.sort(key=lambda x: x[1], reverse=True)
            winner = benchmark_scores[0][0]
            score = benchmark_scores[0][1]
            f.write(f"- **{benchmark.upper()}**: {winner} ({score:.3f})\n")

        f.write("\n### Performance Insights\n\n")
        f.write("- Analysis of relative strengths and weaknesses across different benchmarks\n")
        f.write("- Identification of optimal use cases for each model\n")
        f.write("- Recommendations for model selection based on specific requirements\n\n")

def main():
    """Main execution function"""
    print("[START] Simple ABC Benchmark Pipeline")
    print("=" * 80)

    benchmark = SimpleABCBenchmark()
    results = benchmark.run_benchmark()

    # Save results and create report
    json_file, report_file, chart_files = benchmark.save_results(results)

    print("\n[RESULTS]")
    print(f"  JSON: {json_file}")
    print(f"  Report: {report_file}")
    print(f"  Charts: {len(chart_files)} files")

    # Print summary
    summary = results['summary']
    models = list(benchmark.models.keys())

    print("\n[SUMMARY]")
    for model in models:
        avg_score = summary.get(f'{model}_overall_avg_score', 0)
        avg_time = summary.get(f'{model}_overall_avg_time', 0)
        print(f"  {model}: Score={avg_score:.3f}, Time={avg_time:.2f}s")

    # Determine winner
    model_scores = [(model, summary.get(f'{model}_overall_avg_score', 0)) for model in models]
    model_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n[WINNER] {model_scores[0][0]} (Score: {model_scores[0][1]:.3f})")

    # Play completion sound
    try:
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", "scripts/utils/play_audio_notification.ps1"
        ])
    except:
        pass

    print("\n[COMPLETE] Simple ABC Benchmark finished successfully!")

if __name__ == "__main__":
    main()
