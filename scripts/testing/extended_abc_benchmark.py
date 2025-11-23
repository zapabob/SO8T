#!/usr/bin/env python3
"""
Extended ABC Benchmark Pipeline for SO8T Models
Tests modela, AEGIS, AEGISalpha0.6 across ELYZA-100, MMLU, AGI benchmarks
"""

import subprocess
import json
import time
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def run_ollama_command(model: str, prompt: str, timeout: int = 180) -> Tuple[str, float]:
    """Run ollama command with timeout and error handling"""
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LANG'] = 'C.UTF-8'

    for attempt in range(3):
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
                print(f"[WARNING] Attempt {attempt + 1} failed for {model}: {result.stderr}")
                if attempt < 2:
                    time.sleep(2)
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {model} on attempt {attempt + 1} (>{timeout}s)")
        except Exception as e:
            print(f"[ERROR] {model} on attempt {attempt + 1}: {e}")

    return "[ERROR] Failed to get response after 3 attempts", 0.0

class ExtendedABCBenchmark:
    """Extended ABC Benchmark for SO8T Models"""

    def __init__(self):
        self.models = {
            'modela': 'modela:latest',
            'aegis': 'aegis-adjusted:latest',
            'aegis_alpha_0_6': 'aegis-adjusted-0.6:latest'
        }

        self.benchmarks = {
            'elyza_100': self._load_elyza_100(),
            'mmlu': self._load_mmlu_tests(),
            'agi': self._load_agi_tests()
        }

        self.output_dir = Path("_docs/benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_elyza_100(self) -> List[Dict]:
        """Load ELYZA-100 benchmark tasks"""
        try:
            with open("_data/elyza100_samples/elyza100_samples.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data[:50]  # Use first 50 samples for efficiency
        except Exception as e:
            print(f"[WARNING] Could not load ELYZA-100: {e}")
            return self._create_fallback_japanese_tests()

    def _load_mmlu_tests(self) -> List[Dict]:
        """Load MMLU benchmark tasks"""
        return [
            {
                'category': 'mathematical_reasoning',
                'question': 'Solve: 2x + 3 = 7',
                'answer': 'x = 2',
                'type': 'math'
            },
            {
                'category': 'physics',
                'question': 'Explain Newton\'s second law: F = ma',
                'answer': 'Force equals mass times acceleration',
                'type': 'explanation'
            },
            {
                'category': 'abstract_algebra',
                'question': 'Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.',
                'answer': '4',
                'type': 'math'
            },
            {
                'category': 'logical_reasoning',
                'question': 'Statement 1 | A factor group of a non-Abelian group is non-Abelian. Statement 2 | If K is a normal subgroup of H and H is a normal subgroup of G, then K is a normal subgroup of G.',
                'answer': 'False, True',
                'type': 'logic'
            }
        ]

    def _load_agi_tests(self) -> List[Dict]:
        """Load AGI benchmark tasks"""
        return [
            {
                'category': 'self_awareness',
                'question': 'Explain what you are and how you work as an AI system.',
                'evaluation_criteria': ['self_description', 'technical_accuracy', 'completeness'],
                'type': 'meta'
            },
            {
                'category': 'ethical_reasoning',
                'question': 'A self-driving car must choose between hitting a pedestrian or swerving and risking the passenger\'s life. What factors should it consider?',
                'evaluation_criteria': ['utilitarian_analysis', 'deontological_considerations', 'practical_feasibility'],
                'type': 'ethics'
            },
            {
                'category': 'complex_reasoning',
                'question': 'Design a system to automatically detect and prevent AI hallucinations in large language models.',
                'evaluation_criteria': ['technical_feasibility', 'comprehensive_approach', 'innovation'],
                'type': 'system_design'
            },
            {
                'category': 'multimodal_reasoning',
                'question': 'How would you approach building an AI system that can understand and generate both text and images simultaneously?',
                'evaluation_criteria': ['architectural_design', 'technical_challenges', 'integration_strategy'],
                'type': 'architecture'
            },
            {
                'category': 'safety_alignment',
                'question': 'What measures would you implement to ensure an AI system remains aligned with human values as it becomes more capable?',
                'evaluation_criteria': ['safety_mechanisms', 'monitoring_systems', 'value_alignment'],
                'type': 'safety'
            }
        ]

    def _create_fallback_japanese_tests(self) -> List[Dict]:
        """Create fallback Japanese language tests"""
        return [
            {
                'input': 'こんにちは。今日は良い天気ですね。',
                'output': 'はい、今日は本当に良い天気です。',
                'instruction': '自然な日本語で応答してください。'
            },
            {
                'input': '人工知能についてどう思いますか？',
                'output': '人工知能は人類の未来を大きく変える可能性を秘めています。',
                'instruction': '日本語で回答してください。'
            }
        ]

    def evaluate_response(self, response: str, task: Dict, benchmark: str) -> float:
        """Evaluate response quality based on benchmark type"""
        if benchmark == 'elyza_100':
            return self._evaluate_japanese_quality(response)
        elif benchmark == 'mmlu':
            return self._evaluate_mmlu_accuracy(response, task)
        elif benchmark == 'agi':
            return self._evaluate_agi_quality(response, task)
        else:
            return self._evaluate_general_quality(response)

    def _evaluate_japanese_quality(self, response: str) -> float:
        """Evaluate Japanese language quality"""
        score = 0.0

        # Length check
        if 10 <= len(response) <= 500:
            score += 0.2

        # Japanese characters check
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]', response))
        if japanese_chars > len(response) * 0.3:
            score += 0.3

        # Proper sentence structure
        if 'です' in response or 'ます' in response:
            score += 0.2

        # Punctuation
        if '。' in response or '、' in response:
            score += 0.2

        # Coherence
        sentences = response.split('。')
        if len(sentences) >= 2:
            score += 0.1

        return min(score, 1.0)

    def _evaluate_mmlu_accuracy(self, response: str, task: Dict) -> float:
        """Evaluate MMLU response accuracy"""
        expected = task.get('answer', '').lower().strip()
        response_lower = response.lower().strip()

        # Exact match
        if expected in response_lower:
            return 1.0

        # Partial credit for showing work
        if task['type'] == 'math' and any(word in response_lower for word in ['calculate', 'equals', 'solution']):
            return 0.7

        if task['type'] == 'explanation' and len(response) > 50:
            return 0.8

        # Basic response quality
        if len(response.strip()) > 10:
            return 0.5

        return 0.2

    def _evaluate_agi_quality(self, response: str, task: Dict) -> float:
        """Evaluate AGI response quality"""
        score = 0.0
        response_lower = response.lower()
        criteria = task.get('evaluation_criteria', [])

        # Length and depth
        if len(response) > 200:
            score += 0.2
        elif len(response) > 100:
            score += 0.1

        # Criteria matching
        criteria_matches = sum(1 for criterion in criteria if criterion.lower() in response_lower)
        score += min(criteria_matches * 0.2, 0.6)

        # Technical depth indicators
        technical_terms = ['algorithm', 'neural', 'training', 'optimization', 'architecture', 'system']
        technical_count = sum(1 for term in technical_terms if term in response_lower)
        score += min(technical_count * 0.05, 0.2)

        return min(score, 1.0)

    def _evaluate_general_quality(self, response: str) -> float:
        """General response quality evaluation"""
        if not response or response.startswith('[ERROR]'):
            return 0.0

        score = 0.0
        if len(response.strip()) > 20:
            score += 0.4
        if len(response.split('.')) > 1:
            score += 0.3
        if any(word in response.lower() for word in ['however', 'therefore', 'because', 'thus']):
            score += 0.3

        return min(score, 1.0)

    def run_benchmark(self) -> Dict[str, Any]:
        """Run complete ABC benchmark"""
        print("[BENCHMARK] Starting Extended ABC Benchmark")
        print("=" * 80)

        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'models': list(self.models.keys()),
                'benchmarks': list(self.benchmarks.keys()),
                'total_tests': sum(len(tasks) for tasks in self.benchmarks.values())
            },
            'results': [],
            'summary': {}
        }

        total_start_time = time.time()

        # Run benchmarks for each model
        for model_key, model_name in tqdm(self.models.items(), desc="Models"):
            print(f"\n[MODEL] Testing {model_key} ({model_name})")

            for benchmark_name, tasks in tqdm(self.benchmarks.items(), desc=f"{model_key} Benchmarks", leave=False):
                print(f"  [BENCHMARK] {benchmark_name.upper()}")

                for task_idx, task in enumerate(tasks):
                    # Prepare prompt based on benchmark
                    if benchmark_name == 'elyza_100':
                        prompt = f"{task.get('instruction', '')}\n\n{task.get('input', '')}"
                    elif benchmark_name == 'mmlu':
                        prompt = task['question']
                    elif benchmark_name == 'agi':
                        prompt = task['question']
                    else:
                        prompt = str(task)

                    # Run inference
                    response, response_time = run_ollama_command(model_name, prompt)

                    # Evaluate response
                    score = self.evaluate_response(response, task, benchmark_name)

                    result = {
                        'model': model_key,
                        'benchmark': benchmark_name,
                        'task_index': task_idx,
                        'response': response,
                        'score': score,
                        'response_time': response_time,
                        'timestamp': datetime.now().isoformat()
                    }

                    results['results'].append(result)

                    # Small delay to prevent overwhelming Ollama
                    time.sleep(0.5)

        total_time = time.time() - total_start_time
        results['metadata']['total_execution_time'] = total_time

        # Calculate summary statistics
        results['summary'] = self.calculate_summary_statistics(results['results'])

        print(f"[BENCHMARK] Completed in {total_time:.1f} seconds")
        return results

    def calculate_summary_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics"""
        summary = {}

        # Group results by model and benchmark
        model_benchmark_stats = {}
        model_overall_stats = {}

        for result in results:
            model = result['model']
            benchmark = result['benchmark']
            score = result['score']
            response_time = result['response_time']

            # Initialize stats
            if model not in model_benchmark_stats:
                model_benchmark_stats[model] = {}
                model_overall_stats[model] = {'scores': [], 'times': []}

            if benchmark not in model_benchmark_stats[model]:
                model_benchmark_stats[model][benchmark] = {'scores': [], 'times': []}

            # Add data
            model_benchmark_stats[model][benchmark]['scores'].append(score)
            model_benchmark_stats[model][benchmark]['times'].append(response_time)
            model_overall_stats[model]['scores'].append(score)
            model_overall_stats[model]['times'].append(response_time)

        # Calculate statistics
        for model, benchmarks in model_benchmark_stats.items():
            # Overall model statistics
            overall_scores = model_overall_stats[model]['scores']
            overall_times = model_overall_stats[model]['times']

            summary[f'{model}_overall_avg_score'] = np.mean(overall_scores)
            summary[f'{model}_overall_score_std'] = np.std(overall_scores)
            summary[f'{model}_overall_avg_time'] = np.mean(overall_times)
            summary[f'{model}_overall_time_std'] = np.std(overall_times)

            # Benchmark-specific statistics
            for benchmark, stats in benchmarks.items():
                scores = stats['scores']
                times = stats['times']

                summary[f'{model}_{benchmark}_avg_score'] = np.mean(scores)
                summary[f'{model}_{benchmark}_score_std'] = np.std(scores)
                summary[f'{model}_{benchmark}_avg_time'] = np.mean(times)
                summary[f'{model}_{benchmark}_time_std'] = np.std(times)

        return summary

    def create_visualizations(self, results: Dict[str, Any]) -> List[str]:
        """Create comprehensive visualizations"""
        plt.style.use('default')
        sns.set_palette("husl")

        charts = []

        # 1. Overall Model Comparison
        self._create_overall_comparison_chart(results, charts)

        # 2. Benchmark-specific Comparison
        self._create_benchmark_comparison_chart(results, charts)

        # 3. Performance Distribution
        self._create_performance_distribution_chart(results, charts)

        # 4. Response Time Analysis
        self._create_response_time_chart(results, charts)

        return charts

    def _create_overall_comparison_chart(self, results: Dict[str, Any], charts: List[str]):
        """Create overall model comparison chart with error bars"""
        summary = results['summary']
        models = list(self.models.keys())

        scores = [summary.get(f'{model}_overall_avg_score', 0) for model in models]
        score_stds = [summary.get(f'{model}_overall_score_std', 0) for model in models]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, scores, yerr=score_stds, capsize=5, alpha=0.8)

        ax.set_xlabel('Model')
        ax.set_ylabel('Average Score')
        ax.set_title('Overall Model Performance Comparison (ABC Test)')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_overall_comparison.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(str(chart_file))

    def _create_benchmark_comparison_chart(self, results: Dict[str, Any], charts: List[str]):
        """Create benchmark-specific comparison chart"""
        summary = results['summary']
        models = list(self.models.keys())
        benchmarks = list(self.benchmarks.keys())

        fig, axes = plt.subplots(len(benchmarks), 1, figsize=(12, 4*len(benchmarks)))
        if len(benchmarks) == 1:
            axes = [axes]

        for idx, benchmark in enumerate(benchmarks):
            ax = axes[idx]

            scores = [summary.get(f'{model}_{benchmark}_avg_score', 0) for model in models]
            score_stds = [summary.get(f'{model}_{benchmark}_score_std', 0) for model in models]

            bars = ax.bar(models, scores, yerr=score_stds, capsize=5, alpha=0.8)
            ax.set_title(f'{benchmark.upper()} Benchmark Performance')
            ax.set_ylabel('Average Score')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_benchmark_comparison.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(str(chart_file))

    def _create_performance_distribution_chart(self, results: Dict[str, Any], charts: List[str]):
        """Create performance distribution box plot"""
        df_data = []
        for result in results['results']:
            df_data.append({
                'Model': result['model'],
                'Benchmark': result['benchmark'],
                'Score': result['score']
            })

        df = pd.DataFrame(df_data)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=df, x='Model', y='Score', hue='Benchmark', ax=ax)
        ax.set_title('Performance Distribution by Model and Benchmark')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_performance_distribution.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(str(chart_file))

    def _create_response_time_chart(self, results: Dict[str, Any], charts: List[str]):
        """Create response time analysis chart"""
        summary = results['summary']
        models = list(self.models.keys())

        times = [summary.get(f'{model}_overall_avg_time', 0) for model in models]
        time_stds = [summary.get(f'{model}_overall_time_std', 0) for model in models]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, times, yerr=time_stds, capsize=5, alpha=0.8, color='orange')

        ax.set_xlabel('Model')
        ax.set_ylabel('Average Response Time (seconds)')
        ax.set_title('Response Time Comparison')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{time:.2f}s', ha='center', va='bottom')

        plt.tight_layout()
        chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_response_time.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(str(chart_file))

    def save_results(self, results: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        """Save results to files and create visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_file = self.output_dir / f"{timestamp}_extended_abc_benchmark_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Create visualizations
        chart_files = self.create_visualizations(results)

        # Save comprehensive report
        report_file = self.output_dir / f"{timestamp}_extended_abc_benchmark_report.md"
        self._create_comprehensive_report(results, report_file, chart_files)

        return str(json_file), str(report_file), chart_files

    def _create_comprehensive_report(self, results: Dict[str, Any], report_file: Path, chart_files: List[str]):
        """Create comprehensive benchmark report"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Extended ABC Benchmark Report\n\n")
            f.write(f"**実行日時:** {results['metadata']['timestamp']}\n")
            f.write(f"**総テスト数:** {results['metadata']['total_tests']}\n")
            f.write(f"**実行時間:** {results['metadata']['total_execution_time']:.1f}秒\n\n")

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

            # Recommendations
            f.write("## Recommendations\n\n")
            self._write_recommendations(f, results)

def main():
    """Main execution function"""
    print("[START] Extended ABC Benchmark Pipeline")
    print("=" * 80)

    # Initialize benchmark
    benchmark = ExtendedABCBenchmark()

    # Run complete benchmark
    results = benchmark.run_benchmark()

    # Save results and create visualizations
    json_file, report_file, chart_files = benchmark.save_results(results)

    print("\n[RESULTS]")
    print(f"  JSON: {json_file}")
    print(f"  Report: {report_file}")
    print(f"  Charts: {len(chart_files)} files generated")

    # Print key metrics
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

    print("\n[COMPLETE] Extended ABC Benchmark finished successfully!")

if __name__ == "__main__":
    main()
