#!/usr/bin/env python3
"""
Automated LLM Benchmark Pipeline for A/B Testing
Tests Model A vs AEGIS across multiple categories with comprehensive evaluation
"""

import subprocess
import json
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any
import re

def run_ollama_command(model: str, prompt: str, timeout: int = 120) -> str:
    """Run ollama command with retry logic and timeout"""
    # Set environment to use UTF-8
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LANG'] = 'C.UTF-8'

    for attempt in range(3):
        try:
            result = subprocess.run(
                ['ollama', 'run', model, prompt],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # Handle encoding errors
                timeout=timeout,
                env=env
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"[WARNING] Attempt {attempt + 1} failed for {model}: {result.stderr}")
                if attempt < 2:
                    time.sleep(3)
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {model} on attempt {attempt + 1} (>{timeout}s)")
        except Exception as e:
            print(f"[ERROR] {model} on attempt {attempt + 1}: {e}")

    return "[ERROR] Failed to get response after 3 attempts"

def evaluate_mathematical_accuracy(response: str, expected_answer: str) -> float:
    """Evaluate mathematical answer accuracy (0.0 to 1.0)"""
    response_lower = response.lower().strip()
    expected_lower = expected_answer.lower().strip()

    # Extract numbers from response
    numbers = re.findall(r'\d+(?:\.\d+)?', response)
    expected_numbers = re.findall(r'\d+(?:\.\d+)?', expected_answer)

    if not numbers and not expected_numbers:
        return 0.3 if len(response.strip()) > 10 else 0.1

    if not expected_numbers:
        return 0.5 if numbers else 0.2

    # Check if any expected number appears in response
    for expected_num in expected_numbers:
        for num in numbers:
            if abs(float(num) - float(expected_num)) < 0.01:  # Allow small floating point differences
                return 1.0

    # Partial credit for showing work
    if any(word in response_lower for word in ['calculate', 'compute', 'step', 'method']):
        return 0.6

    return 0.2

def evaluate_logical_reasoning(response: str, expected_patterns: List[str]) -> float:
    """Evaluate logical reasoning quality (0.0 to 1.0)"""
    response_lower = response.lower()
    score = 0.0

    # Check for logical indicators
    logical_indicators = ['therefore', 'because', 'if', 'then', 'conclusion', 'premise', 'valid', 'invalid']
    logic_count = sum(1 for indicator in logical_indicators if indicator in response_lower)
    score += min(logic_count * 0.2, 0.4)

    # Check for expected reasoning patterns
    pattern_matches = sum(1 for pattern in expected_patterns if pattern.lower() in response_lower)
    score += min(pattern_matches * 0.3, 0.6)

    # Length and coherence bonus
    if len(response) > 100:
        score += 0.2
    if len(response.split('.')) > 3:  # Multiple sentences
        score += 0.1

    return min(score, 1.0)

def evaluate_scientific_accuracy(response: str, key_concepts: List[str]) -> float:
    """Evaluate scientific content accuracy (0.0 to 1.0)"""
    response_lower = response.lower()
    score = 0.0

    # Check for key scientific concepts
    concept_matches = sum(1 for concept in key_concepts if concept.lower() in response_lower)
    score += min(concept_matches * 0.4, 0.8)

    # Check for scientific method indicators
    method_indicators = ['hypothesis', 'experiment', 'data', 'analysis', 'conclusion', 'evidence']
    method_count = sum(1 for indicator in method_indicators if indicator in response_lower)
    score += min(method_count * 0.1, 0.2)

    return min(score, 1.0)

def evaluate_language_quality(response: str, language: str = 'ja') -> float:
    """Evaluate language quality and fluency (0.0 to 1.0)"""
    score = 0.0

    if language == 'ja':
        # Japanese quality indicators
        if len(response) > 50:
            score += 0.3
        if 'です' in response or 'ます' in response:
            score += 0.3
        if '、' in response or '。' in response:
            score += 0.2
        if len(response.split('。')) > 2:
            score += 0.2
    else:
        # English quality indicators
        if len(response) > 50:
            score += 0.3
        if any(word in response.lower() for word in ['however', 'therefore', 'moreover', 'consequently']):
            score += 0.3
        if ',' in response and '.' in response:
            score += 0.2
        if len(response.split('.')) > 2:
            score += 0.2

    return min(score, 1.0)

def evaluate_security_ethics(response: str, ethical_requirements: List[str]) -> float:
    """Evaluate security and ethical reasoning (0.0 to 1.0)"""
    response_lower = response.lower()
    score = 0.0

    # Check for ethical considerations
    ethics_matches = sum(1 for req in ethical_requirements if req.lower() in response_lower)
    score += min(ethics_matches * 0.3, 0.6)

    # Check for security awareness
    security_indicators = ['security', 'privacy', 'risk', 'vulnerability', 'protection', 'compliance']
    security_count = sum(1 for indicator in security_indicators if indicator in response_lower)
    score += min(security_count * 0.15, 0.4)

    return min(score, 1.0)

def evaluate_medical_financial(response: str, domain_requirements: List[str]) -> float:
    """Evaluate medical/financial information management (0.0 to 1.0)"""
    response_lower = response.lower()
    score = 0.0

    # Check for domain-specific requirements
    requirement_matches = sum(1 for req in domain_requirements if req.lower() in response_lower)
    score += min(requirement_matches * 0.25, 0.5)

    # Check for professional indicators
    professional_indicators = ['accuracy', 'compliance', 'regulation', 'standard', 'protocol', 'guideline']
    professional_count = sum(1 for indicator in professional_indicators if indicator in response_lower)
    score += min(professional_count * 0.15, 0.3)

    # Check for error handling
    error_indicators = ['verification', 'validation', 'double-check', 'confirmation', 'audit']
    error_count = sum(1 for indicator in error_indicators if indicator in response_lower)
    score += min(error_count * 0.2, 0.2)

    return min(score, 1.0)

class LLMBenchmarkPipeline:
    """Automated LLM Benchmark Pipeline for A/B Testing"""

    def __init__(self):
        self.models = {
            'model_a': 'model-a:q8_0',
            'aegis': 'aegis-borea-phi35-instinct-jp:q8_0'
        }

        self.benchmark_categories = {
            'mathematical_reasoning': {
                'name': 'Mathematical & Logical Reasoning',
                'description': 'Mathematical and logical reasoning capabilities',
                'tests': [
                    {
                        'name': 'Basic Algebra',
                        'prompt': 'Solve: 2x + 3 = 7',
                        'expected_answer': 'x = 2',
                        'evaluation_type': 'mathematical'
                    }
                ]
            },
            'scientific_knowledge': {
                'name': 'Scientific & Technical Knowledge',
                'description': 'Scientific and technical knowledge understanding',
                'tests': [
                    {
                        'name': 'Physics Concept',
                        'prompt': 'Explain Newton\'s second law: F = ma',
                        'key_concepts': ['force', 'mass', 'acceleration'],
                        'evaluation_type': 'scientific'
                    }
                ]
            },
            'japanese_language': {
                'name': 'Japanese Language Understanding',
                'description': 'Japanese language understanding and generation',
                'tests': [
                    {
                        'name': 'Basic Japanese',
                        'prompt': 'Translate: "Hello, how are you?" to Japanese',
                        'language': 'ja',
                        'evaluation_type': 'language'
                    }
                ]
            },
            'security_ethics': {
                'name': 'Security & Ethical Reasoning',
                'description': 'Security awareness and ethical reasoning',
                'tests': [
                    {
                        'name': 'Privacy Ethics',
                        'prompt': 'Why is user privacy important in software?',
                        'ethical_requirements': ['privacy', 'security', 'consent'],
                        'evaluation_type': 'security'
                    }
                ]
            },
            'medical_financial': {
                'name': 'Medical & Financial Information',
                'description': 'Medical and financial information management',
                'tests': [
                    {
                        'name': 'Basic Healthcare',
                        'prompt': 'What are common symptoms of the flu?',
                        'domain_requirements': ['fever', 'cough', 'fatigue'],
                        'evaluation_type': 'medical_financial'
                    }
                ]
            },
            'general_knowledge': {
                'name': 'General Knowledge & Commonsense',
                'description': 'General knowledge and commonsense reasoning',
                'tests': [
                    {
                        'name': 'Basic Knowledge',
                        'prompt': 'What is the capital of France?',
                        'evaluation_type': 'general'
                    }
                ]
            }
        }

    def run_single_test(self, model: str, category: str, test: Dict) -> Dict[str, Any]:
        """Run a single test case"""
        print(f"[TEST] {category} - {test['name']} on {model}")

        start_time = time.time()

        # Prepare prompt based on model
        if model == 'agiasi':
            prompt = f"{test['prompt']}\n\n[LOGIC] Logical Accuracy\n[ETHICS] Ethical Validity\n[PRACTICAL] Practical Value\n[CREATIVE] Creative Insight\n\n[FINAL] Final Evaluation"
        else:
            prompt = test['prompt']

        response = run_ollama_command(model, prompt)
        response_time = time.time() - start_time

        # Evaluate response based on test type
        evaluation_type = test.get('evaluation_type', 'general')

        if evaluation_type == 'mathematical':
            score = evaluate_mathematical_accuracy(response, test.get('expected_answer', ''))
        elif evaluation_type == 'scientific':
            score = evaluate_scientific_accuracy(response, test.get('key_concepts', []))
        elif evaluation_type == 'language':
            score = evaluate_language_quality(response, test.get('language', 'ja'))
        elif evaluation_type == 'security':
            score = evaluate_security_ethics(response, test.get('ethical_requirements', []))
        elif evaluation_type == 'medical_financial':
            score = evaluate_medical_financial(response, test.get('domain_requirements', []))
        else:
            # General evaluation based on response quality
            score = min(len(response) / 500.0, 1.0)  # Basic length-based scoring

        return {
            'model': model,
            'category': category,
            'test_name': test['name'],
            'response': response,
            'score': score,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        print("[BENCHMARK] Starting Automated LLM Benchmark Pipeline")
        print("=" * 80)

        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'models': list(self.models.keys()),
                'categories': list(self.benchmark_categories.keys()),
                'total_tests': sum(len(cat['tests']) for cat in self.benchmark_categories.values())
            },
            'results': [],
            'summary': {}
        }

        total_start_time = time.time()

        for model_key, model_name in self.models.items():
            print(f"\n[MODEL] Testing {model_key} ({model_name})")

            for category_key, category in self.benchmark_categories.items():
                print(f"  [CATEGORY] {category['name']}")

                for test in category['tests']:
                    result = self.run_single_test(model_key, category_key, test)
                    results['results'].append(result)

                    # Small delay to prevent overwhelming Ollama
                    time.sleep(0.5)

        total_time = time.time() - total_start_time
        results['metadata']['total_execution_time'] = total_time

        # Calculate summary statistics
        results['summary'] = self.calculate_summary_statistics(results['results'])

        print(f"\n[BENCHMARK] Completed in {total_time:.1f} seconds")
        print(f"[RESULTS] {len(results['results'])} test cases executed")

        return results

    def calculate_summary_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from results"""
        summary = {}

        # Group by model and category
        model_stats = {}
        category_stats = {}

        for result in results:
            model = result['model']
            category = result['category']

            if model not in model_stats:
                model_stats[model] = {'scores': [], 'times': []}
            if category not in category_stats:
                category_stats[category] = {'model_scores': {}}

            model_stats[model]['scores'].append(result['score'])
            model_stats[model]['times'].append(result['response_time'])

            if model not in category_stats[category]['model_scores']:
                category_stats[category]['model_scores'][model] = []
            category_stats[category]['model_scores'][model].append(result['score'])

        # Calculate averages
        for model, stats in model_stats.items():
            summary[f'{model}_avg_score'] = np.mean(stats['scores'])
            summary[f'{model}_avg_time'] = np.mean(stats['times'])
            summary[f'{model}_score_std'] = np.std(stats['scores'])

        for category, stats in category_stats.items():
            summary[f'{category}_model_a_avg'] = np.mean(stats['model_scores'].get('model_a', [0]))
            summary[f'{category}_agiasi_avg'] = np.mean(stats['model_scores'].get('agiasi', [0]))

        return summary

    def save_results(self, results: Dict[str, Any], output_dir: str = "_docs/benchmark_results"):
        """Save results to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_file = os.path.join(output_dir, f"{timestamp}_automated_benchmark_results.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save text report
        report_file = os.path.join(output_dir, f"{timestamp}_automated_benchmark_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Automated LLM Benchmark A/B Test Report\n\n")
            f.write(f"**実行日時:** {results['metadata']['timestamp']}\n")
            f.write(f"**総テスト数:** {results['metadata']['total_tests']}\n")
            f.write(f"**実行時間:** {results['metadata']['total_execution_time']:.1f}秒\n\n")

            f.write("## モデルの比較\n\n")

            summary = results['summary']
            f.write("| 指標 | Model A | AEGIS | 差異 |\n")
            f.write("|------|---------|--------|------|\n")
            f.write(f"| 平均スコア | {summary.get('model_a_avg_score', 0):.3f} | {summary.get('agiasi_avg_score', 0):.3f} | {summary.get('agiasi_avg_score', 0) - summary.get('model_a_avg_score', 0):+.3f} |\n")
            f.write(f"| 平均応答時間 | {summary.get('model_a_avg_time', 0):.2f}s | {summary.get('agiasi_avg_time', 0):.2f}s | {summary.get('agiasi_avg_time', 0) - summary.get('model_a_avg_time', 0):+.2f}s |\n")
            f.write(f"| スコア標準偏差 | {summary.get('model_a_score_std', 0):.3f} | {summary.get('agiasi_score_std', 0):.3f} | - |\n\n")

            f.write("## カテゴリ別比較\n\n")

            categories = self.benchmark_categories.keys()
            f.write("| カテゴリ | Model A | AEGIS | 勝者 |\n")
            f.write("|----------|---------|--------|------|\n")

            for category in categories:
                model_a_score = summary.get(f'{category}_model_a_avg', 0)
                agiasi_score = summary.get(f'{category}_agiasi_avg', 0)
                winner = "AEGIS" if agiasi_score > model_a_score else "Model A" if model_a_score > agiasi_score else "引き分け"
                f.write(f"| {self.benchmark_categories[category]['name']} | {model_a_score:.3f} | {agiasi_score:.3f} | {winner} |\n")

            f.write("\n## 結論\n\n")

            avg_score_diff = summary.get('agiasi_avg_score', 0) - summary.get('model_a_avg_score', 0)
            if avg_score_diff > 0.1:
                f.write("**AEGISが優位**: 総合的な性能でModel Aを上回っています。\n\n")
            elif avg_score_diff < -0.1:
                f.write("**Model Aが優位**: 総合的な性能でAEGISを上回っています。\n\n")
            else:
                f.write("**拮抗**: 両モデルの性能に大きな差異はありません。\n\n")

            f.write("**詳細結果:** 個別のテストケース結果はJSONファイルで確認してください。\n")

        return json_file, report_file

    def create_visualizations(self, results: Dict[str, Any], output_dir: str = "_docs/benchmark_results"):
        """Create visualization charts"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        summary = results['summary']

        # Category comparison chart
        categories = list(self.benchmark_categories.keys())
        category_names = [self.benchmark_categories[cat]['name'] for cat in categories]

        model_a_scores = [summary.get(f'{cat}_model_a_avg', 0) for cat in categories]
        agiasi_scores = [summary.get(f'{cat}_agiasi_avg', 0) for cat in categories]

        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, model_a_scores, width, label='Model A', alpha=0.8)
        bars2 = ax.bar(x + width/2, agiasi_scores, width, label='AEGIS', alpha=0.8)

        ax.set_xlabel('カテゴリ')
        ax.set_ylabel('平均スコア')
        ax.set_title('LLMベンチマーク A/Bテスト カテゴリ別比較')
        ax.set_xticks(x)
        ax.set_xticklabels(category_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        chart_file = os.path.join(output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_benchmark_comparison.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()

        return chart_file

def main():
    """Main execution function"""
    print("[START] Automated LLM Benchmark Pipeline")
    print("=" * 80)

    pipeline = LLMBenchmarkPipeline()

    # Run full benchmark
    results = pipeline.run_full_benchmark()

    # Save results
    json_file, report_file = pipeline.save_results(results)

    # Create visualizations
    chart_file = pipeline.create_visualizations(results)

    print("\n[RESULTS]")
    print(f"  JSON: {json_file}")
    print(f"  Report: {report_file}")
    print(f"  Chart: {chart_file}")

    # Print key findings
    summary = results['summary']
    model_a_avg = summary.get('model_a_avg_score', 0)
    agiasi_avg = summary.get('agiasi_avg_score', 0)

    print("\n[SUMMARY]")
    print(f"  Model A Average Score: {model_a_avg:.3f}")
    print(f"  AEGIS Average Score: {agiasi_avg:.3f}")
    print(f"  Difference: {agiasi_avg - model_a_avg:+.3f}")

    if agiasi_avg > model_a_avg + 0.05:
        print("  Winner: AEGIS ✨")
    elif model_a_avg > agiasi_avg + 0.05:
        print("  Winner: Model A")
    else:
        print("  Result: Very Close Competition")

    # Play completion sound
    try:
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", "scripts/utils/play_audio_notification.ps1"
        ])
    except:
        pass

    print("\n[BENCHMARK COMPLETE] SUCCESS")

if __name__ == "__main__":
    main()
