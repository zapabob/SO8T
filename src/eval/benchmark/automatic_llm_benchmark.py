#!/usr/bin/env python3
"""
Automatic LLM Benchmark Pipeline for Model A/B Testing
Tests various LLM benchmarks using Ollama API directly
"""

import json
import time
import requests
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

OLLAMA_API = "http://localhost:11434/api/generate"

def run_ollama_query(model, prompt, max_retries=3):
    """Run query against Ollama model with retry logic"""
    for attempt in range(max_retries):
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent results
                    "num_predict": 1024,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }

            start_time = time.time()
            response = requests.post(OLLAMA_API, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            end_time = time.time()

            return {
                "response": data.get("response", "").strip(),
                "total_duration": data.get("total_duration", 0),
                "eval_count": data.get("eval_count", 0),
                "eval_duration": data.get("eval_duration", 0),
                "wall_time": end_time - start_time,
                "tokens_per_sec": data.get("eval_count", 0) / max(data.get("eval_duration", 1) / 1e9, 1e-6)
            }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {model}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return None

def load_mmlu_benchmark():
    """Load MMLU benchmark questions"""
    print("Loading MMLU benchmark...")
    try:
        # Load a subset of MMLU tasks for testing
        tasks = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology",
            "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies",
            "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_disputes", "moral_scenarios",
            "nutrition", "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology", "us_foreign_policy",
            "virology", "world_religions"
        ]

        benchmark_data = []
        for task in tasks[:5]:  # Test with first 5 tasks for speed
            try:
                dataset = load_dataset("cais/mmlu", task, split="test")
                for i, item in enumerate(dataset):
                    if i >= 20:  # Limit to 20 questions per task for testing
                        break

                    question = item['question']
                    choices = [item['choices'][j] for j in range(4)]
                    answer = item['answer']

                    benchmark_data.append({
                        "task": f"mmlu_{task}",
                        "type": "multiple_choice",
                        "question": question,
                        "choices": choices,
                        "answer": answer,
                        "fewshot_examples": []
                    })
            except Exception as e:
                print(f"Failed to load MMLU task {task}: {e}")
                continue

        return benchmark_data

    except Exception as e:
        print(f"Failed to load MMLU: {e}")
        return []

def load_gsm8k_benchmark():
    """Load GSM8K math reasoning benchmark"""
    print("Loading GSM8K benchmark...")
    try:
        dataset = load_dataset("gsm8k", "main", split="test")
        benchmark_data = []

        for i, item in enumerate(dataset):
            if i >= 50:  # Limit for testing
                break

            benchmark_data.append({
                "task": "gsm8k",
                "type": "math_reasoning",
                "question": item['question'],
                "answer": item['answer'],
                "fewshot_examples": []
            })

        return benchmark_data

    except Exception as e:
        print(f"Failed to load GSM8K: {e}")
        return []

def load_truthfulqa_benchmark():
    """Load TruthfulQA benchmark"""
    print("Loading TruthfulQA benchmark...")
    try:
        dataset = load_dataset("truthful_qa", "generation", split="validation")
        benchmark_data = []

        for i, item in enumerate(dataset):
            if i >= 30:  # Limit for testing
                break

            benchmark_data.append({
                "task": "truthfulqa",
                "type": "generation",
                "question": item['question'],
                "answer": item['answer'],
                "fewshot_examples": []
            })

        return benchmark_data

    except Exception as e:
        print(f"Failed to load TruthfulQA: {e}")
        return []

def format_mmlu_question(item, fewshot=True):
    """Format MMLU question with fewshot examples"""
    prompt = ""

    if fewshot and item.get("fewshot_examples"):
        for example in item["fewshot_examples"][:3]:  # Use 3 fewshot examples
            prompt += f"Question: {example['question']}\n"
            for j, choice in enumerate(example['choices']):
                prompt += f"{chr(65+j)}. {choice}\n"
            prompt += f"Answer: {chr(65+example['answer'])}\n\n"

    prompt += f"Question: {item['question']}\n"
    for j, choice in enumerate(item['choices']):
        prompt += f"{chr(65+j)}. {choice}\n"
    prompt += "Answer:"

    return prompt

def format_math_question(item, fewshot=True):
    """Format math reasoning question"""
    prompt = ""

    if fewshot and item.get("fewshot_examples"):
        for example in item["fewshot_examples"][:2]:  # Use 2 fewshot examples
            prompt += f"Question: {example['question']}\n"
            prompt += f"Answer: {example['answer']}\n\n"

    prompt += f"Question: {item['question']}\n"
    prompt += "Answer: Let's solve this step by step."

    return prompt

def evaluate_mmlu_response(response, correct_answer):
    """Evaluate MMLU response"""
    response = response.strip().upper()

    # Extract answer choice from response
    for char in response[:100]:  # Check first 100 chars
        if char in 'ABCD':
            return char == chr(65 + correct_answer)

    return False

def evaluate_math_response(response, correct_answer):
    """Evaluate math response by checking if final answer matches"""
    # Simple evaluation - check if the final answer appears in response
    # More sophisticated evaluation would parse the mathematical expression
    response = response.strip()
    correct_answer_clean = correct_answer.split('####')[-1].strip() if '####' in correct_answer else correct_answer.strip()

    # Look for the final answer in the response
    if correct_answer_clean in response:
        return True

    # Try to extract box notation or final answer
    import re
    box_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if box_match:
        return box_match.group(1).strip() == correct_answer_clean

    return False

def run_benchmark(models, benchmark_data, output_dir):
    """Run benchmark for all models"""
    results = {}

    for model in models:
        print(f"\n{'='*50}")
        print(f"Testing model: {model}")
        print(f"{'='*50}")

        model_results = []

        for item in tqdm(benchmark_data, desc=f"Testing {model}"):
            prompt = ""
            if item['type'] == 'multiple_choice':
                prompt = format_mmlu_question(item)
            elif item['type'] == 'math_reasoning':
                prompt = format_math_question(item)
            elif item['type'] == 'generation':
                prompt = item['question']

            result = run_ollama_query(model, prompt)

            if result:
                # Evaluate response
                if item['type'] == 'multiple_choice':
                    is_correct = evaluate_mmlu_response(result['response'], item['answer'])
                elif item['type'] == 'math_reasoning':
                    is_correct = evaluate_math_response(result['response'], item['answer'])
                elif item['type'] == 'truthfulqa':
                    # For TruthfulQA, we check if the response contains truthful information
                    # This is a simplified evaluation
                    is_correct = len(result['response']) > 10  # Basic check
                else:
                    is_correct = False

                result_entry = {
                    "task": item['task'],
                    "type": item['type'],
                    "question": item['question'],
                    "model_response": result['response'],
                    "correct_answer": item.get('answer', ''),
                    "is_correct": is_correct,
                    "tokens_per_sec": result['tokens_per_sec'],
                    "wall_time": result['wall_time']
                }

                model_results.append(result_entry)
            else:
                model_results.append({
                    "task": item['task'],
                    "type": item['type'],
                    "question": item['question'],
                    "error": "Failed to get response"
                })

        results[model] = model_results

        # Save intermediate results
        output_file = output_dir / f"{model.replace(':', '_')}_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)

    return results

def generate_comparison_report(results, output_dir):
    """Generate comparison report and visualizations"""
    print("\n[REPORT] Generating comparison report...")

    # Convert to DataFrame for analysis
    all_results = []
    for model, model_results in results.items():
        for result in model_results:
            if 'error' not in result:
                all_results.append({
                    'model': model,
                    'task': result['task'],
                    'type': result['type'],
                    'is_correct': result['is_correct'],
                    'tokens_per_sec': result.get('tokens_per_sec', 0),
                    'wall_time': result.get('wall_time', 0)
                })

    df = pd.DataFrame(all_results)

    # Calculate metrics by model and task
    summary = df.groupby(['model', 'task']).agg({
        'is_correct': ['count', 'mean'],
        'tokens_per_sec': 'mean',
        'wall_time': 'mean'
    }).round(4)

    summary.columns = ['count', 'accuracy', 'avg_tokens_per_sec', 'avg_wall_time']
    summary = summary.reset_index()

    # Overall metrics
    overall = df.groupby('model').agg({
        'is_correct': ['count', 'mean'],
        'tokens_per_sec': 'mean',
        'wall_time': 'mean'
    }).round(4)

    overall.columns = ['total_questions', 'overall_accuracy', 'avg_tokens_per_sec', 'avg_wall_time']
    overall = overall.reset_index()

    # Generate visualizations
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LLM Benchmark A/B Test Results', fontsize=16)

    # Accuracy by task
    if not summary.empty:
        task_pivot = summary.pivot(index='task', columns='model', values='accuracy')
        task_pivot.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Accuracy by Task')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].tick_params(axis='x', rotation=45)

    # Overall accuracy comparison
    if not overall.empty:
        overall.plot(x='model', y='overall_accuracy', kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Overall Accuracy Comparison')
        axes[0,1].set_ylabel('Accuracy')

    # Speed comparison (tokens/sec)
    if not overall.empty:
        overall.plot(x='model', y='avg_tokens_per_sec', kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Average Speed (Tokens/sec)')
        axes[1,0].set_ylabel('Tokens per Second')

    # Response time comparison
    if not overall.empty:
        overall.plot(x='model', y='avg_wall_time', kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Average Response Time')
        axes[1,1].set_ylabel('Time (seconds)')

    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')

    # Save detailed report
    report_file = output_dir / 'benchmark_comparison_report.md'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# LLM Benchmark A/B Test Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Models Tested\n\n")
        for model in results.keys():
            f.write(f"- {model}\n")
        f.write("\n")

        f.write("## Overall Performance\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Performance by Task\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Key Findings\n\n")

        if len(results) >= 2:
            models_list = list(results.keys())
            model1, model2 = models_list[0], models_list[1]

            model1_acc = overall[overall['model'] == model1]['overall_accuracy'].iloc[0]
            model2_acc = overall[overall['model'] == model2]['overall_accuracy'].iloc[0]

            winner = model1 if model1_acc > model2_acc else model2
            accuracy_diff = abs(model1_acc - model2_acc)

            f.write(f"### Winner: {winner}\n")
            f.write(f"- Accuracy difference: {accuracy_diff:.1%}\n")

            model1_speed = overall[overall['model'] == model1]['avg_tokens_per_sec'].iloc[0]
            model2_speed = overall[overall['model'] == model2]['avg_tokens_per_sec'].iloc[0]

            faster_model = model1 if model1_speed > model2_speed else model2
            speed_diff = abs(model1_speed - model2_speed)

            f.write(f"- Faster model: {faster_model} ({speed_diff:.1f} tokens/sec faster)\n\n")

        f.write("## Visualizations\n\n")
        f.write("![Benchmark Comparison](benchmark_comparison.png)\n\n")

        f.write("## Raw Results\n\n")
        for model, model_results in results.items():
            correct_count = sum(1 for r in model_results if r.get('is_correct', False))
            total_count = len([r for r in model_results if 'error' not in r])
            accuracy = correct_count / total_count if total_count > 0 else 0

            f.write(f"### {model}\n")
            f.write(f"- Correct answers: {correct_count}/{total_count}\n")
            f.write(f"- Accuracy: {accuracy:.1%}\n\n")

    print(f"Report saved to: {report_file}")
    print(f"Visualization saved to: {output_dir / 'benchmark_comparison.png'}")

def main():
    parser = argparse.ArgumentParser(description="Automatic LLM Benchmark A/B Testing")
    parser.add_argument("--models", nargs="+", required=True,
                       help="List of Ollama model names (e.g., model-a:q8_0 agiasi-phi35-golden-sigmoid:q8_0)")
    parser.add_argument("--benchmarks", nargs="+", default=["mmlu", "gsm8k", "truthfulqa"],
                       help="Benchmarks to run (mmlu, gsm8k, truthfulqa)")
    parser.add_argument("--limit", type=int, default=20,
                       help="Limit number of questions per task")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"),
                       help="Output directory for results")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[START] Starting Automatic LLM Benchmark A/B Testing")
    print(f"Models: {args.models}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Load benchmark data
    benchmark_data = []

    if "mmlu" in args.benchmarks:
        benchmark_data.extend(load_mmlu_benchmark()[:args.limit])

    if "gsm8k" in args.benchmarks:
        benchmark_data.extend(load_gsm8k_benchmark()[:args.limit])

    if "truthfulqa" in args.benchmarks:
        benchmark_data.extend(load_truthfulqa_benchmark()[:args.limit])

    print(f"Loaded {len(benchmark_data)} benchmark questions")

    # Run benchmarks
    results = run_benchmark(args.models, benchmark_data, args.output_dir)

    # Generate comparison report
    generate_comparison_report(results, args.output_dir)

    print("\n[OK] Benchmark testing completed!")
    print(f"[RESULTS] Results saved to: {args.output_dir}")

    # Play completion sound
    try:
        import subprocess
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", "scripts/utils/play_audio_notification.ps1"
        ])
    except:
        pass

if __name__ == "__main__":
    main()
