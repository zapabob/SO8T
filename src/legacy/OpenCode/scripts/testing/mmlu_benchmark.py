#!/usr/bin/env python3
"""
MMLU-style Benchmark Runner
Evaluates LLMs using 30-question MMLU-style English benchmark
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List

def run_ollama(model: str, prompt: str, timeout: int = 60) -> tuple[str, float]:
    """Run inference via Ollama"""
    start_time = time.time()
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8'
        )
        duration = time.time() - start_time
        return result.stdout.strip(), duration
    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout
    except Exception as e:
        return f"ERROR: {e}", 0.0

def evaluate_mmlu(model_name: str, model_id: str, trial: int = 1) -> Dict:
    """Run MMLU benchmark on a model"""
    print(f"\n{'='*60}")
    print(f"MMLU Benchmark - Model: {model_name} (Trial {trial})")
    print(f"{'='*60}")
    
    # Load questions
    tasks_file = Path("_data/mmlu_samples/mmlu_tasks.json")
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    results = []
    correct = 0
    total = len(tasks)
    
    for task in tasks:
        # Construct prompt with choices
        prompt = f"""Question: {task['question']}

{chr(10).join(task['choices'])}

Provide your answer as a single letter (A, B, C, or D) followed by a brief explanation.
Answer:"""
        
        print(f"\nQ{task['id']} ({task['domain']}): ", end="", flush=True)
        response, duration = run_ollama(model_id, prompt)
        
        # Extract answer (first letter A-D)
        answer_extracted = None
        for char in response.upper():
            if char in ['A', 'B', 'C', 'D']:
                answer_extracted = char
                break
        
        is_correct = (answer_extracted == task['answer'])
        if is_correct:
            correct += 1
            print(f"✓ ({answer_extracted})", end="")
        else:
            print(f"✗ ({answer_extracted} vs {task['answer']})", end="")
        
        print(f" [{duration:.1f}s]")
        
        results.append({
            "task_id": task['id'],
            "domain": task['domain'],
            "question": task['question'],
            "expected": task['answer'],
            "predicted": answer_extracted,
            "correct": is_correct,
            "duration": duration,
            "response": response[:200]  # Truncate for storage
        })
    
    accuracy = (correct / total) * 100
    print(f"\n{'='*60}")
    print(f"Results: {correct}/{total} correct ({accuracy:.1f}%)")
    print(f"{'='*60}")
    
    # Domain breakdown
    domain_stats = {}
    for r in results:
        domain = r['domain']
        if domain not in domain_stats:
            domain_stats[domain] = {'correct': 0, 'total': 0}
        domain_stats[domain]['total'] += 1
        if r['correct']:
            domain_stats[domain]['correct'] += 1
    
    print("\nDomain Breakdown:")
    for domain, stats in domain_stats.items():
        acc = (stats['correct'] / stats['total']) * 100
        print(f"  {domain}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    return {
        "model_name": model_name,
        "model_id": model_id,
        "trial": trial,
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": accuracy,
        "domain_breakdown": {
            domain: {
                "correct": stats['correct'],
                "total": stats['total'],
                "accuracy": (stats['correct'] / stats['total']) * 100
            }
            for domain, stats in domain_stats.items()
        },
        "detailed_results": results
    }

def main():
    """Run MMLU benchmark on all models"""
    models = {
        "Model_A": "model-a:q8_0",
        "AEGIS_0.8": "aegis-0.8",
        "AEGIS_0.6": "aegis-adjusted-0.6"
    }
    
    num_trials = 3
    all_results = []
    
    for model_name, model_id in models.items():
        for trial in range(1, num_trials + 1):
            result = evaluate_mmlu(model_name, model_id, trial)
            all_results.append(result)
            time.sleep(2)  # Brief pause between trials
    
    # Save results
    output_dir = Path("_docs/benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "mmlu_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ MMLU benchmark complete! Results saved to {output_file}")

if __name__ == "__main__":
    main()
