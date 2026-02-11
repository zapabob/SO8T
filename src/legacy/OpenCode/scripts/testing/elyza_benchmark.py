#!/usr/bin/env python3
"""
ELYZA-100 Benchmark Runner
Evaluates LLMs using 20-question Japanese language capability benchmark
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

def evaluate_answer(response: str, expected: str, category: str) -> bool:
    """Evaluate if response contains expected answer (keyword-based)"""
    response_clean = response.lower().strip()
    expected_clean = expected.lower().strip()
    
    # For multiple choice, check if option letter is present
    if ':' in expected or ')' in expected:
        # Extract option (e.g., "a) 寿司" -> "a")
        option = expected.split(')')[0].strip().lower()
        return option in response_clean
    
    # For numerical answers
    if category == "mathematics":
        # Extract numbers from response
        import re
        response_nums = re.findall(r'\d+', response)
        expected_nums = re.findall(r'\d+', expected)
        if response_nums and expected_nums:
            return response_nums[0] == expected_nums[0]
    
    # General keyword matching
    # Split expected into keywords
    keywords = [k.strip() for k in expected.split(',')]
    # Check if any keyword present
    for keyword in keywords:
        if keyword.lower() in response_clean:
            return True
    
    return False

def evaluate_elyza(model_name: str, model_id: str, trial: int = 1) -> Dict:
    """Run ELYZA-100 benchmark on a model"""
    print(f"\n{'='*60}")
    print(f"ELYZA-100 Benchmark - Model: {model_name} (Trial {trial})")
    print(f"{'='*60}")
    
    # Load questions
    tasks_file = Path("_data/elyza100_samples/elyza_tasks.json")
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    results = []
    correct = 0
    total = len(tasks)
    
    for task in tasks:
        prompt = f"{task['question']}\n\n回答:"
        
        print(f"\nQ{task['id']} ({task['category']}): ", end="", flush=True)
        response, duration = run_ollama(model_id, prompt)
        
        is_correct = evaluate_answer(response, task['expected_answer'], task['category'])
        
        if is_correct:
            correct += 1
            print(f"✓", end="")
        else:
            print(f"✗", end="")
        
        print(f" [{duration:.1f}s]")
        
        results.append({
            "task_id": task['id'],
            "category": task['category'],
            "question": task['question'],
            "expected": task['expected_answer'],
            "response": response[:200],
            "correct": is_correct,
            "duration": duration
        })
    
    accuracy = (correct / total) * 100
    print(f"\n{'='*60}")
    print(f"Results: {correct}/{total} correct ({accuracy:.1f}%)")
    print(f"{'='*60}")
    
    # Category breakdown
    category_stats = {}
    for r in results:
        category = r['category']
        if category not in category_stats:
            category_stats[category] = {'correct': 0, 'total': 0}
        category_stats[category]['total'] += 1
        if r['correct']:
            category_stats[category]['correct'] += 1
    
    print("\nCategory Breakdown:")
    for category, stats in category_stats.items():
        acc = (stats['correct'] / stats['total']) * 100
        print(f"  {category}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    return {
        "model_name": model_name,
        "model_id": model_id,
        "trial": trial,
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": accuracy,
        "category_breakdown": {
            category: {
                "correct": stats['correct'],
                "total": stats['total'],
                "accuracy": (stats['correct'] / stats['total']) * 100
            }
            for category, stats in category_stats.items()
        },
        "detailed_results": results
    }

def main():
    """Run ELYZA-100 benchmark on all models"""
    models = {
        "Model_A": "model-a:q8_0",
        "AEGIS_0.8": "aegis-0.8",
        "AEGIS_0.6": "aegis-adjusted-0.6"
    }
    
    num_trials = 3
    all_results = []
    
    for model_name, model_id in models.items():
        for trial in range(1, num_trials + 1):
            result = evaluate_elyza(model_name, model_id, trial)
            all_results.append(result)
            time.sleep(2)  # Brief pause between trials
    
    # Save results
    output_dir = Path("_docs/benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "elyza_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ ELYZA-100 benchmark complete! Results saved to {output_file}")

if __name__ == "__main__":
    main()
