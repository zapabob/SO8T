#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELYZA-100 Benchmark Runner (Evaluation Version)
Evaluates LLMs using Japanese language capability benchmark
Integrated with industry-standard benchmark pipeline
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_OUTPUT_ROOT = Path(r"D:/webdataset/benchmark_results/industry_standard")
ELYZA_TASKS_FILE = Path("_data/elyza100_samples/elyza_tasks.json")


def run_ollama(model: str, prompt: str, timeout: int = 120) -> Tuple[str, float]:
    """Run inference via Ollama"""
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LANG'] = 'C.UTF-8'
    
    start_time = time.time()
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        duration = time.time() - start_time
        if result.returncode == 0:
            return result.stdout.strip(), duration
        else:
            return f"[ERROR] {result.stderr}", duration
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return "[ERROR] Timeout", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return f"[ERROR] {e}", elapsed


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


def run_elyza_benchmark(
    model_name: str,
    model_id: str,
    output_dir: Path,
    tasks_file: Path = ELYZA_TASKS_FILE,
    limit: Optional[int] = None,
) -> Dict:
    """Run ELYZA-100 benchmark on a model"""
    print(f"\n{'='*60}")
    print(f"ELYZA-100 Benchmark - Model: {model_name}")
    print(f"{'='*60}")
    
    # Load questions
    if not tasks_file.exists():
        raise FileNotFoundError(f"ELYZA tasks file not found: {tasks_file}")
    
    with tasks_file.open('r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    # Apply limit if specified
    if limit is not None:
        tasks = tasks[:limit]
    
    results = []
    correct = 0
    total = len(tasks)
    
    print(f"[INFO] Running {total} ELYZA-100 tasks...")
    
    for idx, task in enumerate(tasks, 1):
        prompt = f"{task['question']}\n\n回答:"
        
        print(f"[{idx}/{total}] Q{task['id']} ({task['category']}): ", end="", flush=True)
        response, duration = run_ollama(model_id, prompt, timeout=120)
        
        is_correct = evaluate_answer(response, task['expected_answer'], task['category'])
        
        if is_correct:
            correct += 1
            print(f"[OK]", end="")
        else:
            print(f"[NG]", end="")
        
        print(f" [{duration:.1f}s]")
        
        results.append({
            "task_id": task['id'],
            "category": task['category'],
            "question": task['question'],
            "expected": task['expected_answer'],
            "response": response[:500],  # 最初の500文字を保存
            "correct": is_correct,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
        })
        
        # レート制限対策
        time.sleep(0.5)
    
    accuracy = (correct / total) * 100 if total > 0 else 0.0
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
        acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0.0
        print(f"  {category}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    # Prepare result summary
    result_summary = {
        "model_name": model_name,
        "model_id": model_id,
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": accuracy,
        "category_breakdown": {
            category: {
                "correct": stats['correct'],
                "total": stats['total'],
                "accuracy": (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0.0
            }
            for category, stats in category_stats.items()
        },
        "detailed_results": results,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"elyza_{model_name.replace(':', '_').replace('/', '_')}_results.json"
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SAVE] Results saved to {output_file}")
    
    return result_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ELYZA-100 Benchmark Runner (Evaluation Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Ollama model name (e.g., 'model-a:q8_0', 'aegis-phi3.5-fixed-0.8:latest')",
    )
    parser.add_argument(
        "--model-display-name",
        default=None,
        help="Display name for the model (defaults to model-name)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (defaults to output-root/model_name)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for output (used if output-dir not specified)",
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=ELYZA_TASKS_FILE,
        help="Path to ELYZA tasks JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to run (for testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_safe_name = args.model_name.replace(':', '_').replace('/', '_')
        output_dir = args.output_root / "elyza" / model_safe_name
    
    # Determine display name
    display_name = args.model_display_name or args.model_name
    
    # Run benchmark
    try:
        result = run_elyza_benchmark(
            model_name=display_name,
            model_id=args.model_name,
            output_dir=output_dir,
            tasks_file=args.tasks_file,
            limit=args.limit,
        )
        
        print(f"\n[SUCCESS] ELYZA-100 benchmark completed for {display_name}")
        print(f"  Accuracy: {result['accuracy']:.1f}%")
        print(f"  Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n[ERROR] ELYZA-100 benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()


