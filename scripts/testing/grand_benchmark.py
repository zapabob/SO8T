#!/usr/bin/env python3
"""
Grand Benchmark Orchestrator
Executes massive-scale evaluation on MMLU, GSM8K, MATH, ELYZA-100, AGIEval.
Features:
- Resumable execution
- A/B testing (interleaved)
- Robust Ollama client
"""

import json
import time
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
import subprocess
import re
import shutil
import glob
import os
from tqdm import tqdm

# Configuration
DATASET_DIR = Path("D:/webdataset")
RESULTS_DIR = Path("_docs/grand_benchmark_results")
CHECKPOINT_DIR = Path("_docs/grand_benchmark_checkpoints")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Model_A": "model-a:q8_0",
    "AEGIS_0.6": "aegis-adjusted-0.6",
    "AEGIS_0.8": "aegis-0.8"
}

class CheckpointManager:
    def __init__(self, interval_seconds=180, max_keep=5):
        self.interval = interval_seconds
        self.max_keep = max_keep
        self.last_save_time = time.time()
        
    def should_save(self):
        return (time.time() - self.last_save_time) >= self.interval
        
    def save_checkpoint(self, state: Dict):
        """Save rolling checkpoint"""
        timestamp = int(time.time())
        ckpt_path = CHECKPOINT_DIR / f"checkpoint_{timestamp}.json"
        
        # Save state
        with open(ckpt_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
            
        print(f"\n[Checkpoint] Saved state to {ckpt_path.name}")
        self.last_save_time = time.time()
        self.cleanup_old_checkpoints()
        
    def cleanup_old_checkpoints(self):
        """Keep only max_keep latest checkpoints"""
        ckpts = sorted(CHECKPOINT_DIR.glob("checkpoint_*.json"), key=os.path.getmtime)
        while len(ckpts) > self.max_keep:
            oldest = ckpts.pop(0)
            try:
                oldest.unlink()
            except Exception as e:
                print(f"Error deleting checkpoint {oldest}: {e}")

    def load_latest_checkpoint(self) -> Dict:
        ckpts = sorted(CHECKPOINT_DIR.glob("checkpoint_*.json"), key=os.path.getmtime)
        if not ckpts:
            return None
        latest = ckpts[-1]
        print(f"[Resume] Loading latest checkpoint: {latest.name}")
        with open(latest, 'r', encoding='utf-8') as f:
            return json.load(f)

def run_ollama(model: str, prompt: str, timeout: int = 120) -> str:
    """Robust Ollama inference"""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8'
        )
        if result.returncode != 0:
            return f"ERROR: {result.stderr.strip()}"
        
        output = result.stdout.strip()
        if not output and result.stderr:
             return f"ERROR: {result.stderr.strip()}"
             
        return output
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"

class Evaluator:
    def evaluate(self, model: str, item: Dict) -> Dict:
        raise NotImplementedError

class MMLUEvaluator(Evaluator):
    def evaluate(self, model: str, item: Dict) -> Dict:
        prompt = f"""Question: {item['question']}
{chr(10).join(item['choices'])}
Answer with the single letter of the correct option (A, B, C, or D).
Answer:"""
        response = run_ollama(model, prompt)
        # Extract A, B, C, D
        match = re.search(r'\b([A-D])\b', response.upper())
        prediction = match.group(1) if match else "UNKNOWN"
        correct = (prediction == ["A", "B", "C", "D"][item['answer']])
        return {
            "prediction": prediction,
            "correct": correct,
            "prompt": prompt,
            "response": response
        }

class GSM8KEvaluator(Evaluator):
    def evaluate(self, model: str, item: Dict) -> Dict:
        prompt = f"""Question: {item['question']}
Let's think step by step.
Answer:"""
        response = run_ollama(model, prompt)
        # Extract number from response (simplified)
        # GSM8K usually has #### <answer>
        match = re.search(r'####\s*(-?\d+\.?\d*)', item['answer'])
        expected_val = float(match.group(1)) if match else None
        
        # Naive extraction from response
        nums = re.findall(r'-?\d+\.?\d*', response)
        predicted_val = float(nums[-1]) if nums else None
        
        correct = (expected_val is not None and predicted_val is not None and abs(expected_val - predicted_val) < 1e-6)
        return {
            "prediction": predicted_val,
            "correct": correct,
            "prompt": prompt,
            "response": response
        }

class ELYZAEvaluator(Evaluator):
    def evaluate(self, model: str, item: Dict) -> Dict:
        # Simplified ELYZA evaluation (keyword match as placeholder)
        # Real eval requires LLM-as-a-Judge or complex rules
        prompt = f"{item['question']}\n\n回答:"
        response = run_ollama(model, prompt)
        return {
            "response": response,
            "prompt": prompt,
            "correct": None # Requires manual or LLM judge
        }

def load_dataset_jsonl(path: Path, limit: int = None) -> List[Dict]:
    data = []
    if not path.exists():
        print(f"Warning: {path} not found.")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per dataset")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    # Define Benchmarks
    benchmarks = [
        ("MMLU", DATASET_DIR / "mmlu_test.jsonl", MMLUEvaluator()),
        ("GSM8K", DATASET_DIR / "gsm8k_test.jsonl", GSM8KEvaluator()),
        ("ELYZA", DATASET_DIR / "elyza100_test.jsonl", ELYZAEvaluator())
    ]
    
    ckpt_mgr = CheckpointManager(interval_seconds=180, max_keep=5)
    start_benchmark_idx = 0
    start_item_idx = 0
    
    if args.resume:
        state = ckpt_mgr.load_latest_checkpoint()
        if state:
            start_benchmark_idx = state.get('benchmark_idx', 0)
            start_item_idx = state.get('item_idx', 0)
            print(f"Resuming from Benchmark Index {start_benchmark_idx}, Item {start_item_idx}")

    for b_idx, (name, path, evaluator) in enumerate(benchmarks):
        if b_idx < start_benchmark_idx:
            continue
            
        print(f"\nStarting Benchmark: {name}")
        dataset = load_dataset_jsonl(path, args.limit)
        if not dataset:
            continue

        results_file = RESULTS_DIR / f"{name}_results.jsonl"
        
        # If resuming within the same benchmark, skip processed items
        current_start_item = start_item_idx if b_idx == start_benchmark_idx else 0
        
        # Create tqdm progress bar
        pbar = tqdm(enumerate(dataset), total=len(dataset), desc=f"Running {name}", initial=current_start_item)
        
        for i, item in pbar:
            if i < current_start_item:
                continue
            
            # Process all models for this item
            for model_name, model_id in MODELS.items():
                res = evaluator.evaluate(model_id, item)
                result_entry = {
                    "benchmark": name,
                    "model": model_name,
                    "id": i,
                    "result": res
                }
                with open(results_file, 'a', encoding='utf-8') as f:
                    json.dump(result_entry, f, ensure_ascii=False)
                    f.write('\n')
            
            # Checkpoint logic
            if ckpt_mgr.should_save():
                state = {
                    'benchmark_idx': b_idx,
                    'item_idx': i + 1, # Next item to process
                    'timestamp': time.time()
                }
                ckpt_mgr.save_checkpoint(state)
                pbar.set_postfix({"saved": "yes"})
                
        print(f"\n{name} Completed.")
        # Reset item index for next benchmark
        start_item_idx = 0

if __name__ == "__main__":
    main()
