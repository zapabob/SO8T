# benchmark_ollama.py
"""Run benchmarks against Ollama models.

Usage:
    python benchmark_ollama.py --models "agiasi:q8,agiasi:q4,borea:latest" --output results.json
"""
import argparse
import json
import time
import requests
from pathlib import Path
from datetime import datetime

OLLAMA_API = "http://localhost:11434/api/generate"

def run_prompt(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 512
        }
    }
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_API, json=payload)
        response.raise_for_status()
        data = response.json()
        end_time = time.time()
        
        return {
            "response": data.get("response", ""),
            "total_duration": data.get("total_duration", 0), # nanoseconds
            "load_duration": data.get("load_duration", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "eval_count": data.get("eval_count", 0),
            "eval_duration": data.get("eval_duration", 0), # nanoseconds
            "tokens_per_sec": data.get("eval_count", 0) / (data.get("eval_duration", 1) / 1e9),
            "wall_time": end_time - start_time
        }
    except Exception as e:
        print(f"Error running {model}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of model tags")
    parser.add_argument("--prompts", type=Path, default=Path("prompts.json"))
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    
    results = []

    print(f"Benchmarking models: {models}")
    
    for model in models:
        print(f"\n--- Testing {model} ---")
        for p in prompts:
            print(f"  Running {p['id']}...")
            metrics = run_prompt(model, p["prompt"])
            if metrics:
                result_entry = {
                    "model": model,
                    "category": p["category"],
                    "test_id": p["id"],
                    "prompt": p["prompt"],
                    "response": metrics["response"],
                    "tokens_per_sec": metrics["tokens_per_sec"],
                    "eval_count": metrics["eval_count"],
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result_entry)
                # print(f"    Speed: {metrics['tokens_per_sec']:.2f} t/s")
            else:
                print("    Failed.")
    
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results to {args.output}")

if __name__ == "__main__":
    main()
