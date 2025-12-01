#!/usr/bin/env python3
"""Simple A/B test for SO8T models"""

import os
import sys
import json
import subprocess
from pathlib import Path

def run_single_test(model_path, model_name, output_file):
    """Run lm_eval for a single model"""
    lm_eval_dir = Path(__file__).parent.parent.parent / 'lm-evaluation-harness'

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", "hellaswag",
        "--batch_size", "auto",
        "--device", "cuda:0",
        "--limit", "10"
    ]

    print(f"Running: {' '.join(cmd)}")

    with open(output_file, 'w', encoding='utf-8') as f:
        result = subprocess.run(cmd, cwd=str(lm_eval_dir), stdout=f, stderr=subprocess.STDOUT, text=True)

    return result.returncode == 0

def parse_lm_eval_output(output_file):
    """Parse lm_eval output"""
    if not output_file.exists():
        return None

    with open(output_file, 'r', errors='ignore') as f:
        content = f.read()

    lines = content.split('\n')
    results = {}

    # Find the table with results
    for line in lines:
        if 'hellaswag' in line and 'acc' in line:
            parts = line.split('|')
            if len(parts) >= 8:
                try:
                    acc_value = float(parts[7].strip())
                    acc_norm_value = float(parts[9].strip()) if len(parts) > 9 else None

                    results['hellaswag'] = {
                        'acc': acc_value,
                        'acc_norm': acc_norm_value
                    }
                    break
                except (ValueError, IndexError):
                    continue

    return results

def main():
    # Test with different models
    model_a_path = "../models/Borea-Phi-3.5-mini-Instruct-Jp"
    model_b_path = "microsoft/DialoGPT-medium"

    model_a_name = "Borea-Phi-3.5-Base"
    model_b_name = "DialoGPT-Medium"

    output_dir = Path("simple_ab_test_results")
    output_dir.mkdir(exist_ok=True)

    # Run tests
    print(f"Testing Model A: {model_a_name}")
    a_output = output_dir / "model_a_output.txt"
    a_success = run_single_test(model_a_path, model_a_name, a_output)

    print(f"Testing Model B: {model_b_name}")
    b_output = output_dir / "model_b_output.txt"
    b_success = run_single_test(model_b_path, model_b_name, b_output)

    if not a_success or not b_success:
        print("One or both model tests failed!")
        return

    # Parse results
    a_results = parse_lm_eval_output(a_output)
    b_results = parse_lm_eval_output(b_output)

    print("\nResults:")
    print(f"Model A ({model_a_name}): {a_results}")
    print(f"Model B ({model_b_name}): {b_results}")

    if a_results and b_results and 'hellaswag' in a_results and 'hellaswag' in b_results:
        a_acc = a_results['hellaswag'].get('acc', 0)
        b_acc = b_results['hellaswag'].get('acc', 0)

        improvement = b_acc - a_acc
        print(f"Model A accuracy: {a_acc:.4f}")
        print(f"Model B accuracy: {b_acc:.4f}")
        if improvement > 0:
            print("✅ Model B shows improvement!")
        elif improvement < 0:
            print("❌ Model B shows degradation!")
        else:
            print("🟡 No difference between models!")

if __name__ == "__main__":
    main()
