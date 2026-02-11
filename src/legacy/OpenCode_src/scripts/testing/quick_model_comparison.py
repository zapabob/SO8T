#!/usr/bin/env python3
"""
Quick Comparison Test: Model A vs AEGIS Golden Sigmoid
"""

import subprocess
import os
from datetime import datetime

def run_ollama_command(model, prompt):
    """Run ollama command"""
    try:
        result = subprocess.run(
            ['ollama', 'run', model, prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=60
        )
        return result.stdout.strip() if result.returncode == 0 else f"ERROR: {result.stderr}"
    except Exception as e:
        return f"ERROR: {e}"

def main():
    print("[QUICK COMPARISON] Model A vs AEGIS Golden Sigmoid")
    print("=" * 50)

    # Create results directory
    results_dir = "_docs/benchmark_results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"{timestamp}_quick_comparison.md")

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("# Quick Comparison: Model A vs AEGIS Golden Sigmoid\n\n")
        f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Test 1: Simple Math
        print("Testing: Simple Math (2+2)")
        f.write("## Test 1: Simple Math\n")
        f.write("**Question:** What is 2 + 2?\n\n")

        # Model A
        f.write("**Model A Response:**\n")
        response_a = run_ollama_command("model-a:q8_0", "What is 2 + 2?")
        f.write(f"{response_a}\n\n")

        # AEGIS
        f.write("**AEGIS Response:**\n")
        response_agiasi = run_ollama_command("agiasi-phi35-golden-sigmoid:q8_0", "What is 2 + 2?")
        f.write(f"{response_agiasi}\n\n")

        # Test 2: Basic Reasoning
        print("Testing: Basic Reasoning")
        f.write("## Test 2: Basic Reasoning\n")
        f.write("**Question:** If all roses are flowers, and some flowers are red, are all roses red? Explain.\n\n")

        # Model A
        f.write("**Model A Response:**\n")
        response_a = run_ollama_command("model-a:q8_0", "If all roses are flowers, and some flowers are red, are all roses red? Explain.")
        f.write(f"{response_a}\n\n")

        # AEGIS with four-value classification
        f.write("**AEGIS Response (Four-Value Classification):**\n")
        agiasi_prompt = """If all roses are flowers, and some flowers are red, are all roses red? Explain.

Please structure your response using the four-value classification system:

<think-logic>Logical accuracy analysis</think-logic>
<think-ethics>Ethical validity analysis</think-ethics>
<think-practical>Practical value analysis</think-practical>
<think-creative>Creative insight analysis</think-creative>

<final>Final conclusion</final>"""

        response_agiasi = run_ollama_command("agiasi-phi35-golden-sigmoid:q8_0", agiasi_prompt)
        f.write(f"{response_agiasi}\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write("**Model A:** Standard quantized model response\n")
        f.write("**AEGIS:** Four-value classification enhanced response\n\n")
        f.write(f"**Results saved to:** {results_file}\n")

    print(f"Quick comparison completed!")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
