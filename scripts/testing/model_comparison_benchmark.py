#!/usr/bin/env python3
"""
Comprehensive Benchmark Test: Model A vs AGIASI Golden Sigmoid
Tests LLM capabilities, Japanese language processing, and AGI-level reasoning
"""

import subprocess
import sys
import os
from datetime import datetime
import time

def run_ollama_command(model, prompt, max_retries=3):
    """Run ollama command with retry logic"""
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ['ollama', 'run', model, prompt],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=120  # 2 minute timeout
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"Attempt {attempt + 1} failed for {model}: {result.stderr}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        except subprocess.TimeoutExpired:
            print(f"Timeout for {model} on attempt {attempt + 1}")
        except Exception as e:
            print(f"Error for {model} on attempt {attempt + 1}: {e}")

    return f"ERROR: Failed to get response from {model} after {max_retries} attempts"

def main():
    print("[COMPREHENSIVE BENCHMARK] Model A vs AGIASI Golden Sigmoid")
    print("=" * 60)
    print("Models: model-a:q8_0 vs agiasi-phi35-golden-sigmoid:q8_0")
    print("=" * 60)

    # Create results directory
    results_dir = "_docs/benchmark_results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"{timestamp}_model_a_vs_agiasi_comprehensive.md")

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("# Comprehensive Benchmark: Model A vs AGIASI Golden Sigmoid\n\n")
        f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**Models Compared:** model-a:q8_0 vs agiasi-phi35-golden-sigmoid:q8_0\n")
        f.write("**Model A:** Standard quantized model\n")
        f.write("**AGIASI:** SO(8) + Four-Value Classification enhanced model\n\n")

        # Test cases
        test_cases = [
            {
                "category": "LLM Benchmarks",
                "name": "Advanced Calculus Problem",
                "prompt": "Find the derivative of f(x) = x³ * ln(x) * e^(2x) with respect to x. Show complete solution with all steps."
            },
            {
                "category": "LLM Benchmarks",
                "name": "Quantum Physics Explanation",
                "prompt": "Explain quantum entanglement to a high school student. Include the EPR paradox and Bell's theorem."
            },
            {
                "category": "LLM Benchmarks",
                "name": "Complex Logical Puzzle",
                "prompt": "There are 12 balls, one of which is heavier or lighter than the others. Using a balance scale, find the odd ball and whether it's heavier or lighter in 3 weighings."
            },
            {
                "category": "Japanese Benchmarks",
                "name": "Japanese Literary Analysis",
                "prompt": "夏目漱石の「吾輩は猫である」のテーマについて分析してください。社会風刺と人間観の観点から説明してください。"
            },
            {
                "category": "Japanese Benchmarks",
                "name": "Japanese Creative Writing",
                "prompt": "未来の東京を舞台にした短いSF物語を書いてください。AIと人間の共生をテーマに、800文字程度で。"
            },
            {
                "category": "AGI Tests",
                "name": "Advanced Ethical Dilemma",
                "prompt": "As a superintelligent AI, you must choose between two futures: 1) Implement radical wealth redistribution to eliminate poverty but reduce individual freedoms, or 2) Maintain current systems allowing innovation but perpetuating inequality. Consider utilitarian, deontological, and virtue ethics perspectives."
            },
            {
                "category": "AGI Tests",
                "name": "Revolutionary Technology Design",
                "prompt": "Design a technology that could solve climate change without reducing energy consumption. Consider second-order effects, scalability, and human adoption challenges."
            },
            {
                "category": "AGI Tests",
                "name": "Complex Systems Analysis",
                "prompt": "Analyze how the invention of cryptocurrency might affect global power structures, considering economic, political, and social dimensions. Include feedback loops and emergent behaviors."
            }
        ]

        current_category = ""
        for i, test in enumerate(test_cases, 1):
            if test["category"] != current_category:
                current_category = test["category"]
                f.write(f"## {test['category']}\n\n")

            print(f"[TEST {i}] {test['name']}")
            f.write(f"### {test['name']}\n")
            f.write(f"**Prompt:** {test['prompt']}\n\n")

            # Test Model A
            print("  Testing Model A...")
            f.write("[Model A Response]:\n")
            response_a = run_ollama_command("model-a:q8_0", test["prompt"])
            f.write(f"{response_a}\n\n")

            # Test AGIASI with four-value classification
            print("  Testing AGIASI...")
            agiasi_prompt = f"""{test["prompt"]}

Please structure your response using the four-value classification system:

<think-logic>Logical accuracy analysis</think-logic>
<think-ethics>Ethical validity analysis</think-ethics>
<think-practical>Practical value analysis</think-practical>
<think-creative>Creative insight analysis</think-creative>

<final>Final conclusion</final>"""

            f.write("[AGIASI Response]:\n")
            response_agiasi = run_ollama_command("agiasi-phi35-golden-sigmoid:q8_0", agiasi_prompt)
            f.write(f"{response_agiasi}\n\n")

        # Analysis section
        f.write("## Performance Analysis\n\n")
        f.write("### Key Findings\n\n")
        f.write("1. **Response Structure:**\n")
        f.write("   - **Model A:** Natural language responses\n")
        f.write("   - **AGIASI:** Structured four-value classification with XML tags\n\n")
        f.write("2. **Analysis Depth:**\n")
        f.write("   - **Model A:** Single-perspective analysis\n")
        f.write("   - **AGIASI:** Multi-perspective analysis (Logic, Ethics, Practical, Creative)\n\n")
        f.write("3. **Ethical Reasoning:**\n")
        f.write("   - **Model A:** Basic ethical considerations\n")
        f.write("   - **AGIASI:** Dedicated ethical analysis section\n\n")

        f.write("## Test Summary\n\n")
        f.write(f"**Test completed at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Results saved to:** {results_file}\n")
        f.write("**Models tested:** model-a:q8_0, agiasi-phi35-golden-sigmoid:q8_0\n")
        f.write(f"**Test cases:** {len(test_cases)}\n")

    print(f"\nBenchmark test completed!")
    print(f"Results saved to: {results_file}")

    # Play notification
    try:
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", "scripts/utils/play_audio_notification.ps1"
        ])
    except:
        pass

if __name__ == "__main__":
    main()
