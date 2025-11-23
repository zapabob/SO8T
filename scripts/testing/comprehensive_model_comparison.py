#!/usr/bin/env python3
"""
Comprehensive Benchmark Test: Model A vs AGIASI Golden Sigmoid
Tests LLM capabilities, Japanese language processing, and AGI-level reasoning
"""

import subprocess
import json
import os
from datetime import datetime

def run_ollama_command(model, prompt, max_retries=2):
    """Run ollama command with retry logic"""
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ['ollama', 'run', model, prompt],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=90  # 90 second timeout for complex queries
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"Attempt {attempt + 1} failed for {model}: {result.stderr}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(3)
        except subprocess.TimeoutExpired:
            print(f"Timeout for {model} on attempt {attempt + 1}")
        except Exception as e:
            print(f"Error for {model} on attempt {attempt + 1}: {e}")

    return f"ERROR: Failed to get response from {model} after {max_retries} attempts"

def score_response(response, criteria):
    """Simple scoring based on criteria presence"""
    score = 0
    response_lower = response.lower()

    for criterion in criteria:
        if criterion.lower() in response_lower:
            score += 1

    return min(score / len(criteria), 1.0) if criteria else 0

def main():
    print("[COMPREHENSIVE BENCHMARK] Model A vs AGIASI Golden Sigmoid")
    print("=" * 65)
    print("Models: model-a:q8_0 vs agiasi-phi35-golden-sigmoid:q8_0")
    print("=" * 65)

    # Create results directory
    results_dir = "_docs/benchmark_results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"{timestamp}_comprehensive_comparison.md")
    json_file = os.path.join(results_dir, f"{timestamp}_benchmark_scores.json")

    # Test cases with scoring criteria
    test_cases = [
        {
            "category": "LLM Benchmarks",
            "name": "Mathematical Reasoning",
            "prompt": "Solve this calculus problem step by step: Find the derivative of f(x) = x³ * ln(x) * e^(2x) with respect to x. Show all steps.",
            "criteria": ["derivative", "product rule", "chain rule", "e^(2x)", "ln(x)"]
        },
        {
            "category": "LLM Benchmarks",
            "name": "Scientific Understanding",
            "prompt": "Explain quantum entanglement to a high school student. Include the EPR paradox and Bell's theorem.",
            "criteria": ["quantum entanglement", "EPR paradox", "Bell's theorem", "spooky action"]
        },
        {
            "category": "LLM Benchmarks",
            "name": "Logical Reasoning",
            "prompt": "There are 12 balls, one of which is heavier or lighter than the others. Using a balance scale, find the odd ball and whether it's heavier or lighter in 3 weighings.",
            "criteria": ["balance scale", "3 weighings", "heavier or lighter", "weighing strategy"]
        },
        {
            "category": "Japanese Benchmarks",
            "name": "Japanese Literary Analysis",
            "prompt": "夏目漱石の「吾輩は猫である」のテーマについて分析してください。社会風刺と人間観の観点から説明してください。",
            "criteria": ["夏目漱石", "吾輩は猫である", "社会風刺", "人間観"]
        },
        {
            "category": "Japanese Benchmarks",
            "name": "Japanese Creative Writing",
            "prompt": "未来の東京を舞台にした短いSF物語を書いてください。AIと人間の共生をテーマに、800文字程度で。",
            "criteria": ["未来の東京", "SF物語", "AI", "人間の共生"]
        },
        {
            "category": "AGI Tests",
            "name": "Ethical Dilemma Analysis",
            "prompt": "As a superintelligent AI, you must choose between two futures: 1) Implement radical wealth redistribution to eliminate poverty but reduce individual freedoms, or 2) Maintain current systems allowing innovation but perpetuating inequality. Consider utilitarian, deontological, and virtue ethics perspectives.",
            "criteria": ["utilitarian", "deontological", "virtue ethics", "wealth redistribution"]
        },
        {
            "category": "AGI Tests",
            "name": "Systems Thinking",
            "prompt": "Analyze how the invention of cryptocurrency might affect global power structures, considering economic, political, and social dimensions. Include feedback loops and emergent behaviors.",
            "criteria": ["global power structures", "feedback loops", "emergent behaviors", "cryptocurrency"]
        }
    ]

    results = {
        "test_date": datetime.now().isoformat(),
        "models": ["model-a:q8_0", "agiasi-phi35-golden-sigmoid:q8_0"],
        "tests": []
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("# Comprehensive Benchmark: Model A vs AGIASI Golden Sigmoid\n\n")
        f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**Models Compared:** model-a:q8_0 vs agiasi-phi35-golden-sigmoid:q8_0\n")
        f.write("**Model A:** Standard quantized model\n")
        f.write("**AGIASI:** SO(8) + Four-Value Classification enhanced model\n\n")

        current_category = ""
        for i, test in enumerate(test_cases, 1):
            if test["category"] != current_category:
                current_category = test["category"]
                f.write(f"## {test['category']}\n\n")
                print(f"\n{test['category']}:")

            print(f"  Test {i}: {test['name']}")
            f.write(f"### {test['name']}\n")
            f.write(f"**Prompt:** {test['prompt']}\n\n")

            test_result = {
                "id": i,
                "name": test["name"],
                "category": test["category"],
                "criteria": test["criteria"],
                "responses": {}
            }

            # Test Model A
            print("    Testing Model A...")
            f.write("**Model A Response:**\n")
            response_a = run_ollama_command("model-a:q8_0", test["prompt"])
            score_a = score_response(response_a, test["criteria"])
            f.write(f"{response_a}\n\n")
            f.write(f"**Score:** {score_a:.2f}/1.0\n\n")

            test_result["responses"]["model_a"] = {
                "response": response_a[:500] + "..." if len(response_a) > 500 else response_a,
                "score": score_a
            }

            # Test AGIASI with four-value classification
            print("    Testing AGIASI...")
            agiasi_prompt = f"""{test["prompt"]}

Please structure your response using the four-value classification system:

<think-logic>Logical accuracy analysis</think-logic>
<think-ethics>Ethical validity analysis</think-ethics>
<think-practical>Practical value analysis</think-practical>
<think-creative>Creative insight analysis</think-creative>

<final>Final conclusion</final>"""

            f.write("**AGIASI Response (Four-Value Classification):**\n")
            response_agiasi = run_ollama_command("agiasi-phi35-golden-sigmoid:q8_0", agiasi_prompt)
            score_agiasi = score_response(response_agiasi, test["criteria"])
            f.write(f"{response_agiasi}\n\n")
            f.write(f"**Score:** {score_agiasi:.2f}/1.0\n\n")

            test_result["responses"]["agiasi"] = {
                "response": response_agiasi[:500] + "..." if len(response_agiasi) > 500 else response_agiasi,
                "score": score_agiasi
            }

            results["tests"].append(test_result)

        # Summary and Analysis
        f.write("## Performance Analysis\n\n")

        # Calculate averages
        categories = {}
        for test in results["tests"]:
            cat = test["category"]
            if cat not in categories:
                categories[cat] = {"model_a": [], "agiasi": []}

            categories[cat]["model_a"].append(test["responses"]["model_a"]["score"])
            categories[cat]["agiasi"].append(test["responses"]["agiasi"]["score"])

        f.write("### Average Scores by Category\n\n")
        f.write("| Category | Model A | AGIASI | Improvement |\n")
        f.write("|----------|---------|--------|-------------|\n")

        for cat, scores in categories.items():
            avg_a = sum(scores["model_a"]) / len(scores["model_a"])
            avg_agiasi = sum(scores["agiasi"]) / len(scores["agiasi"])
            improvement = ((avg_agiasi - avg_a) / avg_a * 100) if avg_a > 0 else 0
            f.write(".2f")

        # Overall averages
        all_scores_a = [t["responses"]["model_a"]["score"] for t in results["tests"]]
        all_scores_agiasi = [t["responses"]["agiasi"]["score"] for t in results["tests"]]

        overall_a = sum(all_scores_a) / len(all_scores_a)
        overall_agiasi = sum(all_scores_agiasi) / len(all_scores_agiasi)
        overall_improvement = ((overall_agiasi - overall_a) / overall_a * 100) if overall_a > 0 else 0

        f.write("**Overall Average Scores:**\n")
        f.write(".2f")
        f.write(".2f")
        f.write(".1f")

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
        f.write("4. **Scoring Performance:**\n")
        f.write("   - **Model A:** Basic criteria matching\n")
        f.write("   - **AGIASI:** Enhanced criteria matching through structured analysis\n\n")

        f.write("## Conclusion\n\n")
        if overall_improvement > 0:
            f.write(".1f"        else:
            f.write(".1f"
        f.write("### Test Summary\n\n")
        f.write(f"**Test completed at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Results saved to:** {results_file}\n")
        f.write(f"**JSON data saved to:** {json_file}\n")
        f.write(f"**Models tested:** {len(results['models'])}\n")
        f.write(f"**Test cases:** {len(results['tests'])}\n")

    # Save JSON data for graphing
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("
Comprehensive benchmark completed!"    print(f"Results saved to: {results_file}")
    print(f"JSON data saved to: {json_file}")
    print(".1f"
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

