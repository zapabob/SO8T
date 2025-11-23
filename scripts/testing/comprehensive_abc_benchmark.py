import subprocess
import time
import csv
import re
import os

# Models to test
MODELS = {
    "Model_A_Base": "model-a:q8_0",
    # "AGIASI_Golden": "agiasi-phi35-golden-sigmoid:q8_0",
    "AEGIS_0.6": "aegis-adjusted-0.6"
}

# Test Prompts (Tasks 1-5)
TASKS = [
    {"id": "T1", "q": "Natalia sold clips to 48 friends in April, and then half as many in May. How many did she sell in total?"},
    {"id": "T2", "q": "I have 2 blue robes and 1 white robe. How many bolts of cloth do I need if one robe takes 1 bolt?"},
    {"id": "T3", "q": "A house cost 80k. I spent 50k on repairs. I want to sell it for a 50% profit on my total investment. What is the selling price?"},
    {"id": "T4", "q": "40 chickens eat 40 cups of feed. How many cups do 120 chickens eat?"},
    {"id": "T5", "q": "A company produces 20 units/hour. They work 18 hours/day for 7 days. Total production?"}
]

def run_ollama(model, prompt):
    start_time = time.time()
    try:
        # ollama run command
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, timeout=60, encoding='utf-8' # Timeout extended for safety
        )
        duration = time.time() - start_time
        return result.stdout.strip(), duration
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 60.0
    except Exception as e:
        return f"ERROR: {e}", 0.0

def main():
    results = []
    print("🚀 Starting Comprehensive A/B/C Benchmark...")

    # Ensure output directory exists
    os.makedirs("_docs/benchmark_results", exist_ok=True)

    for model_name, model_id in MODELS.items():
        print(f"\n🧪 Testing Model: {model_name} ({model_id})")
        
        for task in TASKS:
            print(f"   - Running {task['id']}...", end="", flush=True)
            output, duration = run_ollama(model_id, task['q'])
            
            # 簡易的な回答長チェック (簡潔性の指標)
            length = len(output)
            
            results.append({
                "Model": model_name,
                "Task": task['id'],
                "Duration": round(duration, 2),
                "Length": length,
                "Output_Head": output[:50].replace("\n", " ") + "..." # ログ用
            })
            print(f" Done ({duration:.2f}s)")

    # Save to CSV
    csv_file = "_docs/benchmark_results/abc_test_results.csv"
    with open(csv_file, "w", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Task", "Duration", "Length", "Output_Head"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ Benchmark Complete! Results saved to {csv_file}")

if __name__ == "__main__":
    main()
