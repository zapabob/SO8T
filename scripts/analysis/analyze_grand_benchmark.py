#!/usr/bin/env python3
"""
Grand Benchmark Analysis
Analyzes massive-scale benchmark results (JSONL) and generates statistical reports.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Configuration
RESULTS_DIR = Path("_docs/grand_benchmark_results")
OUTPUT_DIR = Path("_docs/grand_benchmark_reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_results(benchmark_name: str) -> pd.DataFrame:
    """Load results from JSONL"""
    file_path = RESULTS_DIR / f"{benchmark_name}_results.jsonl"
    if not file_path.exists():
        print(f"Warning: {file_path} not found.")
        return pd.DataFrame()
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    if not data:
        return pd.DataFrame()

    # Flatten result dict
    flat_data = []
    for item in data:
        entry = {
            "model": item["model"],
            "id": item["id"],
            "correct": item["result"].get("correct", False),
            "prediction": item["result"].get("prediction"),
        }
        flat_data.append(entry)
        
    return pd.DataFrame(flat_data)

def calculate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate accuracy and 95% CI"""
    if df.empty:
        return pd.DataFrame()
        
    stats_data = []
    models = df['model'].unique()
    
    for model in models:
        model_df = df[df['model'] == model]
        n = len(model_df)
        k = model_df['correct'].sum()
        acc = k / n if n > 0 else 0
        
        # Wilson Score Interval for Binomial CI
        z = 1.96 # 95% confidence
        denominator = 1 + z**2/n
        center_adjusted_probability = acc + z**2 / (2*n)
        adjusted_standard_deviation = np.sqrt((acc*(1 - acc) + z**2 / (4*n)) / n)
        
        lower_bound = (center_adjusted_probability - z * adjusted_standard_deviation) / denominator
        upper_bound = (center_adjusted_probability + z * adjusted_standard_deviation) / denominator
        
        stats_data.append({
            "Model": model,
            "Accuracy": acc * 100,
            "N": n,
            "Correct": k,
            "CI_Lower": lower_bound * 100,
            "CI_Upper": upper_bound * 100,
            "Error_Bar": (upper_bound - lower_bound) * 100 / 2
        })
        
    return pd.DataFrame(stats_data)

def generate_report(benchmarks: list):
    """Generate Markdown report"""
    report_path = OUTPUT_DIR / "Grand_Benchmark_Report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Grand Benchmark Report\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        for benchmark in benchmarks:
            f.write(f"## {benchmark}\n\n")
            df = load_results(benchmark)
            if df.empty:
                f.write("No data found.\n\n")
                continue
                
            stats_df = calculate_stats(df)
            f.write(stats_df.to_markdown(index=False, floatfmt=".2f"))
            f.write("\n\n")
            
            # Plotting
            plt.figure(figsize=(10, 6))
            sns.barplot(data=stats_df, x='Model', y='Accuracy', yerr=stats_df['Error_Bar'], capsize=0.1)
            plt.title(f"{benchmark} Performance")
            plt.ylim(0, 100)
            plt.ylabel("Accuracy (%)")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{benchmark}_chart.png")
            plt.close()
            
            f.write(f"![{benchmark} Chart]({benchmark}_chart.png)\n\n")

def main():
    benchmarks = ["MMLU", "GSM8K", "ELYZA"]
    generate_report(benchmarks)
    print(f"Report generated at {OUTPUT_DIR / 'Grand_Benchmark_Report.md'}")

if __name__ == "__main__":
    main()
