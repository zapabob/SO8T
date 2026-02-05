# visualize_quantization.py
"""Visualize benchmark results comparing quantization levels.

Usage:
    python visualize_quantization.py --results results.json
"""
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    
    data = json.loads(args.results.read_text(encoding="utf-8"))
    df = pd.DataFrame(data)
    
    # Extract quantization from model name if possible (e.g., "agiasi:q4" -> "q4")
    # This assumes model names are like "name:tag"
    df['quantization'] = df['model'].apply(lambda x: x.split(':')[-1] if ':' in x else x)
    
    # 1. Speed Comparison (Tokens/sec)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='quantization', y='tokens_per_sec', hue='model')
    plt.title('Inference Speed by Quantization Level')
    plt.ylabel('Tokens per Second')
    plt.xlabel('Quantization')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('benchmark_speed.png')
    print("Saved benchmark_speed.png")
    
    # 2. Response Length (Proxy for verbosity/completeness)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='quantization', y='eval_count', hue='model')
    plt.title('Response Length by Quantization Level')
    plt.ylabel('Generated Tokens')
    plt.xlabel('Quantization')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('benchmark_length.png')
    print("Saved benchmark_length.png")

    # 3. Qualitative Table (Save as Markdown)
    # Group by Test ID and show responses side-by-side
    md_output = "# Benchmark Qualitative Comparison\n\n"
    
    test_ids = df['test_id'].unique()
    for tid in test_ids:
        subset = df[df['test_id'] == tid]
        prompt = subset.iloc[0]['prompt']
        md_output += f"## Test: {tid}\n**Prompt**: {prompt}\n\n"
        
        for _, row in subset.iterrows():
            md_output += f"### {row['model']}\n{row['response']}\n\n"
        
        md_output += "---\n"
        
    Path("benchmark_report.md").write_text(md_output, encoding="utf-8")
    print("Saved benchmark_report.md")

if __name__ == "__main__":
    main()
