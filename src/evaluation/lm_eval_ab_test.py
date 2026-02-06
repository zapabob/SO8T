#!/usr/bin/env python3
"""A/B Test using lm-evaluation-harness for SO8T models"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def run_lm_eval(model_name, model_type, model_args, tasks, output_dir, limit=None):
    """Run lm-evaluation-harness for a single model"""

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", model_type,
        "--model_args", model_args,
        "--tasks", tasks,
        "--batch_size", "auto",
        "--device", "cuda:0"
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=LM_EVAL_DIR, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running lm_eval for {model_name}:")
        print(result.stderr)
        return None

    # Save the output to a text file
    output_file = output_dir / "lm_eval_output.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result.stdout)

    return result.stdout

def parse_results(output_dir, model_name):
    """Parse lm-evaluation-harness results from text output"""
    output_file = output_dir / "lm_eval_output.txt"

    if not output_file.exists():
        print(f"Output file not found: {output_file}")
        return None

    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse the table format from lm_eval output
    lines = content.split('\n')
    parsed_results = {}

    # Find the results table
    in_table = False
    for line in lines:
        if 'Tasks' in line and 'Version' in line and 'Metric' in line:
            in_table = True
            continue
        elif in_table and line.strip() and not line.startswith('|') and not line.startswith('-'):
            # Parse result line
            parts = line.split('|')
            if len(parts) >= 6:
                task_name = parts[1].strip()
                metric = parts[5].strip()
                value = parts[7].strip()
                stderr = parts[9].strip() if len(parts) > 9 else ""

                if task_name and metric:
                    if task_name not in parsed_results:
                        parsed_results[task_name] = {}

                    try:
                        value_float = float(value)
                        parsed_results[task_name][metric] = value_float
                    except ValueError:
                        pass

    return parsed_results

def create_comparison_table(model_a_results, model_b_results, tasks):
    """Create comparison table between two models"""
    comparison_data = []

    for task in tasks:
        task_clean = task.replace(',', '').strip()

        a_score = None
        b_score = None

        # Find the actual task name in results
        for result_key in model_a_results.keys():
            if task_clean in result_key:
                # Look for accuracy metrics
                metrics = model_a_results[result_key]
                for metric_key in ['acc', 'acc_norm', 'accuracy', 'exact_match']:
                    if metric_key in metrics:
                        a_score = metrics[metric_key]
                        break

        for result_key in model_b_results.keys():
            if task_clean in result_key:
                metrics = model_b_results[result_key]
                for metric_key in ['acc', 'acc_norm', 'accuracy', 'exact_match']:
                    if metric_key in metrics:
                        b_score = metrics[metric_key]
                        break

        comparison_data.append({
            'task': task_clean,
            'model_a_score': a_score,
            'model_b_score': b_score,
            'improvement': (b_score - a_score) if (a_score is not None and b_score is not None) else None
        })

    return pd.DataFrame(comparison_data)

def create_visualization(df, model_a_name, model_b_name, output_dir):
    """Create visualization of results"""
    # Filter out None values
    df_plot = df.dropna()

    if df_plot.empty:
        print("No valid data for visualization")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar chart comparison
    x = np.arange(len(df_plot))
    width = 0.35

    ax1.bar(x - width/2, df_plot['model_a_score'], width, label=model_a_name, alpha=0.8)
    ax1.bar(x + width/2, df_plot['model_b_score'], width, label=model_b_name, alpha=0.8)

    ax1.set_xlabel('Tasks')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_plot['task'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Improvement plot
    colors = ['green' if x > 0 else 'red' for x in df_plot['improvement']]
    ax2.bar(df_plot['task'], df_plot['improvement'], color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Tasks')
    ax2.set_ylabel('Improvement (Model B - Model A)')
    ax2.set_title('Performance Improvement')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'ab_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Statistical summary
    valid_improvements = df_plot['improvement'].dropna()
    if len(valid_improvements) > 1:
        mean_improvement = valid_improvements.mean()
        std_improvement = valid_improvements.std()
        t_stat, p_value = stats.ttest_1samp(valid_improvements, 0)

        stats_summary = {
            'mean_improvement': mean_improvement,
            'std_improvement': std_improvement,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'sample_size': len(valid_improvements)
        }

        with open(output_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(stats_summary, f, indent=2)

        print("\nStatistical Analysis:")
        print(f"Mean improvement: {stats_summary['mean_improvement']:.4f}")
        print(f"Std improvement: {stats_summary['std_improvement']:.4f}")
        print(f"T-statistic: {stats_summary['t_statistic']:.4f}")
        print(f"P-value: {stats_summary['p_value']:.6f}")
        print(f"Significant improvement: {stats_summary['significant']}")

def create_hf_dataset_format(df, model_a_name, model_b_name, output_dir):
    """Create HuggingFace dataset format for results"""
    # Create dataset structure
    dataset = {
        'task': df['task'].tolist(),
        'model_a_score': df['model_a_score'].tolist(),
        'model_b_score': df['model_b_score'].tolist(),
        'improvement': df['improvement'].tolist(),
        'model_a_name': model_a_name,
        'model_b_name': model_b_name,
        'timestamp': datetime.now().isoformat()
    }

    # Save as JSON
    with open(output_dir / 'ab_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Create markdown summary
    summary = f"""# SO(8) A/B Test Results

## Overview
- **Model A (Baseline)**: {model_a_name}
- **Model B (SO(8) Enhanced)**: {model_b_name}
- **Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Results Summary

| Task | Model A Score | Model B Score | Improvement |
|------|---------------|---------------|-------------|
"""

    for _, row in df.iterrows():
        summary += f"| {row['task']} | {row['model_a_score']:.4f} | {row['model_b_score']:.4f} | {row['improvement']:+.4f} |\n"

    # Add statistical summary if available
    stats_file = output_dir / 'statistical_analysis.json'
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        summary += ".4f"".4f"".4f"".6f"
        summary += f"- **Significant Improvement**: {stats['significant']}\n"

    summary += "\n## Conclusion\n"

    mean_improvement = df['improvement'].mean()
    if mean_improvement > 0:
        summary += f"SO(8) enhancement shows positive improvement with average score increase of {mean_improvement:.4f}.\n"
    else:
        summary += f"SO(8) enhancement did not show improvement, with average score change of {mean_improvement:.4f}.\n"

    with open(output_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(summary)

def main():
    parser = argparse.ArgumentParser(description="SO(8) A/B Test using lm-evaluation-harness")
    parser.add_argument('--model_a', type=str, required=True,
                       help='Model A identifier (HF path)')
    parser.add_argument('--model_b', type=str, required=True,
                       help='Model B identifier (HF path)')
    parser.add_argument('--model_a_name', type=str, default='Model A',
                       help='Display name for Model A')
    parser.add_argument('--model_b_name', type=str, default='Model B (SO8T)',
                       help='Display name for Model B')
    parser.add_argument('--tasks', type=str, default='hellaswag,mmlu',
                       help='Comma-separated list of tasks')
    parser.add_argument('--output_dir', type=str, default='lm_eval_results',
                       help='Output directory')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of examples per task')

    args = parser.parse_args()

    # Setup paths
    global LM_EVAL_DIR
    LM_EVAL_DIR = Path(__file__).parent.parent.parent / 'lm-evaluation-harness'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model configurations
    model_configs = {
        'a': {
            'name': args.model_a,
            'display_name': args.model_a_name,
            'path': args.model_a
        },
        'b': {
            'name': args.model_b,
            'display_name': args.model_b_name,
            'path': args.model_b
        }
    }

    results = {}

    # Run evaluation for each model
    for model_key, config in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Evaluating Model {model_key.upper()}: {config['name']}")
        print(f"{'='*50}")

        model_output_dir = output_dir / f"model_{model_key}"
        model_output_dir.mkdir(exist_ok=True)

        # Prepare model arguments
        model_args = f"pretrained={config['path']}"

        # Run evaluation
        run_lm_eval(config['name'], 'hf', model_args,
                   args.tasks, model_output_dir, args.limit)

        # Parse results
        results[model_key] = parse_results(model_output_dir, config['name'])

        if results[model_key] is None:
            print(f"Failed to get results for Model {model_key}")
            continue

        print(f"Model {model_key} evaluation completed")

    # Create comparison
    if results['a'] and results['b']:
        print(f"\n{'='*50}")
        print("Creating A/B Test Comparison")
        print(f"{'='*50}")

        tasks_list = args.tasks.split(',')
        df = create_comparison_table(results['a'], results['b'], tasks_list)

        # Save comparison table
        df.to_csv(output_dir / 'comparison_table.csv', index=False)

        # Create visualizations
        create_visualization(df, args.model_a_name, args.model_b_name, output_dir)

        # Create HF dataset format
        create_hf_dataset_format(df, args.model_a_name, args.model_b_name, output_dir)

        print("\nComparison completed!")
        print(f"Results saved to: {output_dir}")

        # Print summary
        print("\nSummary:")
        valid_results = df.dropna()
        if not valid_results.empty:
            mean_improvement = valid_results['improvement'].mean()
            print(f"Mean improvement: {mean_improvement:.4f}")
            if mean_improvement > 0:
                print(f"[OK] {args.model_b_name} shows improvement over {args.model_a_name}!")
            else:
                print(f"[NG] {args.model_b_name} did not show improvement over {args.model_a_name}.")

    print("\nA/B test completed successfully!")

if __name__ == "__main__":
    main()
