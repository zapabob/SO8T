#!/usr/bin/env python3
"""
PET Schedule Experiment Script
λ_petの3ステージ境界を動かす実験で最適なタイミングを探索
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
import seaborn as sns
from itertools import product


def load_training_log(log_path: Path) -> List[Dict[str, Any]]:
    """トレーニングログを読み込む"""
    logs = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    log_entry = json.loads(line.strip())
                    logs.append(log_entry)
                except json.JSONDecodeError as e:
                    continue
    return logs


def simulate_pet_schedule(total_steps: int, transition1: float, transition2: float) -> List[float]:
    """PETスケジュールをシミュレート"""
    pet_lambdas = []
    for step in range(total_steps):
        progress = step / total_steps
        if progress < transition1:
            pet_lambda = 0.01  # 探索期
        elif progress < transition2:
            pet_lambda = 0.1   # 遷移期
        else:
            pet_lambda = 1.0   # 安定化期
        pet_lambdas.append(pet_lambda)
    return pet_lambdas


def analyze_schedule_effect(logs: List[Dict[str, Any]], transition1: float, transition2: float) -> Dict[str, Any]:
    """スケジュール効果を分析"""
    train_logs = [log for log in logs if 'loss' in log and 'train_accuracy' in log]
    
    if not train_logs:
        return {"error": "No training data available"}
    
    total_steps = max([log['step'] for log in train_logs])
    pet_lambdas = simulate_pet_schedule(total_steps, transition1, transition2)
    
    # 各フェーズでの統計を計算
    exploration_steps = int(total_steps * transition1)
    transition_steps = int(total_steps * transition2)
    
    exploration_logs = [log for log in train_logs if log['step'] < exploration_steps]
    transition_logs = [log for log in train_logs if exploration_steps <= log['step'] < transition_steps]
    stabilization_logs = [log for log in train_logs if log['step'] >= transition_steps]
    
    def calculate_phase_stats(phase_logs):
        if not phase_logs:
            return {"count": 0, "mean_loss": 0, "mean_accuracy": 0, "mean_pet_loss": 0, "loss_variance": 0}
        
        losses = [log['loss'] for log in phase_logs]
        accuracies = [log['train_accuracy'] for log in phase_logs]
        pet_losses = [log['pet_loss'] for log in phase_logs]
        
        return {
            "count": len(phase_logs),
            "mean_loss": np.mean(losses),
            "mean_accuracy": np.mean(accuracies),
            "mean_pet_loss": np.mean(pet_losses),
            "loss_variance": np.var(losses),
            "accuracy_variance": np.var(accuracies),
            "pet_variance": np.var(pet_losses)
        }
    
    exploration_stats = calculate_phase_stats(exploration_logs)
    transition_stats = calculate_phase_stats(transition_logs)
    stabilization_stats = calculate_phase_stats(stabilization_logs)
    
    # 全体の統計
    all_losses = [log['loss'] for log in train_logs]
    all_accuracies = [log['train_accuracy'] for log in train_logs]
    all_pet_losses = [log['pet_loss'] for log in train_logs]
    
    return {
        "transition1": transition1,
        "transition2": transition2,
        "total_steps": total_steps,
        "exploration": exploration_stats,
        "transition": transition_stats,
        "stabilization": stabilization_stats,
        "overall": {
            "mean_loss": np.mean(all_losses),
            "mean_accuracy": np.mean(all_accuracies),
            "mean_pet_loss": np.mean(all_pet_losses),
            "loss_variance": np.var(all_losses),
            "accuracy_variance": np.var(all_accuracies),
            "pet_variance": np.var(all_pet_losses)
        }
    }


def run_pet_schedule_experiment(logs: List[Dict[str, Any]], output_dir: Path):
    """PETスケジュール実験を実行"""
    # 異なる境界設定をテスト
    transition1_options = [0.2, 0.3, 0.4]  # 20%, 30%, 40%
    transition2_options = [0.6, 0.7, 0.8]  # 60%, 70%, 80%
    
    results = []
    
    print("Running PET Schedule Experiment...")
    print(f"Testing {len(transition1_options)} x {len(transition2_options)} = {len(transition1_options) * len(transition2_options)} combinations")
    
    for t1, t2 in product(transition1_options, transition2_options):
        if t1 >= t2:
            continue  # 無効な組み合わせをスキップ
        
        print(f"Testing transition1={t1:.1f}, transition2={t2:.1f}")
        result = analyze_schedule_effect(logs, t1, t2)
        if "error" not in result:
            results.append(result)
    
    return results


def plot_schedule_comparison(results: List[Dict[str, Any]], output_dir: Path):
    """スケジュール比較を可視化"""
    if not results:
        print("No results to plot!")
        return
    
    # 結果を整理
    t1_values = sorted(list(set([r['transition1'] for r in results])))
    t2_values = sorted(list(set([r['transition2'] for r in results])))
    
    # ヒートマップ用のデータを準備
    loss_variance_matrix = np.zeros((len(t1_values), len(t2_values)))
    accuracy_variance_matrix = np.zeros((len(t1_values), len(t2_values)))
    pet_variance_matrix = np.zeros((len(t1_values), len(t2_values)))
    
    for result in results:
        i = t1_values.index(result['transition1'])
        j = t2_values.index(result['transition2'])
        loss_variance_matrix[i, j] = result['overall']['loss_variance']
        accuracy_variance_matrix[i, j] = result['overall']['accuracy_variance']
        pet_variance_matrix[i, j] = result['overall']['pet_variance']
    
    # 3つのヒートマップを作成
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('PET Schedule Experiment: Transition Boundary Effects', fontsize=16, fontweight='bold')
    
    # Loss Variance ヒートマップ
    sns.heatmap(loss_variance_matrix, 
                xticklabels=[f'{t2:.1f}' for t2 in t2_values],
                yticklabels=[f'{t1:.1f}' for t1 in t1_values],
                annot=True, fmt='.2e', cmap='viridis', ax=ax1)
    ax1.set_title('Loss Variance (Higher = Better Local Minimum Escape)')
    ax1.set_xlabel('Transition 2 (Stabilization Start)')
    ax1.set_ylabel('Transition 1 (Exploration End)')
    
    # Accuracy Variance ヒートマップ
    sns.heatmap(accuracy_variance_matrix, 
                xticklabels=[f'{t2:.1f}' for t2 in t2_values],
                yticklabels=[f'{t1:.1f}' for t1 in t1_values],
                annot=True, fmt='.4f', cmap='plasma', ax=ax2)
    ax2.set_title('Accuracy Variance (Moderate = Healthy)')
    ax2.set_xlabel('Transition 2 (Stabilization Start)')
    ax2.set_ylabel('Transition 1 (Exploration End)')
    
    # PET Variance ヒートマップ
    sns.heatmap(pet_variance_matrix, 
                xticklabels=[f'{t2:.1f}' for t2 in t2_values],
                yticklabels=[f'{t1:.1f}' for t1 in t1_values],
                annot=True, fmt='.2e', cmap='coolwarm', ax=ax3)
    ax3.set_title('PET Loss Variance (Higher = More Active)')
    ax3.set_xlabel('Transition 2 (Stabilization Start)')
    ax3.set_ylabel('Transition 1 (Exploration End)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pet_schedule_experiment.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'pet_schedule_experiment.pdf', bbox_inches='tight')
    plt.show()
    
    return fig


def plot_phase_analysis(results: List[Dict[str, Any]], output_dir: Path):
    """フェーズ別分析を可視化"""
    if not results:
        return
    
    # 最適なスケジュールを見つける（Loss Varianceが最も高いもの）
    best_result = max(results, key=lambda r: r['overall']['loss_variance'])
    
    print(f"Best schedule: transition1={best_result['transition1']:.1f}, transition2={best_result['transition2']:.1f}")
    
    # フェーズ別統計を可視化
    phases = ['Exploration', 'Transition', 'Stabilization']
    phase_data = [best_result['exploration'], best_result['transition'], best_result['stabilization']]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Phase Analysis: Best Schedule (t1={best_result["transition1"]:.1f}, t2={best_result["transition2"]:.1f})', 
                 fontsize=16, fontweight='bold')
    
    # Mean Loss by Phase
    mean_losses = [phase['mean_loss'] for phase in phase_data]
    ax1.bar(phases, mean_losses, color=['blue', 'orange', 'green'], alpha=0.7)
    ax1.set_title('Mean Loss by Phase')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    
    # Mean Accuracy by Phase
    mean_accuracies = [phase['mean_accuracy'] for phase in phase_data]
    ax2.bar(phases, mean_accuracies, color=['blue', 'orange', 'green'], alpha=0.7)
    ax2.set_title('Mean Accuracy by Phase')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0.5, 1.0)
    
    # Loss Variance by Phase
    loss_variances = [phase['loss_variance'] for phase in phase_data]
    ax3.bar(phases, loss_variances, color=['blue', 'orange', 'green'], alpha=0.7)
    ax3.set_title('Loss Variance by Phase')
    ax3.set_ylabel('Loss Variance')
    ax3.set_yscale('log')
    
    # PET Loss by Phase
    mean_pet_losses = [phase['mean_pet_loss'] for phase in phase_data]
    ax4.bar(phases, mean_pet_losses, color=['blue', 'orange', 'green'], alpha=0.7)
    ax4.set_title('Mean PET Loss by Phase')
    ax4.set_ylabel('PET Loss')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'phase_analysis.pdf', bbox_inches='tight')
    plt.show()
    
    return fig


def create_experiment_report(results: List[Dict[str, Any]], output_dir: Path):
    """実験レポートを作成"""
    if not results:
        return
    
    # 最適なスケジュールを見つける
    best_result = max(results, key=lambda r: r['overall']['loss_variance'])
    
    report_file = output_dir / 'pet_schedule_experiment_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("PET Schedule Experiment Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total combinations tested: {len(results)}\n")
        f.write(f"Best schedule: transition1={best_result['transition1']:.1f}, transition2={best_result['transition2']:.1f}\n\n")
        
        f.write("Best Schedule Performance:\n")
        f.write(f"  Overall Loss Variance: {best_result['overall']['loss_variance']:.2e}\n")
        f.write(f"  Overall Accuracy Variance: {best_result['overall']['accuracy_variance']:.4f}\n")
        f.write(f"  Overall PET Variance: {best_result['overall']['pet_variance']:.2e}\n")
        f.write(f"  Mean Loss: {best_result['overall']['mean_loss']:.2e}\n")
        f.write(f"  Mean Accuracy: {best_result['overall']['mean_accuracy']:.4f}\n")
        f.write(f"  Mean PET Loss: {best_result['overall']['mean_pet_loss']:.4f}\n\n")
        
        f.write("Phase Analysis:\n")
        phases = ['Exploration', 'Transition', 'Stabilization']
        phase_data = [best_result['exploration'], best_result['transition'], best_result['stabilization']]
        
        for phase, data in zip(phases, phase_data):
            f.write(f"  {phase} Phase:\n")
            f.write(f"    Steps: {data['count']}\n")
            f.write(f"    Mean Loss: {data['mean_loss']:.2e}\n")
            f.write(f"    Mean Accuracy: {data['mean_accuracy']:.4f}\n")
            f.write(f"    Mean PET Loss: {data['mean_pet_loss']:.4f}\n")
            f.write(f"    Loss Variance: {data['loss_variance']:.2e}\n")
            f.write(f"    Accuracy Variance: {data['accuracy_variance']:.4f}\n")
            f.write(f"    PET Variance: {data['pet_variance']:.2e}\n\n")
        
        f.write("All Results Summary:\n")
        f.write(f"{'T1':<6} {'T2':<6} {'Loss Var':<12} {'Acc Var':<10} {'PET Var':<12}\n")
        f.write("-" * 50 + "\n")
        
        for result in sorted(results, key=lambda r: r['overall']['loss_variance'], reverse=True):
            f.write(f"{result['transition1']:<6.1f} {result['transition2']:<6.1f} "
                   f"{result['overall']['loss_variance']:<12.2e} "
                   f"{result['overall']['accuracy_variance']:<10.4f} "
                   f"{result['overall']['pet_variance']:<12.2e}\n")
    
    print(f"Experiment report saved to: {report_file}")
    return report_file


def main():
    parser = argparse.ArgumentParser(description="PET Schedule Experiment")
    parser.add_argument("--log_file", type=Path, default=Path("chk/so8t_default_train_log.jsonl"), 
                       help="Path to training log file")
    parser.add_argument("--output_dir", type=Path, default=Path("pet_experiments"), 
                       help="Output directory for results")
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    args.output_dir.mkdir(exist_ok=True)
    
    print("Loading training logs...")
    logs = load_training_log(args.log_file)
    
    if not logs:
        print("No training logs found!")
        return
    
    print(f"Loaded {len(logs)} training steps")
    
    # 実験実行
    results = run_pet_schedule_experiment(logs, args.output_dir)
    
    if not results:
        print("No valid results generated!")
        return
    
    # 可視化
    print("\nGenerating visualizations...")
    plot_schedule_comparison(results, args.output_dir)
    plot_phase_analysis(results, args.output_dir)
    
    # レポート作成
    print("Creating experiment report...")
    create_experiment_report(results, args.output_dir)
    
    print(f"\nPET Schedule Experiment complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
