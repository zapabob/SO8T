#!/usr/bin/env python3
"""
SWA比較分析スクリプト
SWAありとなしのトレーニング結果を比較して論文レベルの分析を生成
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import argparse
from tqdm import tqdm


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
                    print(f"JSON decode error at line {line_num}: {e}")
                    continue
    return logs


def plot_swa_comparison(swa_logs: List[Dict[str, Any]], no_swa_logs: List[Dict[str, Any]], output_dir: Path):
    """SWAありとなしの比較可視化"""
    # トレーニングログのみをフィルタリング
    swa_train_logs = [log for log in swa_logs if 'loss' in log and 'train_accuracy' in log]
    no_swa_train_logs = [log for log in no_swa_logs if 'loss' in log and 'train_accuracy' in log]
    
    # 検証ログ
    swa_val_logs = [log for log in swa_logs if 'val_loss' in log and 'val_accuracy' in log]
    no_swa_val_logs = [log for log in no_swa_logs if 'val_loss' in log and 'val_accuracy' in log]
    
    # データ抽出
    swa_steps = [log['step'] for log in swa_train_logs]
    swa_losses = [log['loss'] for log in swa_train_logs]
    swa_accuracies = [log['train_accuracy'] for log in swa_train_logs]
    swa_pet_losses = [log['pet_loss'] for log in swa_train_logs]
    
    no_swa_steps = [log['step'] for log in no_swa_train_logs]
    no_swa_losses = [log['loss'] for log in no_swa_train_logs]
    no_swa_accuracies = [log['train_accuracy'] for log in no_swa_train_logs]
    no_swa_pet_losses = [log['pet_loss'] for log in no_swa_train_logs]
    
    # 4つのサブプロットを作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SWA vs No-SWA Comparison: Anti-Local-Minimum Recovery', fontsize=16, fontweight='bold')
    
    # 1. Total Loss Comparison
    ax1.plot(swa_steps, swa_losses, 'b-', linewidth=1, alpha=0.7, label='With SWA')
    ax1.plot(no_swa_steps, no_swa_losses, 'r-', linewidth=1, alpha=0.7, label='No SWA')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.legend()
    
    # 2. Accuracy Comparison
    ax2.plot(swa_steps, swa_accuracies, 'b-', linewidth=1, alpha=0.7, label='With SWA')
    ax2.plot(no_swa_steps, no_swa_accuracies, 'r-', linewidth=1, alpha=0.7, label='No SWA')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='100% (Overfitting)')
    ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90% (Healthy)')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 1.05)
    ax2.legend()
    
    # 3. PET Loss Comparison
    ax3.plot(swa_steps, swa_pet_losses, 'b-', linewidth=1, alpha=0.7, label='With SWA')
    ax3.plot(no_swa_steps, no_swa_pet_losses, 'r-', linewidth=1, alpha=0.7, label='No SWA')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('PET Loss')
    ax3.set_title('PET Loss Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.legend()
    
    # 4. Loss Variance Comparison (局所安定解脱出の指標)
    window_size = 50
    
    def calculate_variance(steps, losses, window_size):
        if len(losses) <= window_size:
            return [], []
        variance = []
        variance_steps = []
        for i in range(window_size, len(losses)):
            window_losses = losses[i-window_size:i]
            variance.append(np.var(window_losses))
            variance_steps.append(steps[i])
        return variance_steps, variance
    
    swa_var_steps, swa_variance = calculate_variance(swa_steps, swa_losses, window_size)
    no_swa_var_steps, no_swa_variance = calculate_variance(no_swa_steps, no_swa_losses, window_size)
    
    if swa_var_steps and no_swa_var_steps:
        ax4.plot(swa_var_steps, swa_variance, 'b-', linewidth=1, alpha=0.7, label='With SWA')
        ax4.plot(no_swa_var_steps, no_swa_variance, 'r-', linewidth=1, alpha=0.7, label='No SWA')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Loss Variance')
        ax4.set_title('Loss Variance Comparison (Local Minimum Escape)')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for variance calculation', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Loss Variance Comparison (Insufficient Data)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'swa_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def analyze_swa_effect(swa_logs: List[Dict[str, Any]], no_swa_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """SWA効果を分析"""
    # トレーニングログのみをフィルタリング
    swa_train_logs = [log for log in swa_logs if 'loss' in log and 'train_accuracy' in log]
    no_swa_train_logs = [log for log in no_swa_logs if 'loss' in log and 'train_accuracy' in log]
    
    # 検証ログ
    swa_val_logs = [log for log in swa_logs if 'val_loss' in log and 'val_accuracy' in log]
    no_swa_val_logs = [log for log in no_swa_logs if 'val_loss' in log and 'val_accuracy' in log]
    
    # 最近の100ステップを分析
    swa_recent = swa_train_logs[-100:] if len(swa_train_logs) >= 100 else swa_train_logs
    no_swa_recent = no_swa_train_logs[-100:] if len(no_swa_train_logs) >= 100 else no_swa_train_logs
    
    # 基本統計
    swa_losses = [log['loss'] for log in swa_recent]
    swa_accuracies = [log['train_accuracy'] for log in swa_recent]
    swa_pet_losses = [log['pet_loss'] for log in swa_recent]
    
    no_swa_losses = [log['loss'] for log in no_swa_recent]
    no_swa_accuracies = [log['train_accuracy'] for log in no_swa_recent]
    no_swa_pet_losses = [log['pet_loss'] for log in no_swa_recent]
    
    # 分散計算
    swa_loss_var = np.var(swa_losses)
    swa_acc_var = np.var(swa_accuracies)
    swa_pet_var = np.var(swa_pet_losses)
    
    no_swa_loss_var = np.var(no_swa_losses)
    no_swa_acc_var = np.var(no_swa_accuracies)
    no_swa_pet_var = np.var(no_swa_pet_losses)
    
    # 汎化分析
    swa_gen_analysis = {}
    no_swa_gen_analysis = {}
    
    if swa_val_logs:
        final_swa_val = swa_val_logs[-1]
        final_swa_train = swa_recent[-1]
        swa_gen_analysis = {
            "val_loss": final_swa_val['val_loss'],
            "val_accuracy": final_swa_val['val_accuracy'],
            "train_loss": final_swa_train['loss'],
            "train_accuracy": final_swa_train['train_accuracy'],
            "generalization_gap_loss": abs(final_swa_val['val_loss'] - final_swa_train['loss']),
            "generalization_gap_accuracy": abs(final_swa_val['val_accuracy'] - final_swa_train['train_accuracy'])
        }
    
    if no_swa_val_logs:
        final_no_swa_val = no_swa_val_logs[-1]
        final_no_swa_train = no_swa_recent[-1]
        no_swa_gen_analysis = {
            "val_loss": final_no_swa_val['val_loss'],
            "val_accuracy": final_no_swa_val['val_accuracy'],
            "train_loss": final_no_swa_train['loss'],
            "train_accuracy": final_no_swa_train['train_accuracy'],
            "generalization_gap_loss": abs(final_no_swa_val['val_loss'] - final_no_swa_train['loss']),
            "generalization_gap_accuracy": abs(final_no_swa_val['val_accuracy'] - final_no_swa_train['train_accuracy'])
        }
    
    analysis = {
        "swa": {
            "total_steps": len(swa_train_logs),
            "recent_steps": len(swa_recent),
            "loss_variance": swa_loss_var,
            "accuracy_variance": swa_acc_var,
            "pet_variance": swa_pet_var,
            "mean_loss": np.mean(swa_losses),
            "mean_accuracy": np.mean(swa_accuracies),
            "mean_pet_loss": np.mean(swa_pet_losses),
            "min_accuracy": min(swa_accuracies),
            "max_accuracy": max(swa_accuracies),
            "generalization": swa_gen_analysis
        },
        "no_swa": {
            "total_steps": len(no_swa_train_logs),
            "recent_steps": len(no_swa_recent),
            "loss_variance": no_swa_loss_var,
            "accuracy_variance": no_swa_acc_var,
            "pet_variance": no_swa_pet_var,
            "mean_loss": np.mean(no_swa_losses),
            "mean_accuracy": np.mean(no_swa_accuracies),
            "mean_pet_loss": np.mean(no_swa_pet_losses),
            "min_accuracy": min(no_swa_accuracies),
            "max_accuracy": max(no_swa_accuracies),
            "generalization": no_swa_gen_analysis
        }
    }
    
    return analysis


def print_swa_comparison_report(analysis: Dict[str, Any]):
    """SWA比較レポートを表示"""
    print("=" * 80)
    print("SWA vs No-SWA Comparison Report")
    print("=" * 80)
    
    swa = analysis['swa']
    no_swa = analysis['no_swa']
    
    print("Training Statistics Comparison:")
    print(f"{'Metric':<25} {'With SWA':<15} {'No SWA':<15} {'Difference':<15}")
    print("-" * 70)
    
    # Loss Variance
    loss_var_diff = swa['loss_variance'] - no_swa['loss_variance']
    print(f"{'Loss Variance':<25} {swa['loss_variance']:<15.2e} {no_swa['loss_variance']:<15.2e} {loss_var_diff:<15.2e}")
    
    # Accuracy Variance
    acc_var_diff = swa['accuracy_variance'] - no_swa['accuracy_variance']
    print(f"{'Accuracy Variance':<25} {swa['accuracy_variance']:<15.4f} {no_swa['accuracy_variance']:<15.4f} {acc_var_diff:<15.4f}")
    
    # PET Variance
    pet_var_diff = swa['pet_variance'] - no_swa['pet_variance']
    print(f"{'PET Variance':<25} {swa['pet_variance']:<15.2e} {no_swa['pet_variance']:<15.2e} {pet_var_diff:<15.2e}")
    
    # Mean Accuracy
    acc_diff = swa['mean_accuracy'] - no_swa['mean_accuracy']
    print(f"{'Mean Accuracy':<25} {swa['mean_accuracy']:<15.4f} {no_swa['mean_accuracy']:<15.4f} {acc_diff:<15.4f}")
    
    # Accuracy Range
    swa_range = swa['max_accuracy'] - swa['min_accuracy']
    no_swa_range = no_swa['max_accuracy'] - no_swa['min_accuracy']
    range_diff = swa_range - no_swa_range
    print(f"{'Accuracy Range':<25} {swa_range:<15.4f} {no_swa_range:<15.4f} {range_diff:<15.4f}")
    
    print()
    
    # 汎化分析
    if swa['generalization'] and no_swa['generalization']:
        print("Generalization Analysis:")
        swa_gen = swa['generalization']
        no_swa_gen = no_swa['generalization']
        
        print(f"{'Metric':<25} {'With SWA':<15} {'No SWA':<15} {'Difference':<15}")
        print("-" * 70)
        
        gen_gap_acc_diff = swa_gen['generalization_gap_accuracy'] - no_swa_gen['generalization_gap_accuracy']
        print(f"{'Gen Gap (Accuracy)':<25} {swa_gen['generalization_gap_accuracy']:<15.4f} {no_swa_gen['generalization_gap_accuracy']:<15.4f} {gen_gap_acc_diff:<15.4f}")
        
        gen_gap_loss_diff = swa_gen['generalization_gap_loss'] - no_swa_gen['generalization_gap_loss']
        print(f"{'Gen Gap (Loss)':<25} {swa_gen['generalization_gap_loss']:<15.2e} {no_swa_gen['generalization_gap_loss']:<15.2e} {gen_gap_loss_diff:<15.2e}")
        
        print()
    
    # SWA効果の評価
    print("SWA Effect Analysis:")
    
    # Loss Variance (高い方が良い - 局所安定解脱出)
    if swa['loss_variance'] > no_swa['loss_variance']:
        print("  ✅ SWA increases loss variance (better local minimum escape)")
    else:
        print("  ❌ SWA decreases loss variance (worse local minimum escape)")
    
    # Accuracy Variance (適度な方が良い)
    if 0.001 < swa['accuracy_variance'] < 0.01 and 0.001 < no_swa['accuracy_variance'] < 0.01:
        print("  ✅ Both show healthy accuracy variance")
    elif swa['accuracy_variance'] < no_swa['accuracy_variance']:
        print("  ✅ SWA reduces overfitting (lower accuracy variance)")
    else:
        print("  ⚠️  SWA increases accuracy variance")
    
    # Accuracy Range (広い方が良い - 柔軟性)
    if swa_range > no_swa_range:
        print("  ✅ SWA increases accuracy range (more flexibility)")
    else:
        print("  ❌ SWA decreases accuracy range (less flexibility)")
    
    # 汎化ギャップ
    if swa['generalization'] and no_swa['generalization']:
        if swa_gen['generalization_gap_accuracy'] < no_swa_gen['generalization_gap_accuracy']:
            print("  ✅ SWA improves generalization (smaller gap)")
        else:
            print("  ❌ SWA worsens generalization (larger gap)")
    
    print()
    print("Conclusion:")
    if (swa['loss_variance'] > no_swa['loss_variance'] and 
        swa['accuracy_variance'] < no_swa['accuracy_variance'] and
        swa_range > no_swa_range):
        print("  🎉 SWA provides better local minimum escape and generalization!")
    elif (swa['loss_variance'] > no_swa['loss_variance'] and 
          swa['accuracy_variance'] < no_swa['accuracy_variance']):
        print("  ✅ SWA provides some benefits for local minimum escape")
    else:
        print("  ⚠️  SWA effects are mixed or negative")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare SWA vs No-SWA training results")
    parser.add_argument("--swa_log", type=Path, default=Path("chk/so8t_default_train_log.jsonl"), 
                       help="Path to SWA training log file")
    parser.add_argument("--no_swa_log", type=Path, default=Path("chk/no_swa_train_log.jsonl"), 
                       help="Path to No-SWA training log file")
    parser.add_argument("--output_dir", type=Path, default=Path("visualizations"), 
                       help="Output directory for plots")
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    args.output_dir.mkdir(exist_ok=True)
    
    print("Loading training logs...")
    swa_logs = load_training_log(args.swa_log)
    no_swa_logs = load_training_log(args.no_swa_log)
    
    if not swa_logs or not no_swa_logs:
        print("Error: Could not load training logs!")
        return
    
    print(f"Loaded {len(swa_logs)} SWA steps and {len(no_swa_logs)} No-SWA steps")
    
    # 分析実行
    print("\nAnalyzing SWA effects...")
    analysis = analyze_swa_effect(swa_logs, no_swa_logs)
    print_swa_comparison_report(analysis)
    
    # 可視化
    print("\nGenerating comparison visualizations...")
    plot_swa_comparison(swa_logs, no_swa_logs, args.output_dir)
    
    print(f"\nComparison analysis complete! Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
