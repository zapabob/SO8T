#!/usr/bin/env python3
"""
SO8T Training Visualization Script
局所安定解対策の効果を可視化して復興状況を確認する
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
                    print(f"Problematic line: {line.strip()}")
                    continue
    return logs


def plot_training_curves(logs: List[Dict[str, Any]], output_dir: Path):
    """トレーニングカーブを可視化"""
    # トレーニングログと検証ログを分離
    train_logs = [log for log in logs if 'loss' in log and 'train_accuracy' in log]
    val_logs = [log for log in logs if 'val_loss' in log and 'val_accuracy' in log]
    
    steps = [log['step'] for log in train_logs]
    losses = [log['loss'] for log in train_logs]
    ce_losses = [log['ce_loss'] for log in train_logs]
    pet_losses = [log['pet_loss'] for log in train_logs]
    accuracies = [log['train_accuracy'] for log in train_logs]
    
    # 検証データ
    val_steps = [log['step'] for log in val_logs]
    val_losses = [log['val_loss'] for log in val_logs]
    val_accuracies = [log['val_accuracy'] for log in val_logs]
    
    # 4つのサブプロットを作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SO8T Training Progress - Anti-Local-Minimum Recovery', fontsize=16, fontweight='bold')
    
    # 1. Total Loss
    ax1.plot(steps, losses, 'b-', linewidth=1, alpha=0.7, label='Total Loss')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss (Anti-Local-Minimum Effect)')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.legend()
    
    # 2. Loss Components
    ax2.plot(steps, ce_losses, 'r-', linewidth=1, alpha=0.7, label='Cross-Entropy Loss')
    ax2.plot(steps, pet_losses, 'g-', linewidth=1, alpha=0.7, label='PET Loss')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Components (CE + PET)')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.legend()
    
    # 3. Accuracy (Train vs Val)
    ax3.plot(steps, accuracies, 'purple', linewidth=1, alpha=0.7, label='Training Accuracy')
    if val_steps:
        ax3.plot(val_steps, val_accuracies, 'blue', linewidth=2, alpha=0.8, label='Validation Accuracy', marker='o', markersize=4)
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='100% (Overfitting)')
    ax3.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90% (Healthy)')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Training vs Validation Accuracy (Generalization)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.5, 1.05)
    ax3.legend()
    
    # 4. Loss Variance (局所安定解脱出の指標)
    window_size = 50
    if len(losses) > window_size:
        loss_variance = []
        for i in range(window_size, len(losses)):
            window_losses = losses[i-window_size:i]
            variance = np.var(window_losses)
            loss_variance.append(variance)
        
        ax4.plot(steps[window_size:], loss_variance, 'orange', linewidth=1, alpha=0.7, label='Loss Variance (50-step window)')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Variance')
        ax4.set_title('Loss Variance (Local Minimum Escape Indicator)')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Not enough data for variance calculation', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Loss Variance (Insufficient Data)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_pet_schedule_effect(logs: List[Dict[str, Any]], output_dir: Path):
    """PETスケジュールの効果を可視化"""
    # トレーニングログのみをフィルタリング
    train_logs = [log for log in logs if 'loss' in log and 'train_accuracy' in log]
    
    steps = [log['step'] for log in train_logs]
    pet_losses = [log['pet_loss'] for log in train_logs]
    
    # PETスケジュールを計算
    total_steps = max(steps)
    pet_lambdas = []
    for step in steps:
        progress = step / total_steps
        if progress < 0.3:
            pet_lambda = 0.01
        elif progress < 0.7:
            pet_lambda = 0.1
        else:
            pet_lambda = 1.0
        pet_lambdas.append(pet_lambda)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('PET Schedule Effect Analysis', fontsize=14, fontweight='bold')
    
    # PET Lambda Schedule
    ax1.plot(steps, pet_lambdas, 'b-', linewidth=2, label='PET Lambda Schedule')
    ax1.axvline(x=total_steps * 0.3, color='red', linestyle='--', alpha=0.7, label='30% (Exploration → Transition)')
    ax1.axvline(x=total_steps * 0.7, color='green', linestyle='--', alpha=0.7, label='70% (Transition → Stabilization)')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('PET Lambda')
    ax1.set_title('PET Schedule: Exploration → Transition → Stabilization')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # PET Loss vs Schedule
    ax2.plot(steps, pet_losses, 'g-', linewidth=1, alpha=0.7, label='PET Loss')
    ax2.axvline(x=total_steps * 0.3, color='red', linestyle='--', alpha=0.7, label='30%')
    ax2.axvline(x=total_steps * 0.7, color='green', linestyle='--', alpha=0.7, label='70%')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('PET Loss')
    ax2.set_title('PET Loss Response to Schedule')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pet_schedule_effect.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def analyze_recovery_status(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """復興状況を分析"""
    if not logs:
        return {"status": "No data available"}
    
    # トレーニングログと検証ログを分離
    train_logs = [log for log in logs if 'loss' in log and 'train_accuracy' in log]
    val_logs = [log for log in logs if 'val_loss' in log and 'val_accuracy' in log]
    
    recent_logs = train_logs[-100:]  # 最近の100ステップを分析
    
    # デバッグ: recent_logsの内容を確認
    print(f"Recent logs count: {len(recent_logs)}")
    if recent_logs:
        print(f"Last log entry keys: {list(recent_logs[-1].keys())}")
        print(f"Last log entry: {recent_logs[-1]}")
        
        # 各ログエントリのキーを確認
        for i, log in enumerate(recent_logs[:5]):  # 最初の5個を確認
            print(f"Log {i} keys: {list(log.keys())}")
            if 'loss' not in log:
                print(f"Log {i} missing 'loss' key: {log}")
    
    # 基本統計（安全に処理）
    recent_losses = []
    recent_accuracies = []
    recent_pet_losses = []
    
    for i, log in enumerate(recent_logs):
        try:
            recent_losses.append(log['loss'])
            recent_accuracies.append(log['train_accuracy'])
            recent_pet_losses.append(log['pet_loss'])
        except KeyError as e:
            print(f"KeyError at log {i}: {e}")
            print(f"Log content: {log}")
            continue
    
    # 局所安定解脱出の指標
    loss_variance = np.var(recent_losses)
    accuracy_variance = np.var(recent_accuracies)
    pet_variance = np.var(recent_pet_losses)
    
    # 復興状況の判定
    status = "Unknown"
    if loss_variance > 1e-6:  # Lossが動いている
        if accuracy_variance > 0.01:  # Accuracyが変動している
            if pet_variance > 0.1:  # PET Lossが変動している
                status = "Recovering - Anti-local-minimum measures working!"
            else:
                status = "Partially Recovering - Loss moving but PET stable"
        else:
            status = "Partially Recovering - Loss moving but accuracy stable"
    else:
        status = "Still Stuck - Local minimum persists"
    
    # 汎化分析
    generalization_analysis = {}
    if val_logs:
        val_losses = [log['val_loss'] for log in val_logs]
        val_accuracies = [log['val_accuracy'] for log in val_logs]
        val_steps = [log['step'] for log in val_logs]
        
        # 最後の検証データ
        final_val_loss = val_losses[-1] if val_losses else None
        final_val_accuracy = val_accuracies[-1] if val_accuracies else None
        
        # 汎化ギャップ（最後の検証データとトレーニングデータの比較）
        if recent_logs and final_val_loss is not None:
            final_train_loss = recent_logs[-1]['loss']
            final_train_accuracy = recent_logs[-1]['train_accuracy']
            
            generalization_analysis = {
                "final_val_loss": final_val_loss,
                "final_val_accuracy": final_val_accuracy,
                "final_train_loss": final_train_loss,
                "final_train_accuracy": final_train_accuracy,
                "generalization_gap_loss": abs(final_val_loss - final_train_loss),
                "generalization_gap_accuracy": abs(final_val_accuracy - final_train_accuracy),
                "val_data_points": len(val_logs),
                "val_steps": val_steps
            }
    
    analysis = {
        "status": status,
        "total_steps": len(train_logs),
        "recent_steps": len(recent_logs),
        "loss_variance": loss_variance,
        "accuracy_variance": accuracy_variance,
        "pet_variance": pet_variance,
        "mean_loss": np.mean(recent_losses),
        "mean_accuracy": np.mean(recent_accuracies),
        "mean_pet_loss": np.mean(recent_pet_losses),
        "min_accuracy": min(recent_accuracies),
        "max_accuracy": max(recent_accuracies),
        "generalization": generalization_analysis
    }
    
    return analysis


def print_recovery_report(analysis: Dict[str, Any]):
    """復興レポートを表示"""
    print("=" * 60)
    print("SO8T Training Recovery Status Report")
    print("=" * 60)
    print(f"Status: {analysis['status']}")
    print(f"Total Steps: {analysis['total_steps']}")
    print(f"Recent Steps Analyzed: {analysis['recent_steps']}")
    print()
    print("Variance Analysis (Higher = More Movement):")
    print(f"  Loss Variance: {analysis['loss_variance']:.2e}")
    print(f"  Accuracy Variance: {analysis['accuracy_variance']:.4f}")
    print(f"  PET Loss Variance: {analysis['pet_variance']:.4f}")
    print()
    print("Recent Performance:")
    print(f"  Mean Loss: {analysis['mean_loss']:.2e}")
    print(f"  Mean Accuracy: {analysis['mean_accuracy']:.4f}")
    print(f"  Accuracy Range: {analysis['min_accuracy']:.4f} - {analysis['max_accuracy']:.4f}")
    print(f"  Mean PET Loss: {analysis['mean_pet_loss']:.4f}")
    print()
    
    # 汎化分析レポート
    if analysis.get('generalization'):
        gen = analysis['generalization']
        print("Generalization Analysis:")
        print(f"  Validation Data Points: {gen['val_data_points']}")
        print(f"  Final Train Loss: {gen['final_train_loss']:.2e}")
        print(f"  Final Val Loss: {gen['final_val_loss']:.2e}")
        print(f"  Generalization Gap (Loss): {gen['generalization_gap_loss']:.2e}")
        print(f"  Final Train Accuracy: {gen['final_train_accuracy']:.4f}")
        print(f"  Final Val Accuracy: {gen['final_val_accuracy']:.4f}")
        print(f"  Generalization Gap (Accuracy): {gen['generalization_gap_accuracy']:.4f}")
        
        # 汎化ギャップの評価
        if gen['generalization_gap_accuracy'] < 0.1:
            print("  ✅ Good Generalization: Small accuracy gap")
        elif gen['generalization_gap_accuracy'] < 0.2:
            print("  ⚠️  Moderate Generalization: Medium accuracy gap")
        else:
            print("  ❌ Poor Generalization: Large accuracy gap")
        print()
    
    if "Recovering" in analysis['status']:
        print("🎉 SUCCESS: Anti-local-minimum measures are working!")
        print("   - Model is no longer stuck in local minimum")
        print("   - Loss is fluctuating (good sign)")
        print("   - Accuracy is varying (realistic)")
        print("   - PET regularization is active")
    elif "Partially" in analysis['status']:
        print("⚠️  PARTIAL SUCCESS: Some measures working")
        print("   - Some improvement detected")
        print("   - May need further tuning")
    else:
        print("❌ STILL STUCK: Local minimum persists")
        print("   - May need stronger interventions")
        print("   - Consider increasing noise levels")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Visualize SO8T training progress")
    parser.add_argument("--log_file", type=Path, default=Path("chk/so8t_default_train_log.jsonl"), 
                       help="Path to training log file")
    parser.add_argument("--output_dir", type=Path, default=Path("visualizations"), 
                       help="Output directory for plots")
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    args.output_dir.mkdir(exist_ok=True)
    
    print("Loading training logs...")
    logs = load_training_log(args.log_file)
    
    if not logs:
        print("No training logs found!")
        return
    
    print(f"Loaded {len(logs)} training steps")
    
    # 復興状況を分析
    print("\nAnalyzing recovery status...")
    analysis = analyze_recovery_status(logs)
    print_recovery_report(analysis)
    
    # 可視化
    print("\nGenerating visualizations...")
    plot_training_curves(logs, args.output_dir)
    plot_pet_schedule_effect(logs, args.output_dir)
    
    print(f"\nVisualizations saved to: {args.output_dir}")
    print("Training recovery analysis complete!")


if __name__ == "__main__":
    main()
