#!/usr/bin/env python3
"""
Safety Training Visualization
安全重視SO8Tの訓練結果を可視化する
"""

import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False


def load_training_log(log_file: Path) -> List[Dict[str, Any]]:
    """訓練ログを読み込み"""
    if not log_file.exists():
        raise FileNotFoundError(f"Training log file not found: {log_file}")
    
    entries = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping invalid JSON line: {e}")
                continue
    
    return entries


def create_training_curves_plot(entries: List[Dict[str, Any]], output_dir: Path):
    """訓練曲線を可視化"""
    if not entries:
        print("WARNING: No training data found")
        return
    
    # データをDataFrameに変換
    df = pd.DataFrame(entries)
    
    # エポック数を取得
    epochs = df['epoch'].values
    
    # 図を作成
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Safety-Aware SO8T Training Progress', fontsize=16, fontweight='bold')
    
    # 1. 損失曲線
    ax1 = axes[0, 0]
    ax1.plot(epochs, df['loss'], label='Train Loss', color='blue', alpha=0.7)
    ax1.plot(epochs, df['val_loss'], label='Val Loss', color='red', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 精度曲線
    ax2 = axes[0, 1]
    ax2.plot(epochs, df['accuracy'], label='Train Accuracy', color='blue', alpha=0.7)
    ax2.plot(epochs, df['val_accuracy'], label='Val Accuracy', color='red', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 安全スコア曲線（両系統）
    ax3 = axes[0, 2]
    if 'task_safety_score' in df.columns:
        ax3.plot(epochs, df['task_safety_score'], label='Task Safety Score', color='green', alpha=0.7)
    if 'safe_safety_score' in df.columns:
        ax3.plot(epochs, df['safe_safety_score'], label='Safety Score', color='orange', alpha=0.7)
    if 'combined_safety_score' in df.columns:
        ax3.plot(epochs, df['combined_safety_score'], label='Combined Safety Score', color='purple', alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Safety Score')
    ax3.set_title('Safety Scores (Dual System)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. REFUSE再現率
    ax4 = axes[1, 0]
    if 'task_refuse_recall' in df.columns:
        ax4.plot(epochs, df['task_refuse_recall'], label='Task REFUSE Recall', color='red', alpha=0.7)
    if 'safe_refuse_recall' in df.columns:
        ax4.plot(epochs, df['safe_refuse_recall'], label='Safety REFUSE Recall', color='darkred', alpha=0.7)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('REFUSE Recall')
    ax4.set_title('REFUSE Recall (Dual System)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. ESCALATE再現率
    ax5 = axes[1, 1]
    if 'task_escalate_recall' in df.columns:
        ax5.plot(epochs, df['task_escalate_recall'], label='Task ESCALATE Recall', color='blue', alpha=0.7)
    if 'safe_escalate_recall' in df.columns:
        ax5.plot(epochs, df['safe_escalate_recall'], label='Safety ESCALATE Recall', color='darkblue', alpha=0.7)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('ESCALATE Recall')
    ax5.set_title('ESCALATE Recall (Dual System)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 過度な従順率
    ax6 = axes[1, 2]
    if 'task_overcompliance_rate' in df.columns:
        ax6.plot(epochs, df['task_overcompliance_rate'], label='Task Overcompliance', color='orange', alpha=0.7)
    if 'safe_overcompliance_rate' in df.columns:
        ax6.plot(epochs, df['safe_overcompliance_rate'], label='Safety Overcompliance', color='darkorange', alpha=0.7)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Overcompliance Rate')
    ax6.set_title('Overcompliance Rate (Dual System)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / 'safety_training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'safety_training_curves.pdf', bbox_inches='tight')
    print(f"Training curves saved: {output_path}")
    
    plt.show()


def create_safety_metrics_heatmap(entries: List[Dict[str, Any]], output_dir: Path):
    """安全メトリクスのヒートマップを作成"""
    if not entries:
        print("WARNING: No training data found")
        return
    
    # 最新のエントリを取得
    latest = entries[-1]
    
    # タスク系統のメトリクス
    task_metrics = {
        'REFUSE Recall': latest.get('task_refuse_recall', 0),
        'ESCALATE Recall': latest.get('task_escalate_recall', 0),
        'Overcompliance Rate': latest.get('task_overcompliance_rate', 0),
        'Safety Score': latest.get('task_safety_score', 0)
    }
    
    # 安全系統のメトリクス
    safety_metrics = {
        'REFUSE Recall': latest.get('safe_refuse_recall', 0),
        'ESCALATE Recall': latest.get('safe_escalate_recall', 0),
        'Overcompliance Rate': latest.get('safe_overcompliance_rate', 0),
        'Safety Score': latest.get('safe_safety_score', 0)
    }
    
    # ヒートマップ用のデータを作成
    heatmap_data = pd.DataFrame({
        'Task System': list(task_metrics.values()),
        'Safety System': list(safety_metrics.values())
    }, index=list(task_metrics.keys()))
    
    # ヒートマップを作成
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
    plt.title('Safety Metrics Comparison (Latest Epoch)', fontsize=14, fontweight='bold')
    plt.xlabel('System Type')
    plt.ylabel('Metrics')
    
    # 保存
    output_path = output_dir / 'safety_metrics_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'safety_metrics_heatmap.pdf', bbox_inches='tight')
    print(f"Safety metrics heatmap saved: {output_path}")
    
    plt.show()


def create_safety_improvement_plot(entries: List[Dict[str, Any]], output_dir: Path):
    """安全改善の推移を可視化"""
    if not entries:
        print("WARNING: No training data found")
        return
    
    df = pd.DataFrame(entries)
    epochs = df['epoch'].values
    
    # 図を作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Safety Improvement Over Time', fontsize=16, fontweight='bold')
    
    # 1. 安全スコアの改善
    ax1 = axes[0, 0]
    if 'combined_safety_score' in df.columns:
        ax1.plot(epochs, df['combined_safety_score'], label='Combined Safety Score', color='purple', linewidth=2)
    if 'task_safety_score' in df.columns:
        ax1.plot(epochs, df['task_safety_score'], label='Task Safety Score', color='green', alpha=0.7)
    if 'safe_safety_score' in df.columns:
        ax1.plot(epochs, df['safe_safety_score'], label='Safety Score', color='orange', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Safety Score')
    ax1.set_title('Safety Score Improvement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. REFUSE能力の向上
    ax2 = axes[0, 1]
    if 'task_refuse_recall' in df.columns:
        ax2.plot(epochs, df['task_refuse_recall'], label='Task REFUSE', color='red', linewidth=2)
    if 'safe_refuse_recall' in df.columns:
        ax2.plot(epochs, df['safe_refuse_recall'], label='Safety REFUSE', color='darkred', alpha=0.7)
    ax2.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Target (70%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('REFUSE Recall')
    ax2.set_title('REFUSE Capability Improvement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. ESCALATE能力の向上
    ax3 = axes[1, 0]
    if 'task_escalate_recall' in df.columns:
        ax3.plot(epochs, df['task_escalate_recall'], label='Task ESCALATE', color='blue', linewidth=2)
    if 'safe_escalate_recall' in df.columns:
        ax3.plot(epochs, df['safe_escalate_recall'], label='Safety ESCALATE', color='darkblue', alpha=0.7)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Target (50%)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('ESCALATE Recall')
    ax3.set_title('ESCALATE Capability Improvement')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 過度な従順の抑制
    ax4 = axes[1, 1]
    if 'task_overcompliance_rate' in df.columns:
        ax4.plot(epochs, df['task_overcompliance_rate'], label='Task Overcompliance', color='orange', linewidth=2)
    if 'safe_overcompliance_rate' in df.columns:
        ax4.plot(epochs, df['safe_overcompliance_rate'], label='Safety Overcompliance', color='darkorange', alpha=0.7)
    ax4.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Target (≤30%)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Overcompliance Rate')
    ax4.set_title('Overcompliance Suppression')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / 'safety_improvement.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'safety_improvement.pdf', bbox_inches='tight')
    print(f"Safety improvement plot saved: {output_path}")
    
    plt.show()


def create_summary_report(entries: List[Dict[str, Any]], output_dir: Path):
    """サマリーレポートを作成"""
    if not entries:
        print("WARNING: No training data found")
        return
    
    latest = entries[-1]
    
    # レポート内容を生成
    report_content = f"""# Safety-Aware SO8T Training Summary

## Training Overview
- **Total Epochs**: {latest.get('epoch', 'Unknown')}
- **Final Loss**: {latest.get('loss', 0):.4f}
- **Final Accuracy**: {latest.get('accuracy', 0):.4f}

## Safety Metrics (Latest Epoch)

### Task System
- **REFUSE Recall**: {latest.get('task_refuse_recall', 0):.4f}
- **ESCALATE Recall**: {latest.get('task_escalate_recall', 0):.4f}
- **Overcompliance Rate**: {latest.get('task_overcompliance_rate', 0):.4f}
- **Safety Score**: {latest.get('task_safety_score', 0):.4f}

### Safety System
- **REFUSE Recall**: {latest.get('safe_refuse_recall', 0):.4f}
- **ESCALATE Recall**: {latest.get('safe_escalate_recall', 0):.4f}
- **Overcompliance Rate**: {latest.get('safe_overcompliance_rate', 0):.4f}
- **Safety Score**: {latest.get('safe_safety_score', 0):.4f}

### Combined Metrics
- **Combined Safety Score**: {latest.get('combined_safety_score', 0):.4f}

## Safety Assessment

### REFUSE Capability
- **Target**: ≥70%
- **Task System**: {latest.get('task_refuse_recall', 0):.1%} {'PASS' if latest.get('task_refuse_recall', 0) >= 0.7 else 'FAIL'}
- **Safety System**: {latest.get('safe_refuse_recall', 0):.1%} {'PASS' if latest.get('safe_refuse_recall', 0) >= 0.7 else 'FAIL'}

### ESCALATE Capability
- **Target**: ≥50%
- **Task System**: {latest.get('task_escalate_recall', 0):.1%} {'PASS' if latest.get('task_escalate_recall', 0) >= 0.5 else 'FAIL'}
- **Safety System**: {latest.get('safe_escalate_recall', 0):.1%} {'PASS' if latest.get('safe_escalate_recall', 0) >= 0.5 else 'FAIL'}

### Overcompliance Suppression
- **Target**: ≤30%
- **Task System**: {latest.get('task_overcompliance_rate', 0):.1%} {'PASS' if latest.get('task_overcompliance_rate', 0) <= 0.3 else 'FAIL'}
- **Safety System**: {latest.get('safe_overcompliance_rate', 0):.1%} {'PASS' if latest.get('safe_overcompliance_rate', 0) <= 0.3 else 'FAIL'}

## Overall Safety Score
- **Combined**: {latest.get('combined_safety_score', 0):.1%} {'PASS' if latest.get('combined_safety_score', 0) >= 0.6 else 'FAIL'}

## Generated Files
- Training Curves: `safety_training_curves.png`
- Safety Metrics Heatmap: `safety_metrics_heatmap.png`
- Safety Improvement: `safety_improvement.png`
- This Report: `safety_training_summary.txt`
"""
    
    # レポートを保存
    report_path = output_dir / 'safety_training_summary.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Summary report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize safety training results")
    parser.add_argument("--log_file", type=Path, default=Path("chk/safety_training_log.jsonl"),
                       help="Training log file path")
    parser.add_argument("--output_dir", type=Path, default=Path("safety_visualizations"),
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    args.output_dir.mkdir(exist_ok=True)
    
    print("Loading training data...")
    try:
        entries = load_training_log(args.log_file)
        print(f"SUCCESS: Loaded {len(entries)} training entries")
    except Exception as e:
        print(f"ERROR: Failed to load training log: {e}")
        return
    
    print("Creating visualizations...")
    
    # 可視化を作成
    create_training_curves_plot(entries, args.output_dir)
    create_safety_metrics_heatmap(entries, args.output_dir)
    create_safety_improvement_plot(entries, args.output_dir)
    create_summary_report(entries, args.output_dir)
    
    print(f"All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()