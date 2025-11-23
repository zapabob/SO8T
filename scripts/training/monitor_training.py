#!/usr/bin/env python3
"""
SO8T学習監視スクリプト

学習の進行状況をリアルタイムで監視し、電源断からの復旧状況を表示します。

Usage:
    python monitor_training.py
    python monitor_training.py --checkpoint_dir checkpoints/so8t_qwen2.5-7b_session_20251027_201432
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import pandas as pd

def load_training_history(checkpoint_dir: Path) -> List[Dict[str, Any]]:
    """学習履歴を読み込みます。"""
    history = []
    
    # JSONファイルから履歴を読み込み
    json_files = list(checkpoint_dir.glob("*.json"))
    json_files.sort(key=lambda x: x.stat().st_mtime)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'training_history' in data:
                    history.extend(data['training_history'])
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return history

def plot_training_progress(history: List[Dict[str, Any]], output_dir: Path):
    """学習進捗をプロットします。"""
    if not history:
        print("No training history found")
        return
    
    # データフレームに変換
    df = pd.DataFrame(history)
    
    # プロット作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SO8T Training Progress', fontsize=16)
    
    # 損失の推移
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 安全スコアの推移
    axes[0, 1].plot(df['epoch'], df['safety_score'], label='Safety Score', marker='o', color='red')
    axes[0, 1].set_title('Safety Score Progress')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Safety Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 決定分布（最新エポック）
    if 'val_metrics' in df.columns:
        latest_metrics = df['val_metrics'].iloc[-1]
        if isinstance(latest_metrics, dict):
            decisions = list(latest_metrics.keys())
            counts = list(latest_metrics.values())
            axes[1, 0].bar(decisions, counts, color=['green', 'red', 'orange'])
            axes[1, 0].set_title('Latest Decision Distribution')
            axes[1, 0].set_xlabel('Decision Type')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].grid(True)
    
    # 学習時間の推移
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
        time_diffs = timestamps.diff().dt.total_seconds() / 60  # 分単位
        axes[1, 1].plot(df['epoch'], time_diffs, marker='o', color='purple')
        axes[1, 1].set_title('Epoch Duration')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Duration (minutes)')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # 保存
    plot_path = output_dir / "training_progress.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training progress plot saved: {plot_path}")
    
    # 表示
    plt.show()

def print_training_status(checkpoint_dir: Path):
    """学習状況を表示します。"""
    print(f"\n{'='*60}")
    print(f"SO8T Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # チェックポイント一覧
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"\nCheckpoint Directory: {checkpoint_dir}")
    print(f"Total Checkpoints: {len(checkpoints)}")
    
    if checkpoints:
        print(f"\nLatest Checkpoints:")
        for i, checkpoint in enumerate(checkpoints[:5]):  # 最新5個
            stat = checkpoint.stat()
            size_mb = stat.st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"  {i+1}. {checkpoint.name} ({size_mb:.1f}MB, {mtime.strftime('%H:%M:%S')})")
    
    # 学習履歴
    history = load_training_history(checkpoint_dir)
    
    if history:
        print(f"\nTraining History:")
        print(f"Total Epochs: {len(history)}")
        
        if history:
            latest = history[-1]
            print(f"Latest Epoch: {latest.get('epoch', 'N/A')}")
            print(f"Train Loss: {latest.get('train_loss', 0):.4f}")
            print(f"Val Loss: {latest.get('val_loss', 0):.4f}")
            print(f"Safety Score: {latest.get('safety_score', 0):.4f}")
            
            if 'val_metrics' in latest:
                print(f"Val Metrics: {latest['val_metrics']}")
    else:
        print("\nNo training history found")
    
    # 復旧状況
    json_files = list(checkpoint_dir.glob("*.json"))
    if json_files:
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
        try:
            with open(latest_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"\nRecovery Status:")
                print(f"Session ID: {data.get('session_id', 'N/A')}")
                print(f"Last Update: {data.get('timestamp', 'N/A')}")
                print(f"Checkpoint Type: {data.get('checkpoint_type', 'N/A')}")
                print(f"Is Emergency: {data.get('is_emergency', False)}")
        except Exception as e:
            print(f"Error reading latest checkpoint: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SO8T Training Monitor")
    parser.add_argument("--checkpoint_dir", type=str, 
                       default="checkpoints/so8t_qwen2.5-7b_session_20251027_201432",
                       help="Path to checkpoint directory")
    parser.add_argument("--plot", action="store_true", 
                       help="Generate training progress plot")
    parser.add_argument("--watch", action="store_true", 
                       help="Watch mode (continuous monitoring)")
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    if args.watch:
        print("Starting watch mode (press Ctrl+C to stop)...")
        try:
            while True:
                print_training_status(checkpoint_dir)
                time.sleep(30)  # 30秒間隔で更新
        except KeyboardInterrupt:
            print("\nWatch mode stopped")
    else:
        print_training_status(checkpoint_dir)
        
        if args.plot:
            plot_training_progress(load_training_history(checkpoint_dir), checkpoint_dir)

if __name__ == "__main__":
    main()
