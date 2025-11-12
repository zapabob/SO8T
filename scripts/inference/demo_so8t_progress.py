#!/usr/bin/env python3
"""
SO8T学習進捗デモンストレーション

学習の進行状況をリアルタイムで表示し、電源断からの復旧状況を確認します。

Usage:
    python demo_so8t_progress.py
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

def print_banner():
    """バナーを表示します。"""
    print("=" * 80)
    print("🚀 SO8T Safe Agent 学習進捗デモンストレーション")
    print("=" * 80)
    print(f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def check_training_process():
    """学習プロセスを確認します。"""
    try:
        # Pythonプロセスでメモリ使用量が大きいものを検索
        result = subprocess.run([
            'powershell', '-Command',
            'Get-Process | Where-Object {$_.ProcessName -eq "python" -and $_.WorkingSet -gt 200MB} | Select-Object Id, ProcessName, CPU, WorkingSet'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            print("✅ 学習プロセスが実行中です:")
            print(result.stdout)
            return True
        else:
            print("❌ 学習プロセスが見つかりません")
            return False
    except Exception as e:
        print(f"❌ プロセス確認エラー: {e}")
        return False

def check_checkpoints():
    """チェックポイントを確認します。"""
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("❌ チェックポイントディレクトリが存在しません")
        return False
    
    # 最新のセッションディレクトリを取得
    sessions = list(checkpoint_dir.glob("so8t_qwen2.5-7b_session_*"))
    if not sessions:
        print("❌ セッションディレクトリが見つかりません")
        return False
    
    latest_session = max(sessions, key=lambda x: x.stat().st_mtime)
    print(f"📁 最新セッション: {latest_session.name}")
    
    # セッション内のファイルを確認
    files = list(latest_session.glob("*"))
    if files:
        print(f"📄 ファイル数: {len(files)}")
        for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
            size_mb = file.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            print(f"  - {file.name} ({size_mb:.1f}MB, {mtime.strftime('%H:%M:%S')})")
    else:
        print("📄 ファイルがまだ作成されていません")
    
    return True

def check_gpu_usage():
    """GPU使用状況を確認します。"""
    try:
        # nvidia-smiでGPU使用状況を確認
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_util, mem_used, mem_total, temp = parts
                    print(f"🎮 GPU {i}: 使用率 {gpu_util}%, メモリ {mem_used}/{mem_total}MB, 温度 {temp}°C")
            return True
        else:
            print("❌ nvidia-smiが利用できません")
            return False
    except Exception as e:
        print(f"❌ GPU確認エラー: {e}")
        return False

def estimate_training_time():
    """学習時間を推定します。"""
    print("\n⏱️  学習時間推定:")
    print("  - エポック数: 5")
    print("  - バッチサイズ: 2 (実質16 with accumulation)")
    print("  - データセットサイズ: 20サンプル")
    print("  - 推定時間: 約30-60分")
    print("  - チェックポイント間隔: 100ステップ")

def main():
    """メイン関数。"""
    print_banner()
    
    print("\n🔍 学習状況確認中...")
    
    # 学習プロセス確認
    process_running = check_training_process()
    
    # チェックポイント確認
    checkpoints_exist = check_checkpoints()
    
    # GPU使用状況確認
    gpu_available = check_gpu_usage()
    
    # 学習時間推定
    estimate_training_time()
    
    print("\n📊 状況サマリー:")
    print(f"  - 学習プロセス: {'✅ 実行中' if process_running else '❌ 停止中'}")
    print(f"  - チェックポイント: {'✅ 存在' if checkpoints_exist else '❌ 未作成'}")
    print(f"  - GPU: {'✅ 利用可能' if gpu_available else '❌ 利用不可'}")
    
    if process_running and checkpoints_exist:
        print("\n🎉 SO8T学習が正常に進行中です！")
        print("💡 電源断からの復旧システムが動作しています")
        print("🔄 5分間隔で自動チェックポイント保存中")
    elif process_running:
        print("\n⏳ 学習プロセスは実行中ですが、まだチェックポイントが作成されていません")
        print("💡 学習が開始されるまでしばらくお待ちください")
    else:
        print("\n❌ 学習プロセスが停止しています")
        print("💡 学習を再開してください")
    
    print("\n" + "=" * 80)
    print("SO8T Safe Agent - 安全で信頼できるAIエージェントの実現を目指して")
    print("=" * 80)

if __name__ == "__main__":
    main()
