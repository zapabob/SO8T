#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lm-evaluation-harness 利用可能タスク確認スクリプト
"""

import subprocess
import sys
from pathlib import Path

def check_available_tasks():
    """利用可能なタスクを確認"""
    lm_eval_path = Path('./lm-evaluation-harness')

    if not lm_eval_path.exists():
        print("[NG] lm-evaluation-harness ディレクトリが見つかりません")
        return False

    print("🔍 lm-evaluation-harness 利用可能タスクを確認中...")
    print("=" * 60)

    try:
        # タスクリスト取得
        cmd = [sys.executable, '-m', 'lm_eval', '--tasks', 'list']
        result = subprocess.run(
            cmd,
            cwd=str(lm_eval_path),
            capture_output=True,
            text=True,
            check=True
        )

        print("[OK] 利用可能タスク一覧:")
        print(result.stdout)

        # hellaswag と mmlu が利用可能か確認
        key_tasks = ['hellaswag', 'mmlu']
        available_tasks = result.stdout.lower()

        print("\n[TARGET] SO(8)T推奨タスク確認:")
        for task in key_tasks:
            if task.lower() in available_tasks:
                print(f"[OK] {task}: 利用可能")
            else:
                print(f"[NG] {task}: 利用不可")

        return True

    except subprocess.CalledProcessError as e:
        print(f"[NG] タスク確認失敗: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def check_model_loading():
    """モデル読み込みテスト"""
    lm_eval_path = Path('./lm-evaluation-harness')

    print("\n[FIX] HFモデル読み込みテスト...")
    print("=" * 60)

    try:
        # 簡単なHFモデルテスト
        cmd = [
            sys.executable, '-m', 'lm_eval',
            '--model', 'hf',
            '--model_args', 'pretrained=microsoft/phi-2,dtype=float16',
            '--tasks', 'hellaswag',
            '--limit', '1',  # 1サンプルだけ
            '--device', 'cpu'  # CPUでテスト
        ]

        result = subprocess.run(
            cmd,
            cwd=str(lm_eval_path),
            capture_output=True,
            text=True,
            timeout=60  # 60秒タイムアウト
        )

        if result.returncode == 0:
            print("[OK] HFモデル読み込み: 成功")
        else:
            print(f"[WARN] HFモデル読み込み: 失敗 (exit code: {result.returncode})")
            print(f"stderr: {result.stderr[-500:]}")  # 末尾500文字

    except subprocess.TimeoutExpired:
        print("[WARN] HFモデル読み込み: タイムアウト")
    except Exception as e:
        print(f"[WARN] HFモデル読み込み: エラー - {e}")

def main():
    """メイン関数"""
    print("[START] lm-evaluation-harness 環境チェック")
    print("=" * 60)

    # タスク確認
    tasks_ok = check_available_tasks()

    # モデル読み込みテスト
    check_model_loading()

    print("\n" + "=" * 60)
    if tasks_ok:
        print("[OK] lm-evaluation-harness 環境チェック完了")
        print("\n📋 SO(8)Tベンチマーク実行例:")
        print("python so8t_lm_eval_benchmark.py --ab-compare")
        print("python so8t_lm_eval_benchmark.py --model-a-only")
        print("python so8t_lm_eval_benchmark.py --model-b-only")
    else:
        print("[NG] lm-evaluation-harness 環境チェック失敗")

    # 音声通知
    try:
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", "scripts\\utils\\play_audio_notification.ps1"
        ], check=True)
    except Exception as e:
        print(f"[WARNING] 音声通知失敗: {e}")

if __name__ == "__main__":
    main()

