#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lm-eval-harness and ELYZA-100 Setup for A/B Testing

セットアップ内容：
1. lm-eval-harnessのインストールと検証
2. ELYZA-100データセットのダウンロード
3. 評価パイプラインの準備
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def install_lm_eval_harness():
    """lm-eval-harnessのインストール"""
    print("[SETUP] Installing lm-eval-harness...")

    try:
        # pipでインストール
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "lm-eval", "--upgrade"
        ])
        print("[OK] lm-eval-harness installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install lm-eval-harness: {e}")
        return False

def download_elyza_100():
    """ELYZA-100データセットのダウンロード"""
    print("[SETUP] Downloading ELYZA-100 dataset...")

    try:
        # ELYZA-100はHugging Faceからダウンロード
        from huggingface_hub import snapshot_download

        output_dir = Path("data/elyza100")
        output_dir.mkdir(parents=True, exist_ok=True)

        # ELYZA-100のダウンロード
        snapshot_download(
            repo_id="elyza/ELYZA-japanese-tasks",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False
        )

        print(f"[OK] ELYZA-100 downloaded to {output_dir}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to download ELYZA-100: {e}")
        return False

def verify_lm_eval_setup():
    """lm-evalのセットアップ検証"""
    print("[SETUP] Verifying lm-eval setup...")

    try:
        # lm-evalの基本機能をテスト
        result = subprocess.run([
            sys.executable, "-c",
            "import lm_eval; print('lm-eval version:', lm_eval.__version__)"
        ], capture_output=True, text=True, check=True)

        print(f"[OK] lm-eval verification passed: {result.stdout.strip()}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] lm-eval verification failed: {e}")
        return False

def create_evaluation_config():
    """評価設定ファイル作成"""
    print("[SETUP] Creating evaluation configuration...")

    config = {
        "model_args": {
            "model_name": "local_model",
            "model_path": "models/to/evaluate",
            "dtype": "float16"
        },
        "tasks": [
            {
                "task_name": "elyza_tasks",
                "task_type": "japanese_qa",
                "dataset_path": "data/elyza100"
            },
            {
                "task_name": "mmlu",
                "task_type": "multiple_choice",
                "categories": ["mathematics", "physics", "computer_science"]
            },
            {
                "task_name": "hellaswag",
                "task_type": "commonsense_reasoning"
            }
        ],
        "evaluation_settings": {
            "batch_size": 8,
            "max_length": 2048,
            "num_fewshot": 5,
            "seed": 42
        },
        "output_settings": {
            "results_dir": "results/ab_test_results",
            "log_interval": 10,
            "save_predictions": True
        }
    }

    config_path = Path("configs/evaluation_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[OK] Evaluation config created: {config_path}")
    return config_path

def setup_ab_test_directories():
    """A/Bテスト用ディレクトリ作成"""
    print("[SETUP] Setting up A/B test directories...")

    directories = [
        "results/ab_test_results",
        "results/ab_test_results/baseline",
        "results/ab_test_results/aegis",
        "results/ab_test_results/statistics",
        "results/ab_test_results/plots",
        "models/ab_test_models/baseline",
        "models/ab_test_models/aegis"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created directory: {dir_path}")

    return directories

def main():
    """メインセットアップ関数"""
    print("🚀 Setting up lm-eval-harness and ELYZA-100 for A/B testing...")
    print("=" * 60)

    setup_steps = [
        ("Install lm-eval-harness", install_lm_eval_harness),
        ("Download ELYZA-100", download_elyza_100),
        ("Verify lm-eval setup", verify_lm_eval_setup),
        ("Create evaluation config", create_evaluation_config),
        ("Setup A/B test directories", setup_ab_test_directories)
    ]

    results = {}
    for step_name, step_func in setup_steps:
        print(f"\n🔧 {step_name}...")
        try:
            result = step_func()
            results[step_name] = result
            if result:
                print(f"[PASS] {step_name}")
            else:
                print(f"[FAIL] {step_name}")
        except Exception as e:
            print(f"[ERROR] {step_name}: {e}")
            results[step_name] = False

    # 結果サマリー
    print("\n" + "=" * 60)
    print("📊 SETUP RESULTS SUMMARY")
    print("=" * 60)

    all_passed = True
    for step_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {step_name}")
        if not result:
            all_passed = False

    if all_passed:
        print("\n🎉 All setup steps completed successfully!")
        print("🚀 Ready for A/B testing with lm-eval-harness and ELYZA-100")

        # セットアップ完了ログ
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("lm_eval_setup_complete.log", 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] lm-eval-harness and ELYZA-100 setup completed successfully\n")

        return 0
    else:
        print("\n❌ Some setup steps failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())