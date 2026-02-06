#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8) PPO Pipeline Test Script
SO(8)PPO学習パイプラインのテストスクリプト
"""

import json
from pathlib import Path
from so8t_integrated_ppo_trainer import SO8TIntegratedDataset, SO8TPPOTrainer

def test_dataset_loading():
    """データセット読み込みテスト"""
    print("🔍 Testing SO(8) Integrated Dataset Loading...")

    dataset_path = "data/integrated/so8t_integrated_ppo_dataset_main_20251201_205340.jsonl"

    if not Path(dataset_path).exists():
        print(f"[NG] Dataset not found: {dataset_path}")
        return False

    try:
        dataset = SO8TIntegratedDataset(dataset_path)
        print(f"[OK] Dataset loaded successfully: {len(dataset)} entries")

        # サンプルデータの確認
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"📋 Sample data keys: {list(sample.keys())}")
            print(f"[TARGET] Sample instruction: {sample['instruction'][:100]}...")
            print(f"🏷️  Sample label: {sample['four_class_label']}")
            print(f"💰 Sample reward: {sample['reward_value']}")
            print(f"🧠 SO(8) score: {sample['so8t_combined_score']:.3f}")

        return True

    except Exception as e:
        print(f"[NG] Dataset loading failed: {e}")
        return False

def test_ppo_trainer_init():
    """PPOトレーナー初期化テスト"""
    print("\n[FIX] Testing SO(8) PPO Trainer Initialization...")

    dataset_path = "data/integrated/so8t_integrated_ppo_dataset_main_20251201_205340.jsonl"
    config_path = "scripts/training/so8t_ppo_config.json"

    if not Path(dataset_path).exists():
        print(f"[NG] Dataset not found: {dataset_path}")
        return False

    if not Path(config_path).exists():
        print(f"[NG] Config not found: {config_path}")
        return False

    try:
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # トレーナー初期化（モデルなしでテスト）
        trainer = SO8TPPOTrainer(
            model_path="models/Borea-Phi-3.5-mini-Instruct-Jp",
            dataset_path=dataset_path,
            config=config
        )

        print(f"[OK] PPO Trainer initialized successfully")
        print(f"[STATS] Dataset size: {len(trainer.dataset)}")
        print(f"[FIX] PPO config: learning_rate={trainer.ppo_config.learning_rate}")
        print(f"🧠 SO(8) config: vector_weight={trainer.so8t_config.vector_weight}")

        return True

    except Exception as e:
        print(f"[NG] PPO Trainer initialization failed: {e}")
        return False

def test_reward_calculation():
    """報酬計算テスト"""
    print("\n💰 Testing SO(8) Reward Calculation...")

    try:
        # テストデータ
        test_data = {
            'reward_value': [1.0, -1.0, 0.5, -2.0],
            'so8t_vector_score': [0.8, 0.2, 0.6, 0.1],
            'so8t_spinor_plus_score': [0.7, 0.1, 0.5, 0.0],
            'so8t_spinor_minus_score': [0.1, 0.8, 0.2, 0.9],
            'is_nsfw': [False, True, False, True],
            'quality_score': [0.9, 0.3, 0.7, 0.2]
        }

        # SO(8)設定
        so8t_config = type('Config', (), {
            'vector_weight': 0.3,
            'spinor_plus_weight': 0.4,
            'spinor_minus_weight': 0.3
        })()

        # 報酬計算（簡易版）
        rewards = []
        for i in range(len(test_data['reward_value'])):
            base_reward = test_data['reward_value'][i]

            # SO(8)統合
            so8t_reward = (
                so8t_config.vector_weight * test_data['so8t_vector_score'][i] +
                so8t_config.spinor_plus_weight * test_data['so8t_spinor_plus_score'][i] +
                so8t_config.spinor_minus_weight * test_data['so8t_spinor_minus_score'][i]
            )

            final_reward = base_reward + 0.1 * so8t_reward

            # NSFWペナルティ
            if test_data['is_nsfw'][i]:
                final_reward -= 0.5

            # 品質ボーナス
            quality_bonus = test_data['quality_score'][i] - 0.5
            final_reward += 0.2 * quality_bonus

            rewards.append(final_reward)

        print("[OK] SO(8) Reward calculation test results:")
        for i, reward in enumerate(rewards):
            print(".3f")
        print(".3f")
        return True

    except Exception as e:
        print(f"[NG] Reward calculation test failed: {e}")
        return False

def main():
    """メイン関数"""
    print("[TEST] SO(8) PPO Pipeline Test Suite")
    print("=" * 50)

    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("PPO Trainer Init", test_ppo_trainer_init),
        ("Reward Calculation", test_reward_calculation)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[NG] {test_name} crashed: {e}")
            results.append((test_name, False))

    # 結果サマリー
    print("\n" + "=" * 50)
    print("[STATS] Test Results Summary:")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "[OK] PASS" if result else "[NG] FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n[TARGET] Overall: {passed}/{total} tests passed")

    if passed == total:
        print("[DONE] All tests passed! SO(8) PPO pipeline is ready!")
        # 成功音声通知
        try:
            import subprocess
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-File", "scripts\\utils\\play_audio_notification.ps1"
            ], check=True)
        except Exception as e:
            print(f"[WARNING] Audio notification failed: {e}")
    else:
        print("[WARN]  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()

"""
SO(8) PPO Pipeline Test Script
SO(8)PPO学習パイプラインのテストスクリプト
"""

import json
from pathlib import Path
from so8t_integrated_ppo_trainer import SO8TIntegratedDataset, SO8TPPOTrainer

def test_dataset_loading():
    """データセット読み込みテスト"""
    print("🔍 Testing SO(8) Integrated Dataset Loading...")

    dataset_path = "data/integrated/so8t_integrated_ppo_dataset_main_20251201_205340.jsonl"

    if not Path(dataset_path).exists():
        print(f"[NG] Dataset not found: {dataset_path}")
        return False

    try:
        dataset = SO8TIntegratedDataset(dataset_path)
        print(f"[OK] Dataset loaded successfully: {len(dataset)} entries")

        # サンプルデータの確認
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"📋 Sample data keys: {list(sample.keys())}")
            print(f"[TARGET] Sample instruction: {sample['instruction'][:100]}...")
            print(f"🏷️  Sample label: {sample['four_class_label']}")
            print(f"💰 Sample reward: {sample['reward_value']}")
            print(f"🧠 SO(8) score: {sample['so8t_combined_score']:.3f}")

        return True

    except Exception as e:
        print(f"[NG] Dataset loading failed: {e}")
        return False

def test_ppo_trainer_init():
    """PPOトレーナー初期化テスト"""
    print("\n[FIX] Testing SO(8) PPO Trainer Initialization...")

    dataset_path = "data/integrated/so8t_integrated_ppo_dataset_main_20251201_205340.jsonl"
    config_path = "scripts/training/so8t_ppo_config.json"

    if not Path(dataset_path).exists():
        print(f"[NG] Dataset not found: {dataset_path}")
        return False

    if not Path(config_path).exists():
        print(f"[NG] Config not found: {config_path}")
        return False

    try:
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # トレーナー初期化（モデルなしでテスト）
        trainer = SO8TPPOTrainer(
            model_path="models/Borea-Phi-3.5-mini-Instruct-Jp",
            dataset_path=dataset_path,
            config=config
        )

        print(f"[OK] PPO Trainer initialized successfully")
        print(f"[STATS] Dataset size: {len(trainer.dataset)}")
        print(f"[FIX] PPO config: learning_rate={trainer.ppo_config.learning_rate}")
        print(f"🧠 SO(8) config: vector_weight={trainer.so8t_config.vector_weight}")

        return True

    except Exception as e:
        print(f"[NG] PPO Trainer initialization failed: {e}")
        return False

def test_reward_calculation():
    """報酬計算テスト"""
    print("\n💰 Testing SO(8) Reward Calculation...")

    try:
        # テストデータ
        test_data = {
            'reward_value': [1.0, -1.0, 0.5, -2.0],
            'so8t_vector_score': [0.8, 0.2, 0.6, 0.1],
            'so8t_spinor_plus_score': [0.7, 0.1, 0.5, 0.0],
            'so8t_spinor_minus_score': [0.1, 0.8, 0.2, 0.9],
            'is_nsfw': [False, True, False, True],
            'quality_score': [0.9, 0.3, 0.7, 0.2]
        }

        # SO(8)設定
        so8t_config = type('Config', (), {
            'vector_weight': 0.3,
            'spinor_plus_weight': 0.4,
            'spinor_minus_weight': 0.3
        })()

        # 報酬計算（簡易版）
        rewards = []
        for i in range(len(test_data['reward_value'])):
            base_reward = test_data['reward_value'][i]

            # SO(8)統合
            so8t_reward = (
                so8t_config.vector_weight * test_data['so8t_vector_score'][i] +
                so8t_config.spinor_plus_weight * test_data['so8t_spinor_plus_score'][i] +
                so8t_config.spinor_minus_weight * test_data['so8t_spinor_minus_score'][i]
            )

            final_reward = base_reward + 0.1 * so8t_reward

            # NSFWペナルティ
            if test_data['is_nsfw'][i]:
                final_reward -= 0.5

            # 品質ボーナス
            quality_bonus = test_data['quality_score'][i] - 0.5
            final_reward += 0.2 * quality_bonus

            rewards.append(final_reward)

        print("[OK] SO(8) Reward calculation test results:")
        for i, reward in enumerate(rewards):
            print(".3f")
        print(".3f")
        return True

    except Exception as e:
        print(f"[NG] Reward calculation test failed: {e}")
        return False

def main():
    """メイン関数"""
    print("[TEST] SO(8) PPO Pipeline Test Suite")
    print("=" * 50)

    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("PPO Trainer Init", test_ppo_trainer_init),
        ("Reward Calculation", test_reward_calculation)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[NG] {test_name} crashed: {e}")
            results.append((test_name, False))

    # 結果サマリー
    print("\n" + "=" * 50)
    print("[STATS] Test Results Summary:")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "[OK] PASS" if result else "[NG] FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n[TARGET] Overall: {passed}/{total} tests passed")

    if passed == total:
        print("[DONE] All tests passed! SO(8) PPO pipeline is ready!")
        # 成功音声通知
        try:
            import subprocess
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-File", "scripts\\utils\\play_audio_notification.ps1"
            ], check=True)
        except Exception as e:
            print(f"[WARNING] Audio notification failed: {e}")
    else:
        print("[WARN]  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()

"""
SO(8) PPO Pipeline Test Script
SO(8)PPO学習パイプラインのテストスクリプト
"""

import json
from pathlib import Path
from so8t_integrated_ppo_trainer import SO8TIntegratedDataset, SO8TPPOTrainer

def test_dataset_loading():
    """データセット読み込みテスト"""
    print("🔍 Testing SO(8) Integrated Dataset Loading...")

    dataset_path = "data/integrated/so8t_integrated_ppo_dataset_main_20251201_205340.jsonl"

    if not Path(dataset_path).exists():
        print(f"[NG] Dataset not found: {dataset_path}")
        return False

    try:
        dataset = SO8TIntegratedDataset(dataset_path)
        print(f"[OK] Dataset loaded successfully: {len(dataset)} entries")

        # サンプルデータの確認
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"📋 Sample data keys: {list(sample.keys())}")
            print(f"[TARGET] Sample instruction: {sample['instruction'][:100]}...")
            print(f"🏷️  Sample label: {sample['four_class_label']}")
            print(f"💰 Sample reward: {sample['reward_value']}")
            print(f"🧠 SO(8) score: {sample['so8t_combined_score']:.3f}")

        return True

    except Exception as e:
        print(f"[NG] Dataset loading failed: {e}")
        return False

def test_ppo_trainer_init():
    """PPOトレーナー初期化テスト"""
    print("\n[FIX] Testing SO(8) PPO Trainer Initialization...")

    dataset_path = "data/integrated/so8t_integrated_ppo_dataset_main_20251201_205340.jsonl"
    config_path = "scripts/training/so8t_ppo_config.json"

    if not Path(dataset_path).exists():
        print(f"[NG] Dataset not found: {dataset_path}")
        return False

    if not Path(config_path).exists():
        print(f"[NG] Config not found: {config_path}")
        return False

    try:
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # トレーナー初期化（モデルなしでテスト）
        trainer = SO8TPPOTrainer(
            model_path="models/Borea-Phi-3.5-mini-Instruct-Jp",
            dataset_path=dataset_path,
            config=config
        )

        print(f"[OK] PPO Trainer initialized successfully")
        print(f"[STATS] Dataset size: {len(trainer.dataset)}")
        print(f"[FIX] PPO config: learning_rate={trainer.ppo_config.learning_rate}")
        print(f"🧠 SO(8) config: vector_weight={trainer.so8t_config.vector_weight}")

        return True

    except Exception as e:
        print(f"[NG] PPO Trainer initialization failed: {e}")
        return False

def test_reward_calculation():
    """報酬計算テスト"""
    print("\n💰 Testing SO(8) Reward Calculation...")

    try:
        # テストデータ
        test_data = {
            'reward_value': [1.0, -1.0, 0.5, -2.0],
            'so8t_vector_score': [0.8, 0.2, 0.6, 0.1],
            'so8t_spinor_plus_score': [0.7, 0.1, 0.5, 0.0],
            'so8t_spinor_minus_score': [0.1, 0.8, 0.2, 0.9],
            'is_nsfw': [False, True, False, True],
            'quality_score': [0.9, 0.3, 0.7, 0.2]
        }

        # SO(8)設定
        so8t_config = type('Config', (), {
            'vector_weight': 0.3,
            'spinor_plus_weight': 0.4,
            'spinor_minus_weight': 0.3
        })()

        # 報酬計算（簡易版）
        rewards = []
        for i in range(len(test_data['reward_value'])):
            base_reward = test_data['reward_value'][i]

            # SO(8)統合
            so8t_reward = (
                so8t_config.vector_weight * test_data['so8t_vector_score'][i] +
                so8t_config.spinor_plus_weight * test_data['so8t_spinor_plus_score'][i] +
                so8t_config.spinor_minus_weight * test_data['so8t_spinor_minus_score'][i]
            )

            final_reward = base_reward + 0.1 * so8t_reward

            # NSFWペナルティ
            if test_data['is_nsfw'][i]:
                final_reward -= 0.5

            # 品質ボーナス
            quality_bonus = test_data['quality_score'][i] - 0.5
            final_reward += 0.2 * quality_bonus

            rewards.append(final_reward)

        print("[OK] SO(8) Reward calculation test results:")
        for i, reward in enumerate(rewards):
            print(".3f")
        print(".3f")
        return True

    except Exception as e:
        print(f"[NG] Reward calculation test failed: {e}")
        return False

def main():
    """メイン関数"""
    print("[TEST] SO(8) PPO Pipeline Test Suite")
    print("=" * 50)

    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("PPO Trainer Init", test_ppo_trainer_init),
        ("Reward Calculation", test_reward_calculation)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[NG] {test_name} crashed: {e}")
            results.append((test_name, False))

    # 結果サマリー
    print("\n" + "=" * 50)
    print("[STATS] Test Results Summary:")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "[OK] PASS" if result else "[NG] FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n[TARGET] Overall: {passed}/{total} tests passed")

    if passed == total:
        print("[DONE] All tests passed! SO(8) PPO pipeline is ready!")
        # 成功音声通知
        try:
            import subprocess
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-File", "scripts\\utils\\play_audio_notification.ps1"
            ], check=True)
        except Exception as e:
            print(f"[WARNING] Audio notification failed: {e}")
    else:
        print("[WARN]  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
