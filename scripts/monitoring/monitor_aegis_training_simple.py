#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS v2.1 トレーニング進捗モニタリングスクリプト (シンプル版)
tqdmとloggingを使用したリアルタイム進捗表示
"""

import os
import sys
import time
import logging
from tqdm import tqdm
import optuna

# Windows cp932エンコーディング対策
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aegis_training_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_training_stats():
    """トレーニング統計を取得"""
    try:
        study = optuna.load_study(
            study_name='aegis_v21_hyperparameter_optimization',
            storage='sqlite:///aegis_v21_optuna.db'
        )

        total_trials = len(study.trials)
        running_trials = len([t for t in study.trials if t.state == 0])
        completed_trials = len([t for t in study.trials if t.state == 1])
        failed_trials = len([t for t in study.trials if t.state == 2])

        latest_trial = max(study.trials, key=lambda t: t.number)

        stats = {
            'total_trials': total_trials,
            'running_trials': running_trials,
            'completed_trials': completed_trials,
            'failed_trials': failed_trials,
            'progress_percentage': (completed_trials / total_trials) * 100 if total_trials > 0 else 0,
            'latest_trial_number': latest_trial.number,
            'latest_trial_state': latest_trial.state,
            'best_value': study.best_value if hasattr(study, 'best_value') else None,
            'best_params': study.best_params if hasattr(study, 'best_params') else None
        }

        return stats

    except Exception as e:
        logger.error(f"Failed to get training stats: {e}")
        return None

def display_progress_bar(stats):
    """tqdm進捗バーを表示"""
    if not stats:
        return

    completed = stats['completed_trials']
    total = stats['total_trials']

    # メイン進捗バー
    desc = f"[AEGIS] Training Progress ({stats['progress_percentage']:.1f}%)"
    with tqdm(total=total, desc=desc, unit="trial",
              bar_format='{desc}: |{bar}| {n_fmt}/{total_fmt} [{elapsed}]') as pbar:
        pbar.update(completed)

def display_detailed_stats(stats):
    """詳細統計を表示"""
    print("\n" + "="*80)
    print("[INFO] AEGIS v2.1 Training Status")
    print("="*80)

    print(f"[INFO] Total Trials: {stats['total_trials']}")
    print(f"[SUCCESS] Completed: {stats['completed_trials']} ({stats['progress_percentage']:.1f}%)")
    print(f"[RUNNING] Running: {stats['running_trials']}")
    print(f"[ERROR] Failed: {stats['failed_trials']}")

    print(f"\n[INFO] Latest Trial: #{stats['latest_trial_number']}")
    state_str = "Running" if stats['latest_trial_state'] == 0 else "Completed" if stats['latest_trial_state'] == 1 else "Failed"
    print(f"[INFO] Status: {state_str}")

    if stats['best_value'] is not None:
        print(f"[BEST] Best Score: {stats['best_value']:.6f}")

    if stats['best_params']:
        print("\n[PARAMS] Best Parameters:")
        for key, value in stats['best_params'].items():
            print(f"   {key}: {value:.2e}")

    print("="*80)

def monitor_progress():
    """進捗を定期的に監視"""
    logger.info("[START] Starting AEGIS training progress monitoring...")

    last_completed = 0
    last_total = 0

    try:
        while True:
            stats = get_training_stats()

            if stats:
                # 進捗変更を検知
                if (stats['completed_trials'] != last_completed or
                    stats['total_trials'] != last_total):

                    logger.info(f"[UPDATE] Progress: {stats['completed_trials']}/{stats['total_trials']} trials completed")
                    display_progress_bar(stats)
                    display_detailed_stats(stats)

                    last_completed = stats['completed_trials']
                    last_total = stats['total_trials']

                # 完了チェック
                if stats['running_trials'] == 0 and stats['total_trials'] > 0:
                    logger.info("[COMPLETE] All Optuna trials completed!")
                    display_final_results(stats)
                    break

            time.sleep(30)  # 30秒ごとに更新

    except KeyboardInterrupt:
        print("\n\n[STOP] Monitoring stopped by user")
        logger.info("Monitoring stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Error occurred: {e}")
        logger.error(f"Monitoring failed: {e}")

def display_final_results(stats):
    """最終結果を表示"""
    print("\n" + "[SUCCESS]"*40)
    print("[WIN] AEGIS v2.1 Optuna Optimization Complete!")
    print("[SUCCESS]"*40)

    print("\n[STATS] Final Statistics:")
    print(f"   Total Trials: {stats['total_trials']}")
    print(f"   Successful: {stats['completed_trials']}")
    print(f"   Failed: {stats['failed_trials']}")
    print(".1f")

    if stats['best_value'] is not None:
        print("\n[BEST] Optimal Solution:")
        print(".6f")
        if stats['best_params']:
            print(f"   SFT Learning Rate: {stats['best_params']['sft_learning_rate']:.2e}")
            print(f"   PPO Learning Rate: {stats['best_params']['ppo_learning_rate']:.2e}")
            print(f"   SO(8) Adapter Learning Rate: {stats['best_params']['adapter_learning_rate']:.2e}")

    print("\n[NEXT] Next Steps:")
    print("   1. PPO Training Start (Enhanced Dataset)")
    print("   2. Grokking Phenomenon Observation")
    print("   3. Final Model Save")

    print("\n[SYSTEM] Grokking Detection System: Active")
    print("[SUCCESS]"*40)

def main():
    """メイン関数"""
    print("[START] AEGIS v2.1 Training Progress Monitoring")
    print("=" * 60)
    print("[INFO] Real-time training status display")
    print("[STOP] Press Ctrl+C to stop")
    print("=" * 60)

    # 初期状態表示
    initial_stats = get_training_stats()
    if initial_stats:
        display_progress_bar(initial_stats)
        display_detailed_stats(initial_stats)

    print("\n[UPDATE] Updating progress every 30 seconds...")
    logger.info("Starting progress monitoring with 30-second intervals")

    # 進捗監視開始
    monitor_progress()

if __name__ == "__main__":
    main()
