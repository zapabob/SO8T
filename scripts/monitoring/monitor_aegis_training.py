#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS v2.1 トレーニング進捗モニタリングスクリプト
tqdmとloggingを使用したリアルタイム進捗表示
"""

import os
import sys
import time
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
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

class AEGISTrainingMonitor:
    """AEGISトレーニング進捗モニタリングクラス"""

    def __init__(self, study_name='aegis_v21_hyperparameter_optimization', storage_path='sqlite:///aegis_v21_optuna.db'):
        self.study_name = study_name
        self.storage_path = storage_path
        self.last_trial_count = 0
        self.last_completed_count = 0

    def load_study(self):
        """Optuna studyを読み込み"""
        try:
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage_path
            )
            return study
        except Exception as e:
            logger.error(f"Failed to load study: {e}")
            return None

    def get_training_stats(self, study):
        """トレーニング統計を取得"""
        if not study or not study.trials:
            return None

        total_trials = len(study.trials)
        running_trials = len([t for t in study.trials if t.state == 0])  # RUNNING
        completed_trials = len([t for t in study.trials if t.state == 1])  # COMPLETE
        failed_trials = len([t for t in study.trials if t.state == 2])  # FAIL

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

    def display_progress_bar(self, stats):
        """tqdm進捗バーを表示"""
        if not stats:
            return

        completed = stats['completed_trials']
        total = stats['total_trials']

        # メイン進捗バー
        with tqdm(total=total, desc="[AEGIS] Training Progress", unit="trial",
                  bar_format='{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            pbar.update(completed)
            pbar.refresh()

        # 詳細情報表示
        self.display_detailed_stats(stats)

    def display_detailed_stats(self, stats):
        """詳細統計を表示"""
        print("\n" + "="*80)
        print("🎯 AEGIS v2.1 トレーニング詳細状況")
        print("="*80)

        print(f"📊 総トライアル数: {stats['total_trials']}")
        print(f"✅ 完了済み: {stats['completed_trials']} ({stats['progress_percentage']:.1f}%)")
        print(f"🔄 実行中: {stats['running_trials']}")
        print(f"❌ 失敗: {stats['failed_trials']}")

        print(f"\n🎲 最新トライアル: #{stats['latest_trial_number']}")
        state_str = "実行中" if stats['latest_trial_state'] == 0 else "完了" if stats['latest_trial_state'] == 1 else "失敗"
        print(f"📈 状態: {state_str}")

        if stats['best_value'] is not None:
            print(f"🏆 ベストスコア: {stats['best_value']:.6f}")

        if stats['best_params']:
            print("\n🔧 最適パラメータ:")
            for key, value in stats['best_params'].items():
                print(f"   {key}: {value:.2e}")

        print("\n" + "="*80)

    def monitor_progress(self, interval=30):
        """進捗を定期的に監視"""
        logger.info("🎯 Starting AEGIS training progress monitoring...")
        logger.info(f"📊 Monitoring interval: {interval} seconds")

        try:
            while True:
                study = self.load_study()
                if study:
                    stats = self.get_training_stats(study)

                    if stats:
                        # 進捗変更を検知
                        if (stats['completed_trials'] != self.last_completed_count or
                            len(study.trials) != self.last_trial_count):

                            logger.info(f"📈 Progress update: {stats['completed_trials']}/{stats['total_trials']} trials completed")
                            self.display_progress_bar(stats)

                            self.last_completed_count = stats['completed_trials']
                            self.last_trial_count = len(study.trials)

                        # 完了チェック
                        if stats['running_trials'] == 0 and stats['total_trials'] > 0:
                            logger.info("🎉 All Optuna trials completed!")
                            self.display_final_results(study, stats)
                            break

                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")

    def display_final_results(self, study, stats):
        """最終結果を表示"""
        print("\n" + "🎉"*40)
        print("🏆 AEGIS v2.1 Optuna最適化完了！")
        print("🎉"*40)

        print("\n📊 最終統計:")
        print(f"   総トライアル数: {stats['total_trials']}")
        print(f"   成功完了: {stats['completed_trials']}")
        print(f"   失敗: {stats['failed_trials']}")
        print(".1f")
        if stats['best_value'] is not None:
            print("\n🏆 最適解:")
            print(".6f")
            if stats['best_params']:
                print("   SFT学習率: {stats['best_params']['sft_learning_rate']:.2e}")
                print("   PPO学習率: {stats['best_params']['ppo_learning_rate']:.2e}")
                print("   SO(8)アダプタ学習率: {stats['best_params']['adapter_learning_rate']:.2e}")

        print("\n🚀 次のステップ:")
        print("   1. PPOトレーニング開始 (Enhancedデータセット使用)")
        print("   2. Grokking現象観測")
        print("   3. 最終モデル保存")
        print("\n🎯 Grokking検知システム: 活性化済み")
        print("\n" + "🎉"*40)

        logger.info("🏆 Optuna optimization completed successfully")
        logger.info(f"Best value: {stats['best_value']}")
        logger.info(f"Best params: {stats['best_params']}")

    def get_system_status(self):
        """システム状態を取得"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_gb = memory.used / (1024**3)

            print("\n💻 システム状況:")
            print(".1f")
            print(".1f")
            print(".1f")
        except ImportError:
            print("
💻 システム状況: psutil not available"
def create_monitoring_dashboard():
    """モニタリングダッシュボード作成"""
    monitor = AEGISTrainingMonitor()

    print("🎯 AEGIS v2.1 トレーニング進捗モニタリング開始")
    print("=" * 60)
    print("📊 リアルタイムでトレーニング状況を表示します")
    print("🛑 停止するには Ctrl+C を押してください")
    print("=" * 60)

    # 初期状態表示
    study = monitor.load_study()
    if study:
        initial_stats = monitor.get_training_stats(study)
        if initial_stats:
            monitor.display_progress_bar(initial_stats)

    # システム状況表示
    monitor.get_system_status()

    print("
🔄 30秒ごとに進捗を更新します..."
    # 進捗監視開始
    monitor.monitor_progress(interval=30)

def main():
    """メイン関数"""
    try:
        create_monitoring_dashboard()
    except KeyboardInterrupt:
        print("\n\n🛑 モニタリングを停止しました")
        logger.info("Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        logger.error(f"Monitoring failed: {e}")
        raise

if __name__ == "__main__":
    main()
