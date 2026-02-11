#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8)T Automated Pipeline
完全自動化されたSO(8)T学習・評価・アップロードパイプライン
"""

import os
import time
import json
import subprocess
import threading
import schedule
import atexit
import signal
from pathlib import Path
from datetime import datetime, timedelta
import logging
import psutil
import winreg
from typing import Dict, List, Any, Optional

# 自作モジュール
from so8t_sft_training_pipeline import SO8TSFTTrainer, create_sft_config
from so8t_ppo_training_pipeline import SO8TPPOTrainer, create_ppo_config
from so8t_gguf_conversion_pipeline import SO8TGGUFConverter, create_gguf_config
from so8t_benchmark_pipeline import SO8TBenchmarkRunner, EnhancedStatisticalAnalyzer, EnhancedHFPreparator, create_benchmark_config
from data_cleansing import AdvancedDataCleanser, create_cleansing_config

logger = logging.getLogger(__name__)

class SO8TAutomatedPipeline:
    """SO(8)T自動化パイプライン"""

    def __init__(self):
        self.checkpoint_dir = Path('./checkpoints/auto_pipeline')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.checkpoint_dir / 'session_state.json'

        self.rolling_checkpoints = []
        self.max_checkpoints = 5
        self.checkpoint_interval = 180  # 3分

        self.is_running = False
        self.current_stage = "idle"
        self.session_id = self.generate_session_id()
        self.power_interrupt_detected = False

        # 電源監視強化
        self.power_monitor = EnhancedPowerStateMonitor()

        # 自動起動設定
        self.setup_autostart()

        # セッション復旧チェック
        self.check_session_recovery()

        logger.info(f"SO(8)T Automated Pipeline initialized - Session: {self.session_id}")

        # 電源断シグナルハンドラ設定
        self.setup_signal_handlers()

        # プログラム終了時のクリーンアップ
        atexit.register(self.emergency_save)

    def generate_session_id(self) -> str:
        """セッションID生成"""
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    def setup_signal_handlers(self):
        """電源断シグナルハンドラ設定"""
        def signal_handler(signum, frame):
            logger.warning(f"Signal {signum} received - emergency save triggered")
            self.power_interrupt_detected = True
            self.emergency_save()
            # クリーンに終了
            os._exit(1)

        # Windows対応シグナル
        try:
            signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)  # 終了要求
            signal.signal(signal.SIGBREAK, signal_handler)  # Ctrl+Break (Windows)
        except (OSError, ValueError) as e:
            logger.warning(f"Signal handler setup failed: {e}")

    def check_session_recovery(self):
        """セッション復旧チェック"""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)

                last_stage = session_data.get('current_stage', 'idle')
                last_checkpoint = session_data.get('last_checkpoint')

                if last_stage != 'idle' and last_checkpoint:
                    logger.info(f"Previous session found - Stage: {last_stage}")
                    logger.info(f"Last checkpoint: {last_checkpoint}")

                    # ユーザーに復旧確認（自動モード時は自動復旧）
                    if self.should_auto_recover():
                        logger.info("Auto-recovery enabled - resuming from checkpoint")
                        self.recover_from_checkpoint(last_checkpoint)
                    else:
                        logger.info("Manual recovery required - use --recover flag")

            except Exception as e:
                logger.error(f"Session recovery check failed: {e}")

    def should_auto_recover(self) -> bool:
        """自動復旧判定"""
        return os.getenv('SO8T_AUTO_RECOVER', 'true').lower() == 'true'

    def recover_from_checkpoint(self, checkpoint_path: str):
        """チェックポイントから復旧"""
        try:
            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return

            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            self.current_stage = checkpoint_data.get('stage', 'idle')
            logger.info(f"Recovered to stage: {self.current_stage}")

            # ステージに応じた復旧処理
            if self.current_stage in ['sft_training', 'ppo_training']:
                logger.info("Resuming training from checkpoint")
                self.resume_training_from_checkpoint(checkpoint_data)
            elif self.current_stage == 'gguf_conversion':
                logger.info("Resuming GGUF conversion")
                self.resume_gguf_conversion()
            elif self.current_stage in ['ab_testing', 'hf_upload']:
                logger.info("Resuming evaluation/upload")
                self.resume_evaluation()

        except Exception as e:
            logger.error(f"Checkpoint recovery failed: {e}")

    def resume_training_from_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """トレーニングチェックポイントから再開"""
        # トレーニングタイプ判定
        if 'sft' in checkpoint_data.get('stage', ''):
            logger.info("Resuming SFT training")
            self.run_sft_training()
        elif 'ppo' in checkpoint_data.get('stage', ''):
            logger.info("Resuming PPO training")
            self.run_ppo_training()

    def resume_gguf_conversion(self):
        """GGUF変換再開"""
        self.run_gguf_conversion()

    def resume_evaluation(self):
        """評価・アップロード再開"""
        self.run_ab_testing()
        self.run_hf_upload()

    def emergency_save(self):
        """緊急保存（電源断時）"""
        try:
            emergency_data = {
                'session_id': self.session_id,
                'current_stage': self.current_stage,
                'timestamp': datetime.now().isoformat(),
                'power_interrupt': self.power_interrupt_detected,
                'system_info': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent
                }
            }

            # 最新チェックポイント取得
            last_checkpoint = None
            if self.rolling_checkpoints:
                last_checkpoint = str(self.rolling_checkpoints[-1])

            emergency_data['last_checkpoint'] = last_checkpoint

            # 緊急保存ファイル
            emergency_file = self.checkpoint_dir / f'emergency_save_{self.session_id}.json'
            with open(emergency_file, 'w', encoding='utf-8') as f:
                json.dump(emergency_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Emergency save completed: {emergency_file}")

        except Exception as e:
            logger.error(f"Emergency save failed: {e}")

    def save_session_state(self):
        """セッション状態保存"""
        try:
            session_data = {
                'session_id': self.session_id,
                'current_stage': self.current_stage,
                'timestamp': datetime.now().isoformat(),
                'last_checkpoint': str(self.rolling_checkpoints[-1]) if self.rolling_checkpoints else None,
                'power_state': self.power_monitor.get_current_power_state()
            }

            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Session state save failed: {e}")

    def setup_autostart(self):
        """Windows自動起動設定"""
        try:
            # スタートアップに登録
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0, winreg.KEY_SET_VALUE
            )

            script_path = str(Path(__file__).absolute())
            python_path = r"C:\Python312\python.exe"  # Pythonパスは適宜調整

            command = f'"{python_path}" "{script_path}" --autostart'

            winreg.SetValueEx(key, "SO8TAutomatedPipeline", 0, winreg.REG_SZ, command)
            winreg.CloseKey(key)

            logger.info("Autostart registered successfully")

        except Exception as e:
            logger.error(f"Failed to setup autostart: {e}")

    def create_rolling_checkpoint(self):
        """ローリングチェックポイント作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint_data = {
            'timestamp': timestamp,
            'stage': self.current_stage,
            'session_id': self.session_id,
            'power_state': self.power_monitor.get_current_power_state(),
            'system_info': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        }

        # チェックポイント保存
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{timestamp}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        # ローリング管理
        self.rolling_checkpoints.append(checkpoint_file)
        if len(self.rolling_checkpoints) > self.max_checkpoints:
            old_checkpoint = self.rolling_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

        # セッション状態保存
        self.save_session_state()

        logger.info(f"Rolling checkpoint created: {checkpoint_file}")

    def run_sft_training(self):
        """SFTトレーニング実行"""
        logger.info("Starting SFT training...")
        self.current_stage = "sft_training"

        try:
            config = create_sft_config()
            trainer = SO8TSFTTrainer(config)
            trainer, training_time, final_path = trainer.train()

            logger.info(f"SFT training completed in {training_time:.2f}s")
            return True, final_path

        except Exception as e:
            logger.error(f"SFT training failed: {e}")
            return False, None

    def run_ppo_training(self):
        """PPOトレーニング実行"""
        logger.info("Starting PPO training...")
        self.current_stage = "ppo_training"

        try:
            config = create_ppo_config()
            # 更新されたデータセットを使用
            config['ppo_dataset'] = 'data/train_ppo_integrated.jsonl'

            trainer = SO8TPPOTrainer(config)
            final_path = trainer.train()

            logger.info("PPO training completed")
            return True, final_path

        except Exception as e:
            logger.error(f"PPO training failed: {e}")
            return False, None

    def run_gguf_conversion(self):
        """GGUF変換実行"""
        logger.info("Starting GGUF conversion...")
        self.current_stage = "gguf_conversion"

        try:
            config = create_gguf_config()
            converter = SO8TGGUFConverter(config)
            result = converter.run_conversion_pipeline()

            logger.info("GGUF conversion completed")
            return True, result

        except Exception as e:
            logger.error(f"GGUF conversion failed: {e}")
            return False, None

    def run_ab_testing(self):
        """ABテスト実行"""
        logger.info("Starting AB testing...")
        self.current_stage = "ab_testing"

        try:
            config = create_benchmark_config()

            # ABテスト設定
            config['benchmark_types'] = [
                'mmlu', 'hellaswag', 'winogrande', 'arc_challenge',
                'truthfulqa', 'gsm8k', 'math', 'physics'
            ]
            config['elyza_sample_size'] = 100  # 全問テスト

            # ベンチマーク実行
            runner = SO8TBenchmarkRunner(config)
            benchmark_results = runner.run_all_benchmarks()

            # 統計分析（強化版）
            analyzer = EnhancedStatisticalAnalyzer(benchmark_results)
            analyzer.create_visualizations(benchmark_results['benchmarks'])
            analysis_report = analyzer.generate_enhanced_analysis_report(benchmark_results['benchmarks'])

            logger.info("AB testing completed")
            return True, (benchmark_results, analysis_report)

        except Exception as e:
            logger.error(f"AB testing failed: {e}")
        try:
            config = create_benchmark_config()
            hf_preparator = EnhancedHFPreparator(config)
            hf_dir = hf_preparator.prepare_enhanced_hf_structure(benchmark_results, analysis_report)

            # 自動アップロード
            repo_name = self.upload_to_huggingface(hf_dir)
            if repo_name is None:
                raise Exception("HuggingFace upload failed, repo_name is None")

            # アップロード完了後にクリーンアップ
            if self.should_cleanup_after_upload():
                self.final_cleanup()

            logger.info(f"HF upload completed: {repo_name}")
            return True, repo_name

        except Exception as e:
            logger.error(f"HF upload failed: {e}")
            return False, None

    def upload_to_huggingface(self, hf_dir: str):
        """HuggingFaceにアップロード"""
        try:
            # HF CLIを使用
            repo_name = f"so8t-benchmark-results-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            cmd = [
                "huggingface-cli", "upload", repo_name,
                hf_dir, "--repo-type", "dataset"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Successfully uploaded to HuggingFace: {repo_name}")
            return repo_name

        except Exception as e:
            logger.error(f"HF upload failed: {e}")
            return None

    def should_auto_upload(self) -> bool:
        """自動アップロード判定"""
        # 設定ファイルで制御可能
        return os.getenv('SO8T_AUTO_UPLOAD', 'true').lower() == 'true'

    def should_cleanup_after_upload(self) -> bool:
        """アップロード後クリーンアップ判定"""
        return os.getenv('SO8T_AUTO_CLEANUP', 'true').lower() == 'true'

    def final_cleanup(self):
        """最終クリーンアップ"""
        logger.info("Starting final cleanup...")

        try:
            # 自動起動設定の削除
            self.cleanup_autostart()

            # 作業ディレクトリのクリーンアップ（オプション）
            cleanup_dirs = [
                './checkpoints/auto_pipeline',
                './benchmark_results',
                './hf_upload'
            ]

            for dir_path in cleanup_dirs:
                if os.path.exists(dir_path):
                    import shutil
                    shutil.rmtree(dir_path)
                    logger.info(f"Cleaned up directory: {dir_path}")

            # 一時ファイルの削除
            temp_files = [
                'so8t_automated_pipeline.log',
                'so8t_benchmark_pipeline.log'
            ]

            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"Cleaned up temp file: {temp_file}")

            logger.info("Final cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def run_complete_pipeline(self):
        """完全パイプライン実行"""
        logger.info("Starting complete SO(8)T pipeline")

        self.is_running = True

        try:
            # 1. SFTトレーニング
            success, sft_path = self.run_sft_training()
            if not success:
                raise Exception("SFT training failed")

            # 2. PPOトレーニング
            success, ppo_path = self.run_ppo_training()
            if not success:
                raise Exception("PPO training failed")

            # 3. GGUF変換
            success, gguf_result = self.run_gguf_conversion()
            if not success:
                raise Exception("GGUF conversion failed")

            # 4. ABテスト
            success, (benchmark_results, analysis_report) = self.run_ab_testing()
            if not success:
                raise Exception("AB testing failed")

            # 5. HFアップロード
            success, hf_dir = self.run_hf_upload(benchmark_results, analysis_report)
            if not success:
                raise Exception("HF upload failed")

            # 6. 完了処理
            self.on_pipeline_completion()

            logger.info("Complete pipeline finished successfully")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.on_pipeline_failure(e)

        finally:
            self.is_running = False

    def on_pipeline_completion(self):
        """パイプライン完了時の処理"""
        logger.info("Pipeline completed successfully")

        # 完了音声通知
        try:
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-File", "scripts\\utils\\play_audio_notification.ps1"
            ], check=True)
        except Exception as e:
            logger.error(f"Audio notification failed: {e}")

        # 自動クリーンアップ（オプション）
        if os.getenv('SO8T_AUTO_CLEANUP', 'false').lower() == 'true':
            self.cleanup_autostart()

    def on_pipeline_failure(self, error: Exception):
        """パイプライン失敗時の処理"""
        logger.error(f"Pipeline failed: {error}")

        # エラー音声通知（異なる音）
        try:
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-c", "[System.Console]::Beep(800, 1000)"
            ], check=True)
        except Exception as e:
            logger.error(f"Error audio notification failed: {e}")

    def cleanup_autostart(self):
        """自動起動設定のクリーンアップ"""
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0, winreg.KEY_SET_VALUE
            )

            try:
                winreg.DeleteValue(key, "SO8TAutomatedPipeline")
                logger.info("Autostart entry removed")
            except FileNotFoundError:
                pass  # 既に削除されている

            winreg.CloseKey(key)

        except Exception as e:
            logger.error(f"Failed to cleanup autostart: {e}")

    def schedule_checkpoints(self):
        """チェックポイントスケジューリング"""
        def checkpoint_job():
            if self.is_running:
                self.create_rolling_checkpoint()

        # 3分ごとにチェックポイント
        schedule.every(self.checkpoint_interval).seconds.do(checkpoint_job)

        # スケジューラー実行スレッド
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(1)

        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()

    def monitor_power_state(self):
        """電源状態監視（強化版）"""
        def power_monitor_job():
            try:
                if self.power_monitor.is_power_restored():
                    logger.info("Power restored - checking recovery conditions")

                    # 自動再開条件チェック
                    if self.power_monitor.should_resume_after_power_restore():
                        logger.info("Auto-resume conditions met - restarting pipeline")
                        self.run_complete_pipeline()
                    else:
                        logger.info("Manual resume required - waiting for user input")

                # 定期的な電源状態ログ
                current_power = self.power_monitor.get_current_power_state()
                if not current_power and self.is_running:
                    logger.warning("Power interruption detected during pipeline execution")
                    self.power_interrupt_detected = True

            except Exception as e:
                logger.error(f"Power monitoring error: {e}")

        # 15秒ごとに電源状態チェック（高速化）
        schedule.every(15).seconds.do(power_monitor_job)

        # 電源状態レポート（5分ごと）
        def power_report_job():
            try:
                interrupt_count = self.power_monitor.get_power_interrupt_count()
                uptime = self.power_monitor.get_uptime_since_last_interrupt()

                if interrupt_count > 0 or uptime:
                    logger.info(f"Power status report - Interrupts: {interrupt_count}, Uptime: {uptime:.1f}s")
            except Exception as e:
                logger.error(f"Power report error: {e}")

        schedule.every(300).seconds.do(power_report_job)  # 5分ごと

    def start(self, autostart: bool = False, recover: bool = False):
        """パイプライン開始（強化版）"""
        logger.info("SO(8)T Automated Pipeline starting...")

        # スケジューラー開始
        self.schedule_checkpoints()
        self.monitor_power_state()

        if recover:
            # 手動復旧モード
            logger.info("Recovery mode - checking for checkpoints")
            if self.rolling_checkpoints:
                last_checkpoint = str(self.rolling_checkpoints[-1])
                self.recover_from_checkpoint(last_checkpoint)
            else:
                logger.info("No checkpoints found - starting fresh")
                self.run_complete_pipeline()

        elif autostart:
            # 電源投入時の自動開始
            logger.info("Autostart mode - checking power state and recovery conditions")

            # セッション復旧チェック
            if self.should_auto_recover() and self.rolling_checkpoints:
                logger.info("Session recovery available - attempting auto-recovery")
                last_checkpoint = str(self.rolling_checkpoints[-1])
                self.recover_from_checkpoint(last_checkpoint)
            elif self.power_monitor.should_resume_after_power_restore():
                logger.info("Power restoration detected - starting fresh pipeline")
                self.run_complete_pipeline()
            else:
                logger.info("Waiting for power restoration or manual trigger")

        else:
            # 手動開始
            logger.info("Manual start - running complete pipeline")
            self.run_complete_pipeline()

        # メインループ開始
        self.run_main_loop()

    def run_main_loop(self):
        """メイン監視ループ"""
        logger.info("Starting main monitoring loop")

        try:
            while True:
                # スケジュール実行
                schedule.run_pending()

                # 電源状態チェック
                if self.power_monitor.is_power_restored() and not self.is_running:
                    if self.power_monitor.should_resume_after_power_restore():
                        logger.info("Power restored - auto-resuming pipeline")
                        self.run_complete_pipeline()

                # CPU使用率チェック（過負荷防止）
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > 90:
                    logger.warning(f"High CPU usage detected: {cpu_percent}% - throttling")
                    time.sleep(5)  # 負荷軽減のため待機

                # メモリ使用率チェック
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 85:
                    logger.warning(f"High memory usage detected: {memory_percent}% - triggering GC")
                    import gc
                    gc.collect()

                # 短い待機
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Main loop interrupted by user")
            self.emergency_save()
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            self.emergency_save()
            raise

class EnhancedPowerStateMonitor:
    """強化電源状態監視"""

    def __init__(self):
        self.last_power_state = self.get_current_power_state()
        self.power_restored = False
        self.power_interrupt_history = []
        self.monitoring_active = True

        # 電源状態ログファイル
        self.power_log_file = Path('./logs/power_state.log')
        self.power_log_file.parent.mkdir(parents=True, exist_ok=True)

        # ログディレクトリ作成
        Path('./logs').mkdir(parents=True, exist_ok=True)

    def get_current_power_state(self) -> bool:
        """現在の電源状態取得（複数方法で確認）"""
        power_states = []

        # 方法1: powercfgコマンド
        try:
            result = subprocess.run(
                ["powercfg", "/getactivescheme"],
                capture_output=True, text=True, check=True, timeout=5
            )
            power_states.append("Power Scheme GUID" in result.stdout)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            power_states.append(True)  # エラー時はONと仮定

        # 方法2: システム情報確認
        try:
            result = subprocess.run(
                ["systeminfo"],
                capture_output=True, text=True, timeout=10
            )
            system_ok = "OS Name:" in result.stdout
            power_states.append(system_ok)
        except:
            power_states.append(True)

        # 方法3: プロセス実行確認
        try:
            # Python自身が実行できれば電源ON
            result = subprocess.run(
                ["python", "-c", "print('power_ok')"],
                capture_output=True, text=True, timeout=3
            )
            power_states.append(result.returncode == 0)
        except:
            power_states.append(False)

        # 多数決で決定
        power_on_count = sum(power_states)
        current_state = power_on_count >= len(power_states) // 2 + 1

        # ログ記録
        self.log_power_state(current_state, power_states)

        return current_state

    def log_power_state(self, current_state: bool, power_states: List[bool]):
        """電源状態ログ記録"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'power_state': current_state,
                'power_checks': power_states,
                'check_methods': len(power_states)
            }

            with open(self.power_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

        except Exception as e:
            # ログ失敗時は無視（電源監視のメイン機能を阻害しない）
            pass

    def is_power_restored(self) -> bool:
        """電源復旧判定（強化版）"""
        current_state = self.get_current_power_state()

        # 電源復旧判定
        if current_state and not self.last_power_state:
            self.power_restored = True
            self.power_interrupt_history.append({
                'interrupt_time': datetime.now().isoformat(),
                'restored': True
            })

            logger.info("Power restoration detected!")
            self.last_power_state = current_state
            return True

        # 電源断判定
        elif not current_state and self.last_power_state:
            self.power_interrupt_history.append({
                'interrupt_time': datetime.now().isoformat(),
                'restored': False
            })

            logger.warning("Power interruption detected!")

        self.last_power_state = current_state
        return False

    def get_power_interrupt_count(self) -> int:
        """電源断回数取得"""
        return len([h for h in self.power_interrupt_history if not h['restored']])

    def get_uptime_since_last_interrupt(self) -> Optional[float]:
        """最終電源断からの復旧時間（秒）"""
        if not self.power_interrupt_history:
            return None

        last_interrupt = None
        for history in reversed(self.power_interrupt_history):
            if not history['restored']:
                last_interrupt = history
                break

        if last_interrupt:
            interrupt_time = datetime.fromisoformat(last_interrupt['interrupt_time'])
            return (datetime.now() - interrupt_time).total_seconds()

        return None

    def should_resume_after_power_restore(self) -> bool:
        """電源復旧後の再開判定"""
        # 最終電源断から5分以内の復旧は自動再開
        uptime = self.get_uptime_since_last_interrupt()
        if uptime and uptime < 300:  # 5分
            return True

        # 電源断回数が3回未満は自動再開
        if self.get_power_interrupt_count() < 3:
            return True

        return False

# EnhancedStatisticalAnalyzerはso8t_benchmark_pipeline.pyからインポート済み

        # 効果量の追加指標
        analysis_report['enhanced_effect_sizes'] = self.calculate_enhanced_effect_sizes(benchmark_data)

        return analysis_report

    def perform_anova_analysis(self, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """ANOVA分析実行"""
        from scipy import stats

        anova_results = {}

        for benchmark_name in benchmark_data['model_a'].keys():
            scores_a = [benchmark_data['model_a'][benchmark_name].get('score', 0)]
            scores_b = [benchmark_data['model_b'][benchmark_name].get('score', 0)]

            # 一元配置分散分析（サンプルが少ないので参考値）
            try:
                f_stat, p_value = stats.f_oneway(scores_a, scores_b)
                anova_results[benchmark_name] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except:
                anova_results[benchmark_name] = {'error': 'Insufficient data for ANOVA'}

        return anova_results

    def perform_spherical_t_tests(self, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """球面上の仮定をしたt検定"""
        import numpy as np

        spherical_results = {}

        for benchmark_name in benchmark_data['model_a'].keys():
            scores_a = np.array([benchmark_data['model_a'][benchmark_name].get('score', 0)])
            scores_b = np.array([benchmark_data['model_b'][benchmark_name].get('score', 0)])

            # 球面座標変換（簡易版）
            # スコアを角度に変換（0-1 → 0-π）
            angles_a = scores_a * np.pi
            angles_b = scores_b * np.pi

            # 球面上でのt検定（参考実装）
            try:
                # 角度の差の検定
                angle_diff = angles_b - angles_a
                t_stat = np.mean(angle_diff) / (np.std(angle_diff) / np.sqrt(len(angle_diff)))

                # p値計算（近似）
                from scipy import stats
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(angle_diff) - 1))

                spherical_results[benchmark_name] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'angle_difference': float(np.mean(angle_diff))
                }
            except:
                spherical_results[benchmark_name] = {'error': 'Spherical t-test calculation failed'}

        return spherical_results

    def calculate_enhanced_effect_sizes(self, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """強化効果量計算"""
        enhanced_effects = {}

        for benchmark_name in benchmark_data['model_a'].keys():
            scores_a = [benchmark_data['model_a'][benchmark_name].get('score', 0)]
            scores_b = [benchmark_data['model_b'][benchmark_name].get('score', 0)]

            # 追加の効果量指標
            mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
            std_a, std_b = np.std(scores_a), np.std(scores_b)

            # Glass's delta（制御群の標準偏差を使用）
            glass_delta = (mean_b - mean_a) / std_a if std_a > 0 else 0

            # U3指標（非重複率）
            u3 = 1 - self.calculate_overlap(scores_a, scores_b)

            enhanced_effects[benchmark_name] = {
                'glass_delta': glass_delta,
                'u3_nonoverlap': u3,
                'probability_superiority': self.calculate_probability_superiority(scores_a, scores_b)
            }

        return enhanced_effects

    def calculate_overlap(self, scores_a: List[float], scores_b: List[float]) -> float:
        """スコア分布の重複率計算"""
        # 簡易実装：重複領域の割合
        min_score = min(min(scores_a), min(scores_b))
        max_score = max(max(scores_a), max(scores_b))

        # 重複領域を計算
        overlap_start = max(min(scores_a), min(scores_b))
        overlap_end = min(max(scores_a), max(scores_b))

        if overlap_end <= overlap_start:
            return 0.0

        total_range = max_score - min_score
        overlap_range = overlap_end - overlap_start

        return overlap_range / total_range if total_range > 0 else 0.0

    def calculate_probability_superiority(self, scores_a: List[float], scores_b: List[float]) -> float:
        """優越確率計算"""
        total_comparisons = len(scores_a) * len(scores_b)
        superiority_count = 0

        for score_b in scores_b:
            for score_a in scores_a:
                if score_b > score_a:
                    superiority_count += 1

        return superiority_count / total_comparisons if total_comparisons > 0 else 0.0

def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='SO(8)T Automated Pipeline')
    parser.add_argument('--autostart', action='store_true', help='Autostart mode (power-on recovery)')
    parser.add_argument('--recover', action='store_true', help='Manual recovery from last checkpoint')

    args = parser.parse_args()

    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('so8t_automated_pipeline.log'),
            logging.StreamHandler()
        ]
    )

    # パイプライン実行
    pipeline = SO8TAutomatedPipeline()
    pipeline.start(autostart=args.autostart, recover=args.recover)

if __name__ == "__main__":
    main()
