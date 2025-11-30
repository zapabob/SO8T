#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Automatic Pipeline Runner
3分間隔でSO8Tパイプラインを実行し、ローリングチェックポイントを管理
"""

import os
import sys
import time
import signal
import atexit
from pathlib import Path
from datetime import datetime, timedelta
import logging
import subprocess
import psutil
import argparse

# SO8Tモジュールインポート
try:
    from utils.checkpoint_manager import RollingCheckpointManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/so8t_auto_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SO8TAutoPipelineRunner:
    """
    SO8T自動パイプライン実行クラス
    3分間隔で実行し、ローリングチェックポイントを管理
    """

    def __init__(self,
                 pipeline_script: str = "scripts/training/train_borea_phi35_so8t_ppo.py",
                 dataset_path: str = "data/so8t_quadruple_dataset.jsonl",
                 output_dir: str = "outputs/so8t_auto_pipeline",
                 checkpoint_dir: str = "checkpoints/so8t_rolling",
                 interval_minutes: int = 3,
                 max_checkpoints: int = 5,
                 max_iterations: int = None):
        """
        初期化

        Args:
            pipeline_script: 実行するパイプラインスクリプト
            dataset_path: データセットパス
            output_dir: 出力ディレクトリ
            checkpoint_dir: チェックポイントディレクトリ
            interval_minutes: 実行間隔（分）
            max_checkpoints: 最大チェックポイント数
            max_iterations: 最大実行回数（Noneで無限）
        """
        self.pipeline_script = Path(pipeline_script)
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.interval_minutes = interval_minutes
        self.max_checkpoints = max_checkpoints
        self.max_iterations = max_iterations

        # ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ローリングチェックポイントマネージャー
        self.checkpoint_manager = RollingCheckpointManager(
            base_dir=self.checkpoint_dir,
            max_keep=self.max_checkpoints,
            save_interval_sec=180  # 3分
        )

        # 実行状態
        self.running = False
        self.iteration_count = 0
        self.last_execution_time = None

        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # 終了時処理
        atexit.register(self._cleanup)

        logger.info("SO8T Auto Pipeline Runner initialized")
        logger.info(f"Pipeline script: {self.pipeline_script}")
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"Interval: {self.interval_minutes} minutes")
        logger.info(f"Max checkpoints: {self.max_checkpoints}")

    def start(self):
        """自動実行を開始"""
        logger.info("=== SO8T Auto Pipeline Started ===")
        self.running = True

        try:
            while self.running:
                # 最大実行回数チェック
                if self.max_iterations and self.iteration_count >= self.max_iterations:
                    logger.info(f"Reached maximum iterations ({self.max_iterations})")
                    break

                # システムリソースチェック
                if not self._check_system_ready():
                    logger.warning("System not ready, waiting...")
                    time.sleep(60)  # 1分待機
                    continue

                # パイプライン実行
                self._execute_pipeline()

                # 実行カウント更新
                self.iteration_count += 1
                self.last_execution_time = datetime.now()

                # 次回実行まで待機（最大実行回数に達していない場合）
                if self.running and (not self.max_iterations or self.iteration_count < self.max_iterations):
                    self._wait_for_next_execution()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            logger.info("=== SO8T Auto Pipeline Stopped ===")

    def stop(self):
        """実行を停止"""
        logger.info("Stopping SO8T Auto Pipeline...")
        self.running = False

    def _execute_pipeline(self):
        """パイプラインを実行"""
        execution_id = f"iteration_{self.iteration_count:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"=== Executing Pipeline {execution_id} ===")

        try:
            # 実行コマンド構築
            cmd = [
                sys.executable,  # Python実行ファイル
                str(self.pipeline_script),
                "--dataset_path", str(self.dataset_path),
                "--output_dir", str(self.output_dir / execution_id),
                "--checkpoint_dir", str(self.checkpoint_dir),
                "--execution_id", execution_id
            ]

            # 環境変数設定
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{os.getcwd()};{os.getcwd()}/scripts;{os.getcwd()}/utils"

            # プロセス実行
            logger.info(f"Running command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=os.getcwd()
            )

            # 実行監視
            stdout, stderr = process.communicate(timeout=3600)  # 1時間タイムアウト

            # 結果確認
            if process.returncode == 0:
                logger.info(f"✅ Pipeline {execution_id} completed successfully")

                # 成功時のチェックポイント保存
                self._save_execution_checkpoint(execution_id, "success")

            else:
                logger.error(f"[ERROR] Pipeline {execution_id} failed with code {process.returncode}")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")

                # 失敗時のチェックポイント保存
                self._save_execution_checkpoint(execution_id, "failed")

        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Pipeline {execution_id} timed out")
            process.kill()
        except Exception as e:
            logger.error(f"💥 Pipeline {execution_id} error: {e}")

    def _save_execution_checkpoint(self, execution_id: str, status: str):
        """実行チェックポイントを保存"""
        try:
            checkpoint_data = {
                "execution_id": execution_id,
                "timestamp": datetime.now().isoformat(),
                "iteration": self.iteration_count,
                "status": status,
                "output_dir": str(self.output_dir / execution_id),
                "system_info": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage('/').percent
                }
            }

            checkpoint_path = self.checkpoint_dir / f"execution_{execution_id}_{status}.json"

            import json
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

            logger.info(f"[INFO] Execution checkpoint saved: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to save execution checkpoint: {e}")

    def _check_system_ready(self) -> bool:
        """システムが実行準備ができているかチェック"""
        try:
            # CPU使用率チェック (90%未満)
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                logger.warning(f"High CPU usage: {cpu_percent}%")
                return False

            # メモリ使用率チェック (90%未満)
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent}%")
                return False

            # ディスク空き容量チェック (1GB以上)
            disk = psutil.disk_usage('/')
            if disk.free < 1 * 1024 * 1024 * 1024:  # 1GB
                logger.warning(f"Low disk space: {disk.free / (1024**3):.2f} GB")
                return False

            # GPUチェック (利用可能な場合)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_used = torch.cuda.memory_allocated(0)
                    gpu_free = gpu_memory - gpu_used

                    if gpu_free < 2 * 1024 * 1024 * 1024:  # 2GB
                        logger.warning(f"Low GPU memory: {gpu_free / (1024**3):.2f} GB")
                        return False
            except:
                pass  # GPUチェック失敗は無視

            return True

        except Exception as e:
            logger.error(f"System check failed: {e}")
            return False

    def _wait_for_next_execution(self):
        """次回実行まで待機"""
        wait_seconds = self.interval_minutes * 60

        logger.info(f"⏳ Waiting {self.interval_minutes} minutes until next execution...")
        logger.info(f"Next execution at: {datetime.now() + timedelta(minutes=self.interval_minutes)}")

        # 待機（シグナルで中断可能）
        for i in range(wait_seconds):
            if not self.running:
                break
            time.sleep(1)

    def _signal_handler(self, signum, frame):
        """シグナルハンドラー"""
        logger.info(f"Received signal {signum}")
        self.stop()

    def _cleanup(self):
        """クリーンアップ処理"""
        logger.info("Performing cleanup...")

        # 実行状態の保存
        final_status = {
            "total_iterations": self.iteration_count,
            "last_execution": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "final_timestamp": datetime.now().isoformat(),
            "status": "completed" if self.running else "stopped"
        }

        status_path = self.output_dir / "pipeline_status.json"
        import json
        with open(status_path, 'w', encoding='utf-8') as f:
            json.dump(final_status, f, indent=2, ensure_ascii=False)

        logger.info(f"Final status saved to {status_path}")

def main():
    parser = argparse.ArgumentParser(description="SO8T Automatic Pipeline Runner")

    # パイプライン設定
    parser.add_argument("--pipeline-script",
                       default="scripts/training/train_borea_phi35_so8t_ppo.py",
                       help="Pipeline script to execute")
    parser.add_argument("--dataset-path",
                       default="data/so8t_quadruple_dataset.jsonl",
                       help="Dataset path")
    parser.add_argument("--output-dir",
                       default="outputs/so8t_auto_pipeline",
                       help="Output directory")
    parser.add_argument("--checkpoint-dir",
                       default="checkpoints/so8t_rolling",
                       help="Checkpoint directory")

    # 実行設定
    parser.add_argument("--interval-minutes", type=int, default=3,
                       help="Execution interval in minutes")
    parser.add_argument("--max-checkpoints", type=int, default=5,
                       help="Maximum number of checkpoints to keep")
    parser.add_argument("--max-iterations", type=int, default=None,
                       help="Maximum number of iterations (None for infinite)")

    # デバッグ設定
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run mode (don't execute pipeline)")
    parser.add_argument("--single-run", action="store_true",
                       help="Single run mode (execute once and exit)")

    args = parser.parse_args()

    # ログディレクトリ作成
    Path("logs").mkdir(exist_ok=True)

    # 自動実行インスタンス作成
    runner = SO8TAutoPipelineRunner(
        pipeline_script=args.pipeline_script,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        interval_minutes=args.interval_minutes,
        max_checkpoints=args.max_checkpoints,
        max_iterations=1 if args.single_run else args.max_iterations
    )

    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        logger.info("System check result:")
        ready = runner._check_system_ready()
        logger.info(f"System ready: {ready}")
        return

    # 実行開始
    try:
        runner.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()

