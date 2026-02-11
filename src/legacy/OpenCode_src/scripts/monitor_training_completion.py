#!/usr/bin/env python3
"""
SO8T Training Completion Monitor
トレーニング完了を監視し、完了時に自動ワークフロー実行

使用方法:
python scripts/monitor_training_completion.py --start
python scripts/monitor_training_completion.py --stop
python scripts/monitor_training_completion.py --status
"""

import os
import sys
import json
import time
import signal
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import psutil

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent.parent
MONITOR_PID_FILE = PROJECT_ROOT / "logs" / "training_monitor.pid"
MONITOR_LOG_FILE = PROJECT_ROOT / "logs" / "training_monitor.log"

# ロギング設定
logging.basicConfig(
    filename=MONITOR_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingCompletionMonitor:
    """トレーニング完了監視クラス"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.checkpoints_dir = Path("D:/webdataset/checkpoints/training")
        self.check_interval = 300  # 5分間隔
        self.running = False
        self.last_check_time = None

    def is_monitor_running(self) -> bool:
        """モニターが実行中かどうかチェック"""
        if not MONITOR_PID_FILE.exists():
            return False

        try:
            with open(MONITOR_PID_FILE, 'r') as f:
                pid = int(f.read().strip())

            # PIDが有効かチェック
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                # プロセス名がpythonであることを確認
                if "python" in process.name().lower():
                    return True
        except (ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # 無効なPIDファイルは削除
        MONITOR_PID_FILE.unlink(missing_ok=True)
        return False

    def save_pid(self):
        """現在のPIDを保存"""
        MONITOR_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MONITOR_PID_FILE, 'w') as f:
            f.write(str(os.getpid()))

    def cleanup_pid(self):
        """PIDファイルを削除"""
        MONITOR_PID_FILE.unlink(missing_ok=True)

    def check_training_completion(self) -> list[Path]:
        """トレーニング完了をチェック"""
        completed_sessions = []

        if not self.checkpoints_dir.exists():
            return completed_sessions

        # すべてのトレーニングセッションをチェック
        session_dirs = list(self.checkpoints_dir.glob("so8t_*"))
        session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for session_dir in session_dirs:
            final_model = session_dir / "final_model"
            status_file = session_dir / "training_status.json"

            # final_modelが存在するかチェック
            if final_model.exists():
                # まだ処理されていないかチェック
                workflow_flag = session_dir / ".post_training_completed"
                if not workflow_flag.exists():
                    completed_sessions.append(session_dir)
                    logger.info(f"Found unprocessed completed training: {session_dir}")

        return completed_sessions

    def run_post_training_workflow(self, session_dir: Path) -> bool:
        """ポストトレーニングワークフロー実行"""
        try:
            logger.info(f"Starting post-training workflow for: {session_dir}")

            cmd = [
                sys.executable,
                "scripts/post_training_workflow.py",
                "--model-dir", str(session_dir)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                logger.info(f"Post-training workflow completed for: {session_dir}")
                # 完了フラグ作成
                workflow_flag = session_dir / ".post_training_completed"
                workflow_flag.touch()
                return True
            else:
                logger.error(f"Post-training workflow failed for {session_dir}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error running post-training workflow for {session_dir}: {e}")
            return False

    def monitor_loop(self):
        """監視ループ"""
        logger.info("Starting training completion monitor")
        self.running = True
        self.save_pid()

        def signal_handler(signum, frame):
            logger.info("Monitor stopped by signal")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self.running:
                try:
                    # トレーニング完了チェック
                    completed_sessions = self.check_training_completion()

                    # 各完了セッションに対してワークフロー実行
                    for session_dir in completed_sessions:
                        success = self.run_post_training_workflow(session_dir)
                        if success:
                            logger.info(f"Successfully processed completed training: {session_dir}")
                        else:
                            logger.warning(f"Failed to process completed training: {session_dir}")

                    self.last_check_time = datetime.now()

                except Exception as e:
                    logger.error(f"Error in monitor loop: {e}")

                # 次回チェックまで待機
                time.sleep(self.check_interval)

        finally:
            self.cleanup_pid()
            logger.info("Training completion monitor stopped")

    def start_monitor(self):
        """モニター開始"""
        if self.is_monitor_running():
            print("Training monitor is already running")
            return False

        print("Starting training completion monitor...")
        print("Monitor will check for completed training every 5 minutes")
        print("Press Ctrl+C to stop")

        # バックグラウンドで実行
        try:
            if os.name == 'nt':  # Windows
                subprocess.Popen([
                    sys.executable, __file__, "--monitor-loop"
                ], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:  # Unix-like
                subprocess.Popen([
                    sys.executable, __file__, "--monitor-loop"
                ])

            # PID保存まで待機
            time.sleep(2)

            if self.is_monitor_running():
                print("Training monitor started successfully")
                print("Monitor PID saved and running in background")
                return True
            else:
                print("Failed to start training monitor")
                return False

        except Exception as e:
            print(f"Error starting monitor: {e}")
            return False

    def stop_monitor(self):
        """モニター停止"""
        if not self.is_monitor_running():
            print("Training monitor is not running")
            return False

        try:
            with open(MONITOR_PID_FILE, 'r') as f:
                pid = int(f.read().strip())

            os.kill(pid, signal.SIGTERM)
            print("Training monitor stop signal sent")

            # 停止まで待機
            for _ in range(10):  # 最大10秒待機
                if not self.is_monitor_running():
                    print("Training monitor stopped successfully")
                    return True
                time.sleep(1)

            print("Training monitor did not stop gracefully, forcing termination")
            os.kill(pid, signal.SIGKILL)
            return True

        except Exception as e:
            print(f"Error stopping monitor: {e}")
            return False

    def show_status(self):
        """モニター状態表示"""
        running = self.is_monitor_running()

        print("=== SO8T Training Completion Monitor Status ===")
        print(f"Status: {'Running' if running else 'Stopped'}")

        if running:
            try:
                with open(MONITOR_PID_FILE, 'r') as f:
                    pid = int(f.read().strip())
                print(f"PID: {pid}")

                process = psutil.Process(pid)
                start_time = datetime.fromtimestamp(process.create_time())
                print(f"Started: {start_time}")
                print(f"CPU Usage: {process.cpu_percent()}%")
                print(f"Memory Usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")

            except Exception as e:
                print(f"Process info error: {e}")

        if self.last_check_time:
            print(f"Last Check: {self.last_check_time}")
        else:
            print("Last Check: Never")

        print(f"Check Interval: {self.check_interval} seconds")
        print(f"Log File: {MONITOR_LOG_FILE}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SO8T Training Completion Monitor")
    parser.add_argument("--start", action="store_true", help="Start the monitor")
    parser.add_argument("--stop", action="store_true", help="Stop the monitor")
    parser.add_argument("--status", action="store_true", help="Show monitor status")
    parser.add_argument("--monitor-loop", action="store_true", help="Internal: run monitor loop")

    args = parser.parse_args()

    monitor = TrainingCompletionMonitor()

    if args.monitor_loop:
        # モニターループ実行（内部使用）
        monitor.monitor_loop()
    elif args.start:
        monitor.start_monitor()
    elif args.stop:
        monitor.stop_monitor()
    elif args.status:
        monitor.show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
