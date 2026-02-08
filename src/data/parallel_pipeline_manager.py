#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完全バックグラウンド並列DeepResearch Webスクレイピングパイプラインマネージャー

10個のDeepResearch webスクレイピングパイプラインを完全バックグラウンドで並列実行します。

Usage:
    python scripts/data/parallel_pipeline_manager.py --num-instances 10 --daemon
"""

import sys
import logging
import asyncio
import argparse
import signal
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import psutil

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/parallel_pipeline_manager.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ParallelPipelineManager:
    """完全バックグラウンド並列パイプラインマネージャー"""
    
    def __init__(
        self,
        num_instances: int = 10,
        base_output_dir: Path = Path('D:/webdataset/processed'),
        base_port: int = 9222,
        daemon_mode: bool = False,
        auto_restart: bool = True,
        restart_delay: float = 60.0,
        max_memory_gb_per_instance: float = 8.0,
        max_cpu_percent_per_instance: float = 80.0
    ):
        """
        初期化
        
        Args:
            num_instances: 並列インスタンス数
            base_output_dir: ベース出力ディレクトリ
            base_port: ベースリモートデバッグポート
            daemon_mode: デーモンモード（完全バックグラウンド実行）
            auto_restart: 自動再起動
            restart_delay: 再起動待機時間（秒）
            max_memory_gb_per_instance: インスタンスあたりの最大メモリ使用量（GB）
            max_cpu_percent_per_instance: インスタンスあたりの最大CPU使用率（%）
        """
        self.num_instances = num_instances
        self.base_output_dir = Path(base_output_dir)
        self.base_port = base_port
        self.daemon_mode = daemon_mode
        self.auto_restart = auto_restart
        self.restart_delay = restart_delay
        self.max_memory_gb_per_instance = max_memory_gb_per_instance
        self.max_cpu_percent_per_instance = max_cpu_percent_per_instance
        
        self.running = True
        self.processes: Dict[int, subprocess.Popen] = {}
        self.instance_status: Dict[int, Dict] = {}
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._signal_handler)
        
        logger.info("="*80)
        logger.info("Parallel Pipeline Manager Initialized")
        logger.info("="*80)
        logger.info(f"Number of instances: {self.num_instances}")
        logger.info(f"Base output directory: {self.base_output_dir}")
        logger.info(f"Base port: {self.base_port}")
        logger.info(f"Daemon mode: {self.daemon_mode}")
        logger.info(f"Auto restart: {self.auto_restart}")
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー"""
        logger.info(f"[SIGNAL] Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.stop_all_instances()
    
    def get_instance_config(self, instance_id: int) -> Dict:
        """
        インスタンス設定を取得
        
        Args:
            instance_id: インスタンスID（0-9）
        
        Returns:
            config: インスタンス設定
        """
        return {
            'instance_id': instance_id,
            'output_dir': self.base_output_dir / f'parallel_instance_{instance_id:02d}',
            'remote_debugging_port': self.base_port + instance_id,
            'log_file': PROJECT_ROOT / 'logs' / f'parallel_instance_{instance_id:02d}.log',
            'pid_file': PROJECT_ROOT / 'logs' / f'parallel_instance_{instance_id:02d}.pid'
        }
    
    def start_instance(self, instance_id: int) -> bool:
        """
        インスタンスを起動
        
        Args:
            instance_id: インスタンスID
        
        Returns:
            True: 起動成功、False: 起動失敗
        """
        try:
            config = self.get_instance_config(instance_id)
            
            # 出力ディレクトリを作成
            config['output_dir'].mkdir(parents=True, exist_ok=True)
            
            # ログディレクトリを作成
            config['log_file'].parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"[INSTANCE {instance_id:02d}] Starting instance...")
            logger.info(f"[INSTANCE {instance_id:02d}] Output directory: {config['output_dir']}")
            logger.info(f"[INSTANCE {instance_id:02d}] Remote debugging port: {config['remote_debugging_port']}")
            
            # Pythonスクリプトを実行
            script_path = PROJECT_ROOT / 'scripts' / 'data' / 'parallel_deep_research_scraping.py'
            
            cmd = [
                sys.executable,
                str(script_path),
                '--output', str(config['output_dir']),
                '--num-browsers', '10',  # 各インスタンス内で10個のブラウザ
                '--use-cursor-browser',
                '--remote-debugging-port', str(config['remote_debugging_port']),
                '--max-memory-gb', str(self.max_memory_gb_per_instance),
                '--max-cpu-percent', str(self.max_cpu_percent_per_instance),
                '--use-so8t-control'
            ]
            
            # バックグラウンドで実行
            if self.daemon_mode:
                # Windows: CREATE_NEW_CONSOLEで完全バックグラウンド実行
                process = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=open(config['log_file'], 'w', encoding='utf-8'),
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0,
                    start_new_session=True
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=open(config['log_file'], 'w', encoding='utf-8'),
                    stderr=subprocess.STDOUT
                )
            
            # プロセス情報を保存
            self.processes[instance_id] = process
            self.instance_status[instance_id] = {
                'pid': process.pid,
                'start_time': datetime.now().isoformat(),
                'status': 'running',
                'restart_count': 0,
                'config': config
            }
            
            # PIDファイルを保存
            with open(config['pid_file'], 'w', encoding='utf-8') as f:
                f.write(str(process.pid))
            
            logger.info(f"[INSTANCE {instance_id:02d}] Started successfully (PID: {process.pid})")
            return True
        
        except Exception as e:
            logger.error(f"[INSTANCE {instance_id:02d}] Failed to start: {e}", exc_info=True)
            return False
    
    def stop_instance(self, instance_id: int):
        """
        インスタンスを停止
        
        Args:
            instance_id: インスタンスID
        """
        if instance_id not in self.processes:
            return
        
        try:
            process = self.processes[instance_id]
            logger.info(f"[INSTANCE {instance_id:02d}] Stopping instance (PID: {process.pid})...")
            
            # プロセスを終了
            process.terminate()
            
            # 5秒待機してから強制終了
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"[INSTANCE {instance_id:02d}] Process did not terminate, killing...")
                process.kill()
                process.wait()
            
            # プロセス情報を削除
            del self.processes[instance_id]
            self.instance_status[instance_id]['status'] = 'stopped'
            
            # PIDファイルを削除
            config = self.get_instance_config(instance_id)
            if config['pid_file'].exists():
                config['pid_file'].unlink()
            
            logger.info(f"[INSTANCE {instance_id:02d}] Stopped successfully")
        
        except Exception as e:
            logger.error(f"[INSTANCE {instance_id:02d}] Failed to stop: {e}", exc_info=True)
    
    def stop_all_instances(self):
        """すべてのインスタンスを停止"""
        logger.info("[MANAGER] Stopping all instances...")
        
        for instance_id in list(self.processes.keys()):
            self.stop_instance(instance_id)
        
        logger.info("[MANAGER] All instances stopped")
    
    def check_instance_status(self, instance_id: int) -> str:
        """
        インスタンスの状態をチェック
        
        Args:
            instance_id: インスタンスID
        
        Returns:
            status: 'running', 'stopped', 'failed'
        """
        if instance_id not in self.processes:
            return 'stopped'
        
        process = self.processes[instance_id]
        
        # プロセスの状態をチェック
        if process.poll() is None:
            # プロセスが実行中
            # 実際にプロセスが存在するか確認
            try:
                psutil.Process(process.pid)
                return 'running'
            except psutil.NoSuchProcess:
                return 'failed'
        else:
            # プロセスが終了
            return_code = process.returncode
            if return_code == 0:
                return 'stopped'
            else:
                return 'failed'
    
    def monitor_instances(self):
        """インスタンスを監視（自動再起動）"""
        while self.running:
            try:
                for instance_id in range(self.num_instances):
                    if not self.running:
                        break
                    
                    status = self.check_instance_status(instance_id)
                    
                    if status == 'failed' and self.auto_restart:
                        logger.warning(f"[INSTANCE {instance_id:02d}] Instance failed, restarting...")
                        
                        # 古いプロセス情報を削除
                        if instance_id in self.processes:
                            del self.processes[instance_id]
                        
                        # 再起動カウントを増加
                        if instance_id in self.instance_status:
                            self.instance_status[instance_id]['restart_count'] += 1
                        else:
                            self.instance_status[instance_id] = {'restart_count': 0}
                        
                        # 再起動
                        if self.instance_status[instance_id]['restart_count'] < 10:  # 最大10回再起動
                            time.sleep(self.restart_delay)
                            self.start_instance(instance_id)
                        else:
                            logger.error(f"[INSTANCE {instance_id:02d}] Max restart count reached, stopping...")
                    
                    # 状態を更新
                    if instance_id in self.instance_status:
                        self.instance_status[instance_id]['status'] = status
                
                # 5秒待機
                time.sleep(5)
            
            except KeyboardInterrupt:
                logger.info("[MANAGER] Keyboard interrupt received, stopping...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"[MANAGER] Error in monitor loop: {e}", exc_info=True)
                time.sleep(5)
    
    def _convert_to_json_serializable(self, obj):
        """オブジェクトをJSONシリアライズ可能な形式に変換"""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # その他のオブジェクトは文字列に変換
            return str(obj)
    
    def save_status(self):
        """状態を保存"""
        status_file = PROJECT_ROOT / 'logs' / 'parallel_pipeline_status.json'
        
        status_data = {
            'timestamp': datetime.now().isoformat(),
            'num_instances': self.num_instances,
            'instances': {}
        }
        
        for instance_id in range(self.num_instances):
            if instance_id in self.instance_status:
                instance_data = self.instance_status[instance_id].copy()
                instance_data['current_status'] = self.check_instance_status(instance_id)
                
                # すべての値をJSONシリアライズ可能な形式に変換
                instance_data = self._convert_to_json_serializable(instance_data)
                
                status_data['instances'][instance_id] = instance_data
        
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[MANAGER] Status saved to {status_file}")
    
    def run(self):
        """マネージャーを実行"""
        logger.info("[MANAGER] Starting parallel pipeline manager...")
        
        # すべてのインスタンスを起動
        for instance_id in range(self.num_instances):
            if not self.running:
                break
            
            self.start_instance(instance_id)
            time.sleep(2)  # インスタンス間の起動間隔
        
        logger.info(f"[MANAGER] All {self.num_instances} instances started")
        
        # 状態を保存
        self.save_status()
        
        # 監視ループを開始
        if self.daemon_mode:
            logger.info("[MANAGER] Running in daemon mode, starting monitor loop...")
            self.monitor_instances()
        else:
            logger.info("[MANAGER] Running in foreground mode, press Ctrl+C to stop...")
            try:
                while self.running:
                    time.sleep(10)
                    self.save_status()
            except KeyboardInterrupt:
                logger.info("[MANAGER] Keyboard interrupt received, stopping...")
                self.running = False
        
        # すべてのインスタンスを停止
        self.stop_all_instances()
        
        logger.info("[MANAGER] Parallel pipeline manager stopped")


def check_admin_privileges() -> bool:
    """管理者権限をチェック"""
    try:
        # Windowsの場合、net sessionコマンドで管理者権限をチェック
        result = subprocess.run(
            ["net", "session"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        # 非Windows環境やエラー時はFalseを返す
        return False


def setup_auto_start():
    """Windowsタスクスケジューラに自動実行タスクを登録"""
    logger.info("="*80)
    logger.info("Setting up auto-start task")
    logger.info("="*80)
    
    # 管理者権限チェック
    if not check_admin_privileges():
        logger.error("[ERROR] Administrator privileges required")
        logger.error("Please run this script as administrator")
        logger.error("Right-click and select 'Run as administrator'")
        return False
    
    task_name = 'SO8T-ParallelPipelineManager-AutoStart'
    
    # タスクスケジューラ用バッチファイルのパス
    # バッチファイルを使用することで、261文字制限を回避
    batch_file_path = PROJECT_ROOT / 'scripts' / 'data' / 'parallel_pipeline_manager_autostart.bat'
    
    if not batch_file_path.exists():
        logger.error(f"Batch file not found: {batch_file_path}")
        logger.error("Please ensure parallel_pipeline_manager_autostart.bat exists")
        return False
    
    # タスクスケジューラから呼び出されるコマンド（バッチファイルを実行）
    # パスに#が含まれている場合はエラー
    batch_file_path_str = str(batch_file_path)
    if '#' in batch_file_path_str:
        logger.error(f"[ERROR] Invalid character '#' in batch file path: {batch_file_path_str}")
        logger.error("[ERROR] Please remove '#' from the path")
        return False
    
    task_command = f'"{batch_file_path_str}"'
    
    # コマンド内に#が含まれていないか確認
    if '#' in task_command:
        logger.error(f"[ERROR] Invalid character '#' found in task command: {task_command}")
        logger.error("[ERROR] Please check batch file path for '#' characters")
        return False
    
    # 既存のタスクを削除
    try:
        result = subprocess.run(
            ["schtasks", "/query", "/tn", task_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"Removing existing task: {task_name}")
            subprocess.run(
                ["schtasks", "/delete", "/tn", task_name, "/f"],
                check=False
            )
    except Exception as e:
        logger.warning(f"Failed to check/remove existing task: {e}")
    
    # 新しいタスクを作成
    logger.info(f"Creating new task: {task_name}")
    
    # /delayパラメータは/sc onstartと一緒に使えないため、削除
    # 遅延処理はPythonスクリプト内で実装
    # /ruオプションを省略すると現在のユーザーでタスクが作成される（アクセス権限の問題を回避）
    create_cmd = [
        "schtasks", "/create",
        "/tn", task_name,
        "/tr", task_command,
        "/sc", "onstart",  # システム起動時
        "/rl", "highest",  # 最高権限
        "/f"
    ]
    
    try:
        result = subprocess.run(create_cmd, capture_output=True, text=True, check=True)
        logger.info("[OK] Task created successfully")
        logger.info(f"Task name: {task_name}")
        logger.info(f"Trigger: On system start")
        logger.info(f"Command: {task_command}")
        
        subprocess.run(["schtasks", "/query", "/tn", task_name, "/fo", "list", "/v"], check=False)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Failed to create task: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        if "アクセスが拒否されました" in e.stderr or "Access is denied" in e.stderr:
            logger.error("[ERROR] Access denied. Administrator privileges required.")
            logger.error("Please run this script as administrator:")
            logger.error("  Right-click and select 'Run as administrator'")
            logger.error("  Or run: py -3 scripts\\data\\parallel_pipeline_manager.py --setup")
        return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Parallel Pipeline Manager")
    parser.add_argument(
        '--num-instances',
        type=int,
        default=10,
        help='Number of parallel instances'
    )
    parser.add_argument(
        '--base-output',
        type=Path,
        default=Path('D:/webdataset/processed'),
        help='Base output directory'
    )
    parser.add_argument(
        '--base-port',
        type=int,
        default=9222,
        help='Base remote debugging port'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        default=False,
        help='Run as daemon (background process)'
    )
    parser.add_argument(
        '--auto-restart',
        action='store_true',
        default=True,
        help='Auto restart failed instances'
    )
    parser.add_argument(
        '--restart-delay',
        type=float,
        default=60.0,
        help='Delay between restarts (seconds)'
    )
    parser.add_argument(
        '--max-memory-gb',
        type=float,
        default=8.0,
        help='Maximum memory usage per instance (GB)'
    )
    parser.add_argument(
        '--max-cpu-percent',
        type=float,
        default=80.0,
        help='Maximum CPU usage per instance (%)'
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Setup Windows Task Scheduler auto-start task'
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help='Run the pipeline manager (called by task scheduler)'
    )
    
    args = parser.parse_args()
    
    if args.setup:
        # タスクスケジューラ登録
        success = setup_auto_start()
        if success:
            logger.info("[OK] Auto-start task setup completed")
            return 0
        else:
            logger.error("[ERROR] Auto-start task setup failed")
            return 1
    
    elif args.run:
        # タスクスケジューラから呼び出された場合の処理
        # システム起動時の遅延処理（60秒待機）
        delay_seconds = 60
        logger.info(f"Waiting {delay_seconds} seconds before starting pipeline manager (system startup delay)...")
        time.sleep(delay_seconds)
        
        # デーモンモードでマネージャーを起動
        manager = ParallelPipelineManager(
            num_instances=args.num_instances,
            base_output_dir=args.base_output,
            base_port=args.base_port,
            daemon_mode=True,  # タスクスケジューラから呼び出される場合は常にデーモンモード
            auto_restart=args.auto_restart,
            restart_delay=args.restart_delay,
            max_memory_gb_per_instance=args.max_memory_gb,
            max_cpu_percent_per_instance=args.max_cpu_percent
        )
        
        try:
            manager.run()
            logger.info("Parallel Pipeline Manager completed successfully!")
            return 0
            
        except KeyboardInterrupt:
            logger.warning("[WARNING] Pipeline manager interrupted by user")
            return 1
        except Exception as e:
            logger.error(f"[FAILED] Pipeline manager failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    else:
        # 通常の実行
        # マネージャー作成
        manager = ParallelPipelineManager(
            num_instances=args.num_instances,
            base_output_dir=args.base_output,
            base_port=args.base_port,
            daemon_mode=args.daemon,
            auto_restart=args.auto_restart,
            restart_delay=args.restart_delay,
            max_memory_gb_per_instance=args.max_memory_gb,
            max_cpu_percent_per_instance=args.max_cpu_percent
        )
        
        # 実行
        manager.run()


if __name__ == "__main__":
    main()

