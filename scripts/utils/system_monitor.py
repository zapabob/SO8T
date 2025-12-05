#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOONSHOT System Monitor Daemon
電源投入時自動起動 + システム監視 + バックアップ管理

監視機能：
1. システムリソース監視（CPU、メモリ、GPU）
2. プロセス監視（パイプライン実行状態）
3. 自動バックアップ（5分間隔）
4. 異常検知とリカバリー
5. ログ管理
"""

import os
import sys
import time
import json
import psutil
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SystemMonitor')

class SystemMonitor:
    """システム監視デーモン"""

    def __init__(self, daemon_mode: bool = False):
        self.daemon_mode = daemon_mode
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.backup_thread: Optional[threading.Thread] = None

        # 監視設定
        self.monitor_interval = 60  # 1分毎
        self.backup_interval = 300  # 5分毎
        self.max_backups = 10

        # パス設定
        self.project_root = Path(__file__).parent.parent.parent
        self.monitor_data_dir = self.project_root / "system_monitor_data"
        self.backup_dir = self.project_root / "system_backups"

        # ディレクトリ作成
        self.monitor_data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info("SystemMonitor initialized")

    def start_monitoring(self):
        """監視開始"""
        if self.monitoring:
            logger.warning("Monitoring already running")
            return

        self.monitoring = True
        logger.info("Starting system monitoring daemon")

        # 監視スレッド開始
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        # バックアップスレッド開始
        self.backup_thread = threading.Thread(target=self._backup_loop, daemon=True)
        self.backup_thread.start()

        # デーモンモードの場合はメインスレッド維持
        if self.daemon_mode:
            try:
                while self.monitoring:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                self.stop_monitoring()
        else:
            logger.info("Monitoring started in background")

    def stop_monitoring(self):
        """監視停止"""
        logger.info("Stopping system monitoring")
        self.monitoring = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        if self.backup_thread and self.backup_thread.is_alive():
            self.backup_thread.join(timeout=5)

        logger.info("System monitoring stopped")

    def _monitor_loop(self):
        """監視メインループ"""
        logger.info("Monitor loop started")

        while self.monitoring:
            try:
                # システム状態収集
                system_stats = self._collect_system_stats()

                # パイプライン状態確認
                pipeline_stats = self._check_pipeline_status()

                # 統合データ
                monitor_data = {
                    "timestamp": datetime.now().isoformat(),
                    "system_stats": system_stats,
                    "pipeline_stats": pipeline_stats,
                    "health_status": self._assess_system_health(system_stats, pipeline_stats)
                }

                # データ保存
                self._save_monitor_data(monitor_data)

                # 異常検知
                self._check_anomalies(monitor_data)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            time.sleep(self.monitor_interval)

        logger.info("Monitor loop stopped")

    def _backup_loop(self):
        """バックアップループ"""
        logger.info("Backup loop started")

        while self.monitoring:
            try:
                self._perform_backup()
                self._cleanup_old_backups()

            except Exception as e:
                logger.error(f"Backup loop error: {e}")

            time.sleep(self.backup_interval)

        logger.info("Backup loop stopped")

    def _collect_system_stats(self) -> Dict[str, Any]:
        """システム統計収集"""
        try:
            # CPU情報
            cpu_stats = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else None
            }

            # メモリ情報
            memory = psutil.virtual_memory()
            memory_stats = {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            }

            # ディスク情報
            disk = psutil.disk_usage('/')
            disk_stats = {
                "total": disk.total,
                "free": disk.free,
                "percent": disk.percent
            }

            # GPU情報（利用可能であれば）
            gpu_stats = {}
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_stats = {
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal,
                        "memory_free": gpu.memoryFree,
                        "memory_used": gpu.memoryUsed,
                        "memory_percent": gpu.memoryUtil * 100,
                        "gpu_percent": gpu.load * 100,
                        "temperature": gpu.temperature
                    }
            except ImportError:
                gpu_stats = {"gpu_available": False}
            except Exception as e:
                gpu_stats = {"gpu_error": str(e)}

            return {
                "cpu": cpu_stats,
                "memory": memory_stats,
                "disk": disk_stats,
                "gpu": gpu_stats
            }

        except Exception as e:
            logger.error(f"Failed to collect system stats: {e}")
            return {"error": str(e)}

    def _check_pipeline_status(self) -> Dict[str, Any]:
        """パイプライン実行状態確認"""
        pipeline_status = {}

        # 実行中のプロセス確認
        target_processes = [
            "python.exe", "py.exe", "auto_ab_test_pipeline.bat",
            "create_aegis_high_quality_dataset.py",
            "train_so8_phi35_adapter.py",
            "run_llama_cpp_ab_test.py"
        ]

        running_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and any(tp in proc.info['name'].lower() for tp in target_processes):
                    running_processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "cmdline": proc.info['cmdline'][:3] if proc.info['cmdline'] else None
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        pipeline_status["running_processes"] = running_processes
        pipeline_status["pipeline_active"] = len(running_processes) > 0

        # ログファイルの最終更新確認
        log_files = [
            "ab_test_automation.log",
            "automatic_aegis_pipeline.log",
            "system_monitor.log"
        ]

        log_status = {}
        for log_file in log_files:
            log_path = self.project_root / log_file
            if log_path.exists():
                mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
                age_minutes = (datetime.now() - mtime).total_seconds() / 60
                log_status[log_file] = {
                    "last_modified": mtime.isoformat(),
                    "age_minutes": age_minutes,
                    "is_recent": age_minutes < 30  # 30分以内
                }
            else:
                log_status[log_file] = {"status": "not_found"}

        pipeline_status["log_files"] = log_status

        return pipeline_status

    def _assess_system_health(self, system_stats: Dict, pipeline_stats: Dict) -> str:
        """システム健全性評価"""
        health_score = 100

        # CPU使用率チェック
        if system_stats.get("cpu", {}).get("cpu_percent", 0) > 90:
            health_score -= 20

        # メモリ使用率チェック
        if system_stats.get("memory", {}).get("percent", 0) > 90:
            health_score -= 25

        # ディスク使用率チェック
        if system_stats.get("disk", {}).get("percent", 0) > 95:
            health_score -= 30

        # GPUチェック
        gpu_stats = system_stats.get("gpu", {})
        if gpu_stats.get("memory_percent", 0) > 95:
            health_score -= 15

        # パイプラインチェック
        if not pipeline_stats.get("pipeline_active", False):
            health_score -= 10

        # ログファイルの鮮度チェック
        log_files = pipeline_stats.get("log_files", {})
        stale_logs = sum(1 for log_info in log_files.values()
                        if isinstance(log_info, dict) and not log_info.get("is_recent", True))
        health_score -= stale_logs * 5

        # ヘルスステータス決定
        if health_score >= 90:
            return "excellent"
        elif health_score >= 75:
            return "good"
        elif health_score >= 60:
            return "warning"
        elif health_score >= 40:
            return "critical"
        else:
            return "emergency"

    def _check_anomalies(self, monitor_data: Dict):
        """異常検知"""
        anomalies = []

        system_stats = monitor_data.get("system_stats", {})
        pipeline_stats = monitor_data.get("pipeline_stats", {})

        # 高リソース使用率検知
        if system_stats.get("memory", {}).get("percent", 0) > 95:
            anomalies.append("High memory usage (>95%)")

        if system_stats.get("cpu", {}).get("cpu_percent", 0) > 95:
            anomalies.append("High CPU usage (>95%)")

        # GPUメモリ不足検知
        gpu_stats = system_stats.get("gpu", {})
        if gpu_stats.get("memory_percent", 0) > 95:
            anomalies.append("High GPU memory usage (>95%)")

        # パイプライン停止検知
        if not pipeline_stats.get("pipeline_active", False):
            # ログファイルの鮮度で判断
            log_files = pipeline_stats.get("log_files", {})
            recent_logs = sum(1 for log_info in log_files.values()
                            if isinstance(log_info, dict) and log_info.get("is_recent", False))

            if recent_logs == 0:
                anomalies.append("Pipeline appears to be inactive")

        # 異常があればログ出力
        if anomalies:
            logger.warning(f"Anomalies detected: {anomalies}")

            # 緊急時対応
            health_status = monitor_data.get("health_status", "")
            if health_status in ["critical", "emergency"]:
                logger.error(f"Critical system health: {health_status}")
                self._emergency_response(anomalies)

    def _emergency_response(self, anomalies: List[str]):
        """緊急時対応"""
        logger.error("Initiating emergency response")

        # システム状態保存
        emergency_data = {
            "timestamp": datetime.now().isoformat(),
            "anomalies": anomalies,
            "system_snapshot": self._collect_system_stats()
        }

        emergency_file = self.monitor_data_dir / f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(emergency_file, 'w', encoding='utf-8') as f:
            json.dump(emergency_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Emergency data saved to {emergency_file}")

    def _save_monitor_data(self, data: Dict):
        """監視データ保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monitor_{timestamp}.json"
        filepath = self.monitor_data_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # 古いファイル削除（最新10個のみ保持）
        monitor_files = sorted(self.monitor_data_dir.glob("monitor_*.json"),
                              key=lambda x: x.stat().st_mtime, reverse=True)
        if len(monitor_files) > 10:
            for old_file in monitor_files[10:]:
                old_file.unlink()

    def _perform_backup(self):
        """バックアップ実行"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        backup_path = self.backup_dir / backup_name

        try:
            # 重要なファイルのバックアップ
            important_files = [
                "ab_test_automation.log",
                "automatic_aegis_pipeline.log",
                "system_monitor.log",
                "results/ab_test_results/",
                "models/",
                "checkpoints/"
            ]

            backup_path.mkdir(parents=True, exist_ok=True)

            for file_path in important_files:
                src_path = self.project_root / file_path
                dst_path = backup_path / file_path

                if src_path.exists():
                    if src_path.is_file():
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy2(src_path, dst_path)
                    elif src_path.is_dir():
                        import shutil
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

            logger.info(f"Backup completed: {backup_path}")

        except Exception as e:
            logger.error(f"Backup failed: {e}")

    def _cleanup_old_backups(self):
        """古いバックアップ削除"""
        backup_dirs = sorted(self.backup_dir.glob("backup_*"),
                           key=lambda x: x.stat().st_mtime, reverse=True)

        if len(backup_dirs) > self.max_backups:
            for old_backup in backup_dirs[self.max_backups:]:
                try:
                    import shutil
                    shutil.rmtree(old_backup)
                    logger.info(f"Removed old backup: {old_backup}")
                except Exception as e:
                    logger.error(f"Failed to remove old backup {old_backup}: {e}")

def main():
    """メイン実行関数"""
    import argparse

    parser = argparse.ArgumentParser(description="MOONSHOT System Monitor Daemon")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as daemon (continuous monitoring)")
    parser.add_argument("--once", action="store_true",
                       help="Run monitoring once and exit")

    args = parser.parse_args()

    monitor = SystemMonitor(daemon_mode=args.daemon)

    if args.once:
        # 一回のみ実行
        system_stats = monitor._collect_system_stats()
        pipeline_stats = monitor._check_pipeline_status()
        health = monitor._assess_system_health(system_stats, pipeline_stats)

        print("System Monitor - Single Run")
        print("=" * 40)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Health Status: {health}")
        print(f"CPU Usage: {system_stats.get('cpu', {}).get('cpu_percent', 'N/A')}%")
        print(f"Memory Usage: {system_stats.get('memory', {}).get('percent', 'N/A')}%")
        print(f"Pipeline Active: {pipeline_stats.get('pipeline_active', False)}")
        print(f"Running Processes: {len(pipeline_stats.get('running_processes', []))}")

    else:
        # デーモンモード
        print("Starting MOONSHOT System Monitor Daemon...")
        print("Press Ctrl+C to stop")

        try:
            monitor.start_monitoring()
        except KeyboardInterrupt:
            print("\nShutting down monitor daemon...")
            monitor.stop_monitoring()
        except Exception as e:
            logger.error(f"Monitor daemon error: {e}")
            monitor.stop_monitoring()
            sys.exit(1)

if __name__ == "__main__":
    main()