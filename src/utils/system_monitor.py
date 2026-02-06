#!/usr/bin/env python3
"""
AEGISシステム監視・自動復旧マネージャー
電源断やクラッシュからの自動復旧を担当
"""

import os
import sys
import time
import json
import psutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

class AEGISSystemMonitor:
    """AEGISシステムの完全自動監視・復旧マネージャー"""

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir) if project_dir else Path(__file__).parent.parent.parent
        self.monitor_file = self.project_dir / "system_monitor.json"
        self.log_file = self.project_dir / "system_monitor.log"
        self.check_interval = 60  # 1分間隔でチェック

        # 監視対象プロセス
        self.target_processes = [
            "python.exe",  # Pythonスクリプト
            "ollama.exe",  # Ollamaサーバー
        ]

        # システム状態
        self.system_state = {
            "last_check": None,
            "active_tasks": [],
            "failed_tasks": [],
            "system_health": "unknown",
            "restarts_count": 0,
            "last_restart": None,
            "uptime_start": datetime.now().isoformat()
        }

        self._load_state()
        self._setup_logging()

    def _setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            filename=str(self.log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("AEGIS_Monitor")

    def _load_state(self):
        """状態ファイル読み込み"""
        if self.monitor_file.exists():
            try:
                with open(self.monitor_file, 'r', encoding='utf-8') as f:
                    self.system_state.update(json.load(f))
            except Exception as e:
                self.logger.error(f"Failed to load monitor state: {e}")

    def _save_state(self):
        """状態ファイル保存"""
        try:
            with open(self.monitor_file, 'w', encoding='utf-8') as f:
                json.dump(self.system_state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save monitor state: {e}")

    def check_system_health(self) -> Dict:
        """システム全体の健康状態をチェック"""
        health_status = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "gpu_available": False,
            "gpu_memory": 0,
            "network_available": self._check_network(),
            "timestamp": datetime.now().isoformat()
        }

        # GPUチェック（PyTorchが利用可能な場合）
        try:
            import torch
            health_status["gpu_available"] = torch.cuda.is_available()
            if health_status["gpu_available"]:
                health_status["gpu_memory"] = torch.cuda.mem_get_info()[0] / 1024**3  # GB
        except:
            pass

        # 健康度判定
        if (health_status["cpu_usage"] < 90 and
            health_status["memory_usage"] < 90 and
            health_status["disk_usage"] < 95):
            health_status["overall_health"] = "good"
        elif (health_status["cpu_usage"] < 95 and
              health_status["memory_usage"] < 95):
            health_status["overall_health"] = "warning"
        else:
            health_status["overall_health"] = "critical"

        return health_status

    def _check_network(self) -> bool:
        """ネットワーク接続チェック"""
        try:
            import urllib.request
            urllib.request.urlopen('http://www.google.com', timeout=5)
            return True
        except:
            return False

    def check_running_processes(self) -> List[Dict]:
        """実行中の関連プロセスをチェック"""
        running_processes = []

        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                if proc.info['name'] and any(target in proc.info['name'].lower() for target in ['python', 'ollama']):
                    # AEGIS関連プロセスかチェック
                    if self._is_aegis_process(proc):
                        running_processes.append({
                            "pid": proc.info['pid'],
                            "name": proc.info['name'],
                            "cmdline": proc.info['cmdline'],
                            "cpu_percent": proc.info['cpu_percent'],
                            "memory_percent": proc.info['memory_percent'],
                            "is_aegis": True
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return running_processes

    def _is_aegis_process(self, proc) -> bool:
        """AEGIS関連プロセスかどうか判定"""
        try:
            cmdline = proc.info['cmdline'] or []
            cmdline_str = ' '.join(cmdline).lower()

            # AEGIS関連キーワード
            aegis_keywords = [
                'aegis', 'so8t', 'rlpo', 'sunshine', 'training',
                'convert_hf_to_gguf', 'task_manager', 'auto_aegis'
            ]

            return any(keyword in cmdline_str for keyword in aegis_keywords)
        except:
            return False

    def restart_failed_services(self):
        """失敗したサービスを再開"""
        self.logger.info("Checking for services to restart...")

        # Pythonプロセスが実行中かチェック
        running_python = any(
            proc['is_aegis'] for proc in self.check_running_processes()
        )

        if not running_python:
            self.logger.warning("No AEGIS Python processes running. Attempting restart...")
            self._restart_aegis_pipeline()

    def _restart_aegis_pipeline(self):
        """AEGISパイプラインを再開"""
        try:
            import subprocess

            pipeline_script = self.project_dir / "auto_aegis_pipeline.bat"
            if pipeline_script.exists():
                self.logger.info("Restarting AEGIS pipeline...")
                subprocess.Popen(
                    [str(pipeline_script)],
                    cwd=str(self.project_dir),
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                self.system_state["restarts_count"] += 1
                self.system_state["last_restart"] = datetime.now().isoformat()
                self._save_state()
                self.logger.info("AEGIS pipeline restarted successfully")
            else:
                self.logger.error("AEGIS pipeline script not found")

        except Exception as e:
            self.logger.error(f"Failed to restart AEGIS pipeline: {e}")

    def monitor_loop(self):
        """メイン監視ループ"""
        self.logger.info("Starting AEGIS system monitor...")

        while True:
            try:
                # システム健康チェック
                health = self.check_system_health()
                self.logger.info(f"System health: {health['overall_health']} "
                               f"(CPU: {health['cpu_usage']}%, "
                               f"Memory: {health['memory_usage']}%, "
                               f"Disk: {health['disk_usage']}%)")

                # プロセスチェック
                processes = self.check_running_processes()
                aegis_processes = [p for p in processes if p['is_aegis']]

                if aegis_processes:
                    self.logger.info(f"Active AEGIS processes: {len(aegis_processes)}")
                    for proc in aegis_processes[:3]:  # 最初の3つだけログ
                        self.logger.info(f"  PID {proc['pid']}: {proc['name']} "
                                       f"(CPU: {proc['cpu_percent']}%, "
                                       f"Memory: {proc['memory_percent']}%)")
                else:
                    self.logger.warning("No active AEGIS processes found")

                # 自動復旧チェック
                if health['overall_health'] != 'critical':
                    self.restart_failed_services()

                # 状態更新
                self.system_state["last_check"] = datetime.now().isoformat()
                self.system_state["system_health"] = health['overall_health']
                self._save_state()

            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")

            # チェック間隔待機
            time.sleep(self.check_interval)

    def get_system_status(self) -> Dict:
        """現在のシステム状態を取得"""
        health = self.check_system_health()
        processes = self.check_running_processes()

        return {
            "system_health": health,
            "running_processes": processes,
            "monitor_state": self.system_state,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='AEGIS System Monitor')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--restart-services', action='store_true', help='Restart failed services')

    args = parser.parse_args()

    monitor = AEGISSystemMonitor()

    if args.status:
        status = monitor.get_system_status()
        print("=== AEGIS System Status ===")
        print(json.dumps(status, indent=2, ensure_ascii=False))

    elif args.restart_services:
        print("🔄 Restarting failed services...")
        monitor.restart_failed_services()
        print("[OK] Service restart completed")

    elif args.daemon:
        print("🤖 Starting AEGIS system monitor daemon...")
        try:
            monitor.monitor_loop()
        except KeyboardInterrupt:
            print("\n🛑 Monitor stopped by user")
        except Exception as e:
            print(f"\n[NG] Monitor crashed: {e}")

    else:
        print("AEGIS System Monitor")
        print("Usage:")
        print("  python system_monitor.py --daemon        # Run as background monitor")
        print("  python system_monitor.py --status        # Show current system status")
        print("  python system_monitor.py --restart-services  # Restart failed services")

if __name__ == "__main__":
    main()


