#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOONSHOT AEGIS 高度監視スクリプト

tqdmとloggingを使用したリアルタイム進捗監視
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import subprocess

# tqdm for progress bars
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('moonshot_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MoonshotMonitor:
    """MOONSHOT実行状態の高度監視クラス"""

    def __init__(self, log_file: str = "ab_test_automation.log"):
        self.log_file = Path(log_file)
        self.last_position = 0
        self.phases = {
            "Phase 0": "Environment Check",
            "Phase 1": "AEGIS Dataset Creation",
            "Phase 2": "lm-eval Integration",
            "Phase 3": "SO(8) RLPO Training",
            "Phase 4": "GGUF Conversion",
            "Phase 5": "A/B Testing Framework",
            "Phase 6": "Statistical Analysis",
            "Phase 7": "HF Upload Preparation",
            "Phase 8": "Autonomous Completion"
        }
        self.phase_status = {phase: "pending" for phase in self.phases.keys()}
        self.start_time = datetime.now()

        # tqdm progress bars
        self.overall_progress = tqdm(
            total=len(self.phases),
            desc="MOONSHOT Overall Progress",
            unit="phase",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]'
        )

        self.current_phase_bar = tqdm(
            desc="Current Phase",
            unit="%",
            bar_format='{desc}: {postfix}'
        )

        logger.info("MOONSHOT高度監視システムを開始しました")

    def get_log_content(self) -> str:
        """ログファイルの内容を取得"""
        if not self.log_file.exists():
            return ""

        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            return content

    def parse_phase_status(self, content: str) -> Dict[str, str]:
        """ログからPhaseの状態を解析"""
        status_updates = {}

        for phase in self.phases.keys():
            if phase in content:
                if "ERROR" in content or "FAILED" in content:
                    status_updates[phase] = "failed"
                elif "COMPLETED" in content or "SUCCESS" in content:
                    status_updates[phase] = "completed"
                else:
                    status_updates[phase] = "running"

        return status_updates

    def check_processes(self) -> Dict[str, any]:
        """実行中のPythonプロセスをチェック"""
        try:
            # PowerShellでプロセス情報を取得
            cmd = [
                "powershell.exe",
                "-Command",
                "Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.WorkingSet -gt 500MB } | Select-Object Id, CPU, @{Name='MemoryGB'; Expression={[math]::Round($_.WorkingSet/1GB, 2)}}, StartTime | ConvertTo-Json"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')

            if result.returncode == 0 and result.stdout.strip():
                import json
                processes = json.loads(result.stdout)
                if isinstance(processes, dict):
                    processes = [processes]

                return {
                    'count': len(processes),
                    'processes': processes,
                    'total_memory': sum(p.get('MemoryGB', 0) for p in processes),
                    'total_cpu': sum(p.get('CPU', 0) for p in processes)
                }
        except Exception as e:
            logger.warning(f"プロセスチェックエラー: {e}")

        return {'count': 0, 'processes': [], 'total_memory': 0, 'total_cpu': 0}

    def update_progress(self, status_updates: Dict[str, str]):
        """進捗バーを更新"""
        completed_count = 0
        current_running = None

        for phase, status in status_updates.items():
            if status == "completed":
                completed_count += 1
                self.phase_status[phase] = "completed"
            elif status == "running":
                current_running = phase
                self.phase_status[phase] = "running"
            elif status == "failed":
                self.phase_status[phase] = "failed"
                logger.error(f"Phase {phase} が失敗しました")

        # Overall progress update
        self.overall_progress.n = completed_count
        self.overall_progress.refresh()

        # Current phase info
        if current_running:
            phase_name = self.phases.get(current_running, current_running)
            self.current_phase_bar.set_description(f"Current: {phase_name}")
            self.current_phase_bar.set_postfix_str(f"Running {current_running}")
        else:
            self.current_phase_bar.set_postfix_str("Waiting...")

    def check_errors(self, content: str) -> List[str]:
        """エラーをチェック"""
        errors = []
        error_keywords = ["ERROR", "FAILED", "UnicodeEncodeError", "SyntaxError", "RuntimeError", "TypeError"]

        lines = content.split('\n')
        for line in lines:
            if any(keyword in line for keyword in error_keywords):
                errors.append(line.strip())

        return errors

    def get_system_info(self) -> Dict[str, any]:
        """システム情報を取得"""
        try:
            # GPU情報取得
            gpu_info = {}
            try:
                cmd = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu",
                      "--format=csv,noheader,nounits"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

                if result.returncode == 0 and result.stdout.strip():
                    data = result.stdout.strip().split(',')
                    gpu_info = {
                        'name': data[0],
                        'memory_used': int(data[1]),
                        'memory_total': int(data[2]),
                        'temperature': int(data[3]),
                        'utilization': int(data[4])
                    }
            except:
                pass

            # RAM情報取得
            ram_info = {}
            try:
                cmd = ["powershell.exe", "-Command",
                      "Get-WmiObject Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory | ConvertTo-Json"]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')

                if result.returncode == 0 and result.stdout.strip():
                    import json
                    data = json.loads(result.stdout.strip())
                    total_mb = int(data['TotalVisibleMemorySize']) / 1024
                    free_mb = int(data['FreePhysicalMemory']) / 1024
                    ram_info = {
                        'total_gb': round(total_mb / 1024, 1),
                        'free_gb': round(free_mb / 1024, 1),
                        'used_percent': round((1 - free_mb / total_mb) * 100, 1)
                    }
            except:
                pass

            return {
                'gpu': gpu_info,
                'ram': ram_info,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.warning(f"システム情報取得エラー: {e}")
            return {}

    def monitor_loop(self, check_interval: int = 30):
        """メイン監視ループ"""
        logger.info("MOONSHOT監視ループを開始します")
        print("\n" + "="*80)
        print("🚀 MOONSHOT AEGIS 高度監視システム")
        print("="*80)
        print(f"開始時刻: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"ログファイル: {self.log_file}")
        print(f"チェック間隔: {check_interval}秒")
        print("="*80)

        try:
            while True:
                # ログ内容を取得
                content = self.get_log_content()

                # Phase状態を解析
                status_updates = self.parse_phase_status(content)
                self.update_progress(status_updates)

                # エラーチェック
                errors = self.check_errors(content)
                for error in errors[-3:]:  # 最新の3つのエラーのみ表示
                    if "UnicodeEncodeError" in error:
                        logger.warning(f"絵文字エンコードエラー検知: {error}")
                    else:
                        logger.error(f"エラー検知: {error}")

                # プロセス情報
                proc_info = self.check_processes()
                if proc_info['count'] > 0:
                    logger.info(f"実行中プロセス: {proc_info['count']}個, "
                              f"総メモリ: {proc_info['total_memory']:.1f}GB, "
                              f"総CPU: {proc_info['total_cpu']:.1f}%")

                # システム情報
                sys_info = self.get_system_info()
                if sys_info.get('gpu'):
                    gpu = sys_info['gpu']
                    logger.info(f"GPU: {gpu['name']} | メモリ: {gpu['memory_used']}/{gpu['memory_total']}MB "
                              f"({gpu['utilization']}%) | 温度: {gpu['temperature']}°C")

                if sys_info.get('ram'):
                    ram = sys_info['ram']
                    logger.info(f"RAM: {ram['used_percent']}% 使用 "
                              f"({ram['free_gb']:.1f}GB 空き / {ram['total_gb']:.1f}GB 総計)")

                # 完了チェック
                completed_phases = sum(1 for status in self.phase_status.values() if status == "completed")
                if completed_phases >= len(self.phases):
                    logger.info("🎉 MOONSHOT完了を検知しました！")
                    break

                # 待機
                time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("監視をユーザーにより停止されました")
        except Exception as e:
            logger.error(f"監視ループエラー: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """クリーンアップ"""
        self.overall_progress.close()
        self.current_phase_bar.close()

        elapsed = datetime.now() - self.start_time
        completed = sum(1 for status in self.phase_status.values() if status == "completed")
        failed = sum(1 for status in self.phase_status.values() if status == "failed")

        print("\n" + "="*80)
        print("🏁 MOONSHOT高度監視システム終了")
        print("="*80)
        print(f"監視時間: {elapsed}")
        print(f"完了Phase: {completed}/{len(self.phases)}")
        if failed > 0:
            print(f"失敗Phase: {failed}")
        print("="*80)

        logger.info(f"監視完了 - 時間: {elapsed}, 完了: {completed}, 失敗: {failed}")

def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="MOONSHOT AEGIS Advanced Monitor")
    parser.add_argument("--log-file", default="ab_test_automation.log",
                       help="監視対象のログファイル")
    parser.add_argument("--interval", type=int, default=30,
                       help="チェック間隔（秒）")
    parser.add_argument("--quiet", action="store_true",
                       help="詳細ログ出力を抑制")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    monitor = MoonshotMonitor(args.log_file)
    monitor.monitor_loop(args.interval)

if __name__ == "__main__":
    main()
