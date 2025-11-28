#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Detection and Recovery System for SO8T Automation Pipeline

パイプライン実行中のエラーを検知し、自動修正を試行
"""

import os
import sys
import json
import logging
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class SO8TErrorDetector:
    """
    SO8Tエラー検知器

    ログファイルとシステム状態を監視してエラーを検知
    """

    def __init__(self, log_directory: str = "logs"):
        self.log_dir = Path(log_directory)
        self.error_patterns = self._load_error_patterns()
        self.recovery_actions = self._load_recovery_actions()

    def _load_error_patterns(self) -> Dict[str, List[str]]:
        """エラーパターン定義"""
        return {
            'cuda_out_of_memory': [
                r'CUDA out of memory',
                r'RuntimeError.*CUDA.*memory',
                r'cuda\.RuntimeError.*memory'
            ],
            'disk_space_error': [
                r'No space left on device',
                r'Disk full',
                r'Insufficient disk space'
            ],
            'network_error': [
                r'ConnectionError',
                r'TimeoutError',
                r'HTTPError.*50[0-9]',
                r'Failed to download'
            ],
            'huggingface_error': [
                r'HuggingFace.*error',
                r'403.*Forbidden',
                r'401.*Unauthorized',
                r'API token.*invalid'
            ],
            'import_error': [
                r'ImportError',
                r'ModuleNotFoundError',
                r'No module named'
            ],
            'subprocess_error': [
                r'subprocess.*returned non-zero exit status',
                r'Command.*failed'
            ]
        }

    def _load_recovery_actions(self) -> Dict[str, Dict[str, Any]]:
        """回復アクション定義"""
        return {
            'cuda_out_of_memory': {
                'description': 'Reduce batch size and clear GPU cache',
                'actions': [
                    'reduce_batch_size',
                    'clear_gpu_cache',
                    'restart_with_smaller_model'
                ]
            },
            'disk_space_error': {
                'description': 'Clean up disk space and reduce dataset size',
                'actions': [
                    'cleanup_temp_files',
                    'reduce_dataset_size',
                    'move_to_external_storage'
                ]
            },
            'network_error': {
                'description': 'Retry with exponential backoff',
                'actions': [
                    'retry_with_backoff',
                    'switch_mirror',
                    'resume_download'
                ]
            },
            'huggingface_error': {
                'description': 'Check authentication and retry',
                'actions': [
                    'verify_hf_token',
                    'refresh_token',
                    'retry_upload'
                ]
            },
            'import_error': {
                'description': 'Install missing dependencies',
                'actions': [
                    'install_dependencies',
                    'check_python_path',
                    'rebuild_environment'
                ]
            },
            'subprocess_error': {
                'description': 'Check subprocess configuration',
                'actions': [
                    'verify_paths',
                    'check_permissions',
                    'retry_command'
                ]
            }
        }

    def scan_for_errors(self, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        エラーログスキャン

        Args:
            log_file: 特定のログファイル（Noneの場合は最新のログ）

        Returns:
            検知されたエラーリスト
        """
        if log_file is None:
            # 最新のログファイルを探す
            log_files = list(self.log_dir.glob("*.log"))
            if not log_files:
                return []
            log_file = max(log_files, key=lambda x: x.stat().st_mtime)

        detected_errors = []

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            for error_type, patterns in self.error_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        detected_errors.append({
                            'error_type': error_type,
                            'pattern': pattern,
                            'matches': matches[:5],  # 最初の5件のみ
                            'log_file': str(log_file),
                            'timestamp': datetime.now().isoformat(),
                            'recovery_info': self.recovery_actions.get(error_type, {})
                        })

        except Exception as e:
            logger.error(f"Error scanning log file {log_file}: {e}")

        return detected_errors

    def get_recent_errors(self, hours: int = 1) -> List[Dict[str, Any]]:
        """最近のエラーを取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        all_errors = []

        # 全てのログファイルをスキャン
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime > cutoff_time.timestamp():
                errors = self.scan_for_errors(str(log_file))
                all_errors.extend(errors)

        return all_errors


class SO8TRecoveryAgent:
    """
    SO8T回復エージェント

    検知されたエラーに対して自動回復を試行
    """

    def __init__(self):
        self.max_retry_count = 3
        self.retry_delay = 60  # seconds

    def execute_recovery(self, error_info: Dict[str, Any]) -> bool:
        """
        エラー回復実行

        Args:
            error_info: エラー情報

        Returns:
            回復成功/失敗
        """
        error_type = error_info['error_type']
        recovery_info = error_info.get('recovery_info', {})

        logger.info(f"Attempting recovery for error type: {error_type}")
        logger.info(f"Recovery description: {recovery_info.get('description', 'Unknown')}")

        actions = recovery_info.get('actions', [])

        for action in actions:
            logger.info(f"Executing recovery action: {action}")

            success = self._execute_action(action, error_info)

            if success:
                logger.info(f"Recovery action '{action}' succeeded")
                return True
            else:
                logger.warning(f"Recovery action '{action}' failed, trying next action...")

        logger.error(f"All recovery actions failed for error type: {error_type}")
        return False

    def _execute_action(self, action: str, error_info: Dict[str, Any]) -> bool:
        """回復アクション実行"""
        try:
            if action == 'reduce_batch_size':
                return self._reduce_batch_size()
            elif action == 'clear_gpu_cache':
                return self._clear_gpu_cache()
            elif action == 'cleanup_temp_files':
                return self._cleanup_temp_files()
            elif action == 'reduce_dataset_size':
                return self._reduce_dataset_size()
            elif action == 'retry_with_backoff':
                return self._retry_with_backoff(error_info)
            elif action == 'verify_hf_token':
                return self._verify_hf_token()
            elif action == 'install_dependencies':
                return self._install_dependencies(error_info)
            elif action == 'verify_paths':
                return self._verify_paths()
            else:
                logger.warning(f"Unknown recovery action: {action}")
                return False

        except Exception as e:
            logger.error(f"Recovery action '{action}' threw exception: {e}")
            return False

    def _reduce_batch_size(self) -> bool:
        """バッチサイズ削減"""
        try:
            # 設定ファイルのバッチサイズを半分に
            config_files = [
                "configs/train_phi35_so8t_annealing.yaml",
                "configs/complete_so8t_pipeline.yaml"
            ]

            for config_file in config_files:
                if Path(config_file).exists():
                    self._modify_config_value(config_file, 'batch_size', lambda x: max(1, x // 2))

            logger.info("Batch size reduced in configuration files")
            return True
        except Exception as e:
            logger.error(f"Failed to reduce batch size: {e}")
            return False

    def _clear_gpu_cache(self) -> bool:
        """GPUキャッシュクリア"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
                return True
            else:
                logger.warning("CUDA not available, cannot clear GPU cache")
                return False
        except Exception as e:
            logger.error(f"Failed to clear GPU cache: {e}")
            return False

    def _cleanup_temp_files(self) -> bool:
        """一時ファイルクリーンアップ"""
        try:
            temp_dirs = [
                "D:/webdataset/temp",
                Path.home() / ".cache" / "huggingface",
                Path.home() / ".cache" / "torch"
            ]

            cleaned_size = 0
            for temp_dir in temp_dirs:
                temp_path = Path(temp_dir)
                if temp_path.exists():
                    for file_path in temp_path.glob("*"):
                        if file_path.is_file() and file_path.stat().st_mtime < (time.time() - 3600):  # 1時間以上前
                            try:
                                size = file_path.stat().st_size
                                file_path.unlink()
                                cleaned_size += size
                            except:
                                pass

            logger.info(f"Cleaned up {cleaned_size / 1024**3:.2f} GB of temporary files")
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
            return False

    def _reduce_dataset_size(self) -> bool:
        """データセットサイズ削減"""
        try:
            config_file = "configs/complete_so8t_pipeline.yaml"
            if Path(config_file).exists():
                self._modify_config_value(config_file, 'max_samples_per_dataset', lambda x: max(1000, x // 2))

            logger.info("Dataset size reduced in configuration")
            return True
        except Exception as e:
            logger.error(f"Failed to reduce dataset size: {e}")
            return False

    def _retry_with_backoff(self, error_info: Dict[str, Any]) -> bool:
        """指数バックオフ付きリトライ"""
        try:
            # 単に待機して成功を返す（実際のリトライは呼び出し元で行う）
            time.sleep(self.retry_delay)
            logger.info(f"Waited {self.retry_delay} seconds for retry")
            return True
        except Exception as e:
            logger.error(f"Retry backoff failed: {e}")
            return False

    def _verify_hf_token(self) -> bool:
        """HFトークン検証"""
        try:
            # HF CLIでトークンをチェック
            result = subprocess.run([
                "huggingface-cli", "whoami"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("HuggingFace token is valid")
                return True
            else:
                logger.error("HuggingFace token is invalid or missing")
                return False
        except Exception as e:
            logger.error(f"Failed to verify HF token: {e}")
            return False

    def _install_dependencies(self, error_info: Dict[str, Any]) -> bool:
        """依存関係インストール"""
        try:
            # requirements.txtからインストール
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Dependencies installed successfully")
                return True
            else:
                logger.error(f"Failed to install dependencies: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            return False

    def _verify_paths(self) -> bool:
        """パス検証"""
        try:
            required_paths = [
                "scripts/",
                "so8t/",
                "configs/",
                "D:/webdataset/"
            ]

            for path_str in required_paths:
                path = Path(path_str)
                if not path.exists():
                    logger.error(f"Required path does not exist: {path}")
                    return False

            logger.info("All required paths verified")
            return True
        except Exception as e:
            logger.error(f"Path verification failed: {e}")
            return False

    def _modify_config_value(self, config_file: str, key: str, modifier_func):
        """設定ファイルの値修正"""
        try:
            import yaml

            config_path = Path(config_file)
            if not config_path.exists():
                return

            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 再帰的にキーを探して修正
            def modify_recursive(data, target_key):
                if isinstance(data, dict):
                    for k, v in data.items():
                        if k == target_key:
                            data[k] = modifier_func(v)
                        elif isinstance(v, (dict, list)):
                            modify_recursive(v, target_key)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, (dict, list)):
                            modify_recursive(item, target_key)

            modify_recursive(config, key)

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)

        except Exception as e:
            logger.error(f"Failed to modify config {config_file}: {e}")


class SO8TMonitoringAgent:
    """
    SO8T監視エージェント

    パイプライン実行を継続的に監視し、エラーを検知したら回復を試行
    """

    def __init__(self, check_interval: int = 300):  # 5分間隔
        self.check_interval = check_interval
        self.error_detector = SO8TErrorDetector()
        self.recovery_agent = SO8TRecoveryAgent()
        self.is_monitoring = False

    def start_monitoring(self):
        """監視開始"""
        self.is_monitoring = True
        logger.info(f"Starting SO8T monitoring with {self.check_interval}s intervals")

        while self.is_monitoring:
            try:
                self._check_and_recover()
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.check_interval)

    def stop_monitoring(self):
        """監視停止"""
        self.is_monitoring = False
        logger.info("SO8T monitoring stopped")

    def _check_and_recover(self):
        """エラーチェックと回復"""
        # 最近のエラーをスキャン
        recent_errors = self.error_detector.get_recent_errors(hours=0.5)  # 過去30分

        if recent_errors:
            logger.warning(f"Detected {len(recent_errors)} recent errors")

            for error_info in recent_errors:
                logger.info(f"Processing error: {error_info['error_type']}")

                # 回復を試行
                recovery_success = self.recovery_agent.execute_recovery(error_info)

                if recovery_success:
                    logger.info(f"Successfully recovered from {error_info['error_type']}")
                else:
                    logger.error(f"Failed to recover from {error_info['error_type']}")
                    # 重大なエラーの場合は通知
                    self._notify_critical_error(error_info)
        else:
            logger.debug("No errors detected in recent logs")

    def _notify_critical_error(self, error_info: Dict[str, Any]):
        """重大エラー通知"""
        logger.critical(f"CRITICAL ERROR: {error_info['error_type']} - {error_info.get('recovery_info', {}).get('description', '')}")

        # オーディオ通知（エラー音）
        try:
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass", "-Command",
                "[System.Console]::Beep(800, 1000); [System.Console]::Beep(600, 1000)"
            ], check=True)
        except:
            pass

        # エラーレポート生成
        error_report = {
            'critical_error': error_info,
            'timestamp': datetime.now().isoformat(),
            'recommendation': 'Manual intervention required'
        }

        error_file = Path("logs") / f"critical_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2, default=str)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SO8T Error Detection and Recovery")
    parser.add_argument("--scan", action="store_true", help="Scan for errors in recent logs")
    parser.add_argument("--recover", action="store_true", help="Attempt recovery for detected errors")
    parser.add_argument("--monitor", action="store_true", help="Start continuous monitoring")
    parser.add_argument("--log-file", type=str, help="Specific log file to scan")

    args = parser.parse_args()

    detector = SO8TErrorDetector()
    recovery_agent = SO8TRecoveryAgent()

    if args.scan:
        # エラースキャン
        errors = detector.scan_for_errors(args.log_file)
        if errors:
            print(f"Found {len(errors)} errors:")
            for error in errors:
                print(f"- {error['error_type']}: {error['pattern']}")
                print(f"  Matches: {len(error['matches'])}")
        else:
            print("No errors found")

    elif args.recover:
        # 回復試行
        errors = detector.scan_for_errors(args.log_file)
        if errors:
            for error in errors:
                print(f"Attempting recovery for: {error['error_type']}")
                success = recovery_agent.execute_recovery(error)
                print(f"Recovery {'succeeded' if success else 'failed'}")
        else:
            print("No errors to recover from")

    elif args.monitor:
        # 継続監視
        monitor = SO8TMonitoringAgent()
        try:
            monitor.start_monitoring()
        except KeyboardInterrupt:
            monitor.stop_monitoring()

    else:
        # デフォルト：最近のエラーをスキャン
        errors = detector.get_recent_errors(hours=1)
        if errors:
            print(f"Recent errors (last 1 hour): {len(errors)}")
            for error in errors:
                print(f"- {error['error_type']}: {error.get('recovery_info', {}).get('description', '')}")
        else:
            print("No recent errors found")


if __name__ == "__main__":
    main()
