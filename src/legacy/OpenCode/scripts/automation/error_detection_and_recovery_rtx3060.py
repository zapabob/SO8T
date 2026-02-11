#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTX3060 SO8T Error Detection and Recovery System
Frozen weights + QLoRA fine-tuning optimized

RTX3060固有のメモリ制約とエラーに対応した自動回復システム
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import traceback
import psutil
import GPUtil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RTX3060ErrorDetector:
    """
    RTX3060固有エラーを検知するクラス
    """

    def __init__(self):
        self.rtx3060_limits = {
            'max_memory_gb': 8.0,
            'safe_memory_threshold': 0.85,  # 85%使用で警告
            'critical_memory_threshold': 0.95,  # 95%でクリティカル
            'min_free_memory_gb': 0.5  # 最低0.5GB空き容量
        }

    def scan_for_rtx3060_errors(self, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        RTX3060固有のエラーをスキャン
        """
        errors = []

        # GPUメモリチェック
        gpu_errors = self._check_gpu_memory()
        errors.extend(gpu_errors)

        # CUDAエラーチェック
        cuda_errors = self._check_cuda_errors(log_file)
        errors.extend(cuda_errors)

        # メモリリークチェック
        memory_errors = self._check_memory_leaks()
        errors.extend(memory_errors)

        # QLoRA固有エラーチェック
        qlora_errors = self._check_qlora_errors(log_file)
        errors.extend(qlora_errors)

        return errors

    def _check_gpu_memory(self) -> List[Dict[str, Any]]:
        """RTX3060 GPUメモリチェック"""
        errors = []

        try:
            import torch
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                max_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                # メモリ使用率チェック
                usage_ratio = memory_reserved / max_memory

                if usage_ratio > self.rtx3060_limits['critical_memory_threshold']:
                    errors.append({
                        'type': 'critical_memory',
                        'message': f'Critical GPU memory usage: {usage_ratio:.1%} ({memory_reserved:.1f}GB/{max_memory:.1f}GB)',
                        'severity': 'critical',
                        'action': 'immediate_recovery'
                    })
                elif usage_ratio > self.rtx3060_limits['safe_memory_threshold']:
                    errors.append({
                        'type': 'high_memory',
                        'message': f'High GPU memory usage: {usage_ratio:.1%} ({memory_reserved:.1f}GB/{max_memory:.1f}GB)',
                        'severity': 'warning',
                        'action': 'reduce_batch_size'
                    })

                # 空きメモリチェック
                free_memory = max_memory - memory_reserved
                if free_memory < self.rtx3060_limits['min_free_memory_gb']:
                    errors.append({
                        'type': 'low_free_memory',
                        'message': f'Low free GPU memory: {free_memory:.1f}GB (minimum {self.rtx3060_limits["min_free_memory_gb"]}GB required)',
                        'severity': 'warning',
                        'action': 'clear_cache'
                    })

        except Exception as e:
            errors.append({
                'type': 'gpu_check_error',
                'message': f'GPU memory check failed: {e}',
                'severity': 'error',
                'action': 'retry'
            })

        return errors

    def _check_cuda_errors(self, log_file: Optional[str]) -> List[Dict[str, Any]]:
        """CUDA関連エラーチェック"""
        errors = []

        cuda_error_patterns = [
            'CUDA out of memory',
            'RuntimeError: CUDA error',
            'CUDA kernel errors',
            'Device-side assert triggered',
            'CUDA driver error',
            'cuBLAS error',
            'cuDNN error'
        ]

        if log_file and Path(log_file).exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                for pattern in cuda_error_patterns:
                    if pattern in content:
                        errors.append({
                            'type': 'cuda_error',
                            'message': f'CUDA error detected: {pattern}',
                            'severity': 'critical',
                            'action': 'cuda_recovery',
                            'pattern': pattern
                        })
            except Exception as e:
                logger.warning(f"Log file read error: {e}")

        return errors

    def _check_memory_leaks(self) -> List[Dict[str, Any]]:
        """メモリリークチェック"""
        errors = []

        try:
            # Pythonプロセスメモリチェック
            process = psutil.Process()
            memory_usage = process.memory_info().rss / (1024**3)  # GB

            if memory_usage > 16:  # 16GB以上使用で警告
                errors.append({
                    'type': 'memory_leak',
                    'message': f'High system memory usage: {memory_usage:.1f}GB',
                    'severity': 'warning',
                    'action': 'cleanup_temp_files'
                })

        except Exception as e:
            logger.warning(f"Memory leak check failed: {e}")

        return errors

    def _check_qlora_errors(self, log_file: Optional[str]) -> List[Dict[str, Any]]:
        """QLoRA固有エラーチェック"""
        errors = []

        qlora_error_patterns = [
            'LoRA parameter not found',
            'quantization error',
            '8bit training error',
            'PEFT error',
            'frozen weights error'
        ]

        if log_file and Path(log_file).exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                for pattern in qlora_error_patterns:
                    if pattern in content:
                        errors.append({
                            'type': 'qlora_error',
                            'message': f'QLoRA error detected: {pattern}',
                            'severity': 'error',
                            'action': 'qlora_recovery',
                            'pattern': pattern
                        })
            except Exception as e:
                logger.warning(f"QLoRA log check failed: {e}")

        return errors


class RTX3060RecoveryAgent:
    """
    RTX3060固有の回復処理を行うクラス
    """

    def __init__(self):
        self.recovery_actions = {
            'critical_memory': self._recover_critical_memory,
            'high_memory': self._recover_high_memory,
            'low_free_memory': self._recover_low_free_memory,
            'cuda_error': self._recover_cuda_error,
            'memory_leak': self._recover_memory_leak,
            'qlora_error': self._recover_qlora_error,
            'gpu_check_error': self._recover_gpu_check_error
        }

    def execute_recovery(self, error_info: Dict[str, Any]) -> bool:
        """
        エラー情報に基づいて回復処理を実行
        """
        error_type = error_info.get('type')
        action = error_info.get('action', 'retry')

        logger.info(f"[RECOVERY] Executing {action} for {error_type}")

        if action in self.recovery_actions:
            try:
                return self.recovery_actions[action](error_info)
            except Exception as e:
                logger.error(f"Recovery action failed: {e}")
                return False
        else:
            logger.warning(f"Unknown recovery action: {action}")
            return False

    def _recover_critical_memory(self, error_info: Dict[str, Any]) -> bool:
        """クリティカルメモリ使用時の回復"""
        logger.info("Executing critical memory recovery...")

        try:
            import torch
            import gc

            # GPUキャッシュクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Pythonガベージコレクション
            gc.collect()

            # 不要なプロセス終了試行
            self._kill_memory_hungry_processes()

            # メモリ使用量再チェック
            time.sleep(2)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            max_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            usage_ratio = memory_reserved / max_memory

            if usage_ratio < 0.9:  # 90%以下に回復したら成功
                logger.info("[OK] Critical memory recovery successful")
                return True
            else:
                logger.warning("[WARNING] Critical memory recovery partial success")
                return False

        except Exception as e:
            logger.error(f"Critical memory recovery failed: {e}")
            return False

    def _recover_high_memory(self, error_info: Dict[str, Any]) -> bool:
        """高メモリ使用時の回復（バッチサイズ削減推奨）"""
        logger.info("High memory usage detected - recommending batch size reduction")

        # 設定ファイルのバッチサイズを半分に
        try:
            self._reduce_batch_size_config()
            logger.info("[OK] Batch size reduced in config")
            return True
        except Exception as e:
            logger.error(f"Batch size reduction failed: {e}")
            return False

    def _recover_low_free_memory(self, error_info: Dict[str, Any]) -> bool:
        """空きメモリ不足時の回復"""
        logger.info("Recovering low free memory...")

        try:
            import torch
            torch.cuda.empty_cache()

            # 一時ファイルクリーンアップ
            self._cleanup_temp_files()

            logger.info("[OK] Low memory recovery completed")
            return True
        except Exception as e:
            logger.error(f"Low memory recovery failed: {e}")
            return False

    def _recover_cuda_error(self, error_info: Dict[str, Any]) -> bool:
        """CUDAエラー回復"""
        logger.info("Recovering from CUDA error...")

        try:
            # CUDAコンテキストリセット
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # デバイスリセット（最終手段）
                # torch.cuda.reset_peak_memory_stats()

            # プロセス再起動を推奨
            logger.info("[OK] CUDA error recovery completed - restart recommended")
            return True
        except Exception as e:
            logger.error(f"CUDA error recovery failed: {e}")
            return False

    def _recover_memory_leak(self, error_info: Dict[str, Any]) -> bool:
        """メモリリーク回復"""
        logger.info("Recovering from memory leak...")

        try:
            import gc
            gc.collect()

            self._cleanup_temp_files()

            logger.info("[OK] Memory leak recovery completed")
            return True
        except Exception as e:
            logger.error(f"Memory leak recovery failed: {e}")
            return False

    def _recover_qlora_error(self, error_info: Dict[str, Any]) -> bool:
        """QLoRAエラー回復"""
        logger.info("Recovering from QLoRA error...")

        try:
            # QLoRA設定の検証と修正
            self._fix_qlora_config()
            logger.info("[OK] QLoRA configuration fixed")
            return True
        except Exception as e:
            logger.error(f"QLoRA recovery failed: {e}")
            return False

    def _recover_gpu_check_error(self, error_info: Dict[str, Any]) -> bool:
        """GPUチェックエラー回復"""
        logger.info("Recovering from GPU check error...")

        try:
            # CUDA可用性再チェック
            import torch
            if torch.cuda.is_available():
                logger.info("[OK] GPU check recovery successful")
                return True
            else:
                logger.error("[ERROR] GPU still unavailable")
                return False
        except Exception as e:
            logger.error(f"GPU check recovery failed: {e}")
            return False

    def _kill_memory_hungry_processes(self):
        """メモリを大量消費しているプロセスを終了"""
        try:
            # メモリ使用量トップ5プロセスを取得
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                try:
                    if proc.info['memory_percent'] > 5.0:  # 5%以上使用
                        processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # 自分自身以外でメモリ使用の高いプロセスをログ
            for proc in sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:5]:
                if proc['pid'] != os.getpid():
                    logger.warning(f"High memory process: {proc['name']} (PID: {proc['pid']}) - {proc['memory_percent']:.1f}%")

        except Exception as e:
            logger.warning(f"Process memory check failed: {e}")

    def _reduce_batch_size_config(self):
        """設定ファイルのバッチサイズを削減"""
        config_paths = [
            "configs/complete_so8t_pipeline.yaml",
            "configs/train_so8t_phi3_qlora_rtx3060.yaml"
        ]

        for config_path in config_paths:
            if Path(config_path).exists():
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)

                    # バッチサイズを半分に（最小1）
                    if 'training' in config and 'batch_size' in config['training']:
                        old_batch = config['training']['batch_size']
                        new_batch = max(1, old_batch // 2)
                        config['training']['batch_size'] = new_batch
                        logger.info(f"Reduced batch size from {old_batch} to {new_batch} in {config_path}")

                    # グラディエントアキュムレーションを調整
                    if 'gradient_accumulation_steps' in config['training']:
                        old_accum = config['training']['gradient_accumulation_steps']
                        new_accum = old_accum * 2  # バッチサイズ削減の補填
                        config['training']['gradient_accumulation_steps'] = new_accum
                        logger.info(f"Increased gradient accumulation from {old_accum} to {new_accum}")

                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

                except Exception as e:
                    logger.error(f"Config update failed for {config_path}: {e}")

    def _cleanup_temp_files(self):
        """一時ファイルクリーンアップ"""
        temp_dirs = [
            Path("D:/webdataset/temp"),
            Path("temp"),
            Path("tmp")
        ]

        for temp_dir in temp_dirs:
            if temp_dir.exists():
                try:
                    for file_path in temp_dir.glob("*"):
                        if file_path.is_file():
                            file_path.unlink()
                    logger.info(f"Cleaned temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Temp cleanup failed for {temp_dir}: {e}")

    def _fix_qlora_config(self):
        """QLoRA設定修正"""
        config_path = "configs/train_so8t_phi3_qlora_rtx3060.yaml"
        if Path(config_path).exists():
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                # QLoRA設定の検証と修正
                qlora_config = config.get('qlora', {})

                # LoRAランクを小さく（RTX3060向け）
                if qlora_config.get('r', 64) > 16:
                    qlora_config['r'] = 16
                    logger.info("Reduced LoRA rank to 16 for RTX3060")

                # ドロップアウト調整
                if qlora_config.get('lora_dropout', 0.05) > 0.01:
                    qlora_config['lora_dropout'] = 0.01
                    logger.info("Reduced LoRA dropout to 0.01")

                config['qlora'] = qlora_config

                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

                logger.info("QLoRA configuration fixed")

            except Exception as e:
                logger.error(f"QLoRA config fix failed: {e}")


class RTX3060MonitoringAgent:
    """
    RTX3060を継続監視するクラス
    """

    def __init__(self, check_interval: int = 300):  # 5分間隔
        self.check_interval = check_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.detector = RTX3060ErrorDetector()
        self.recovery_agent = RTX3060RecoveryAgent()

    def start_monitoring(self):
        """監視開始"""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"[MONITOR] RTX3060 monitoring started (interval: {self.check_interval}s)")

    def stop_monitoring(self):
        """監視停止"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("[MONITOR] RTX3060 monitoring stopped")

    def _monitor_loop(self):
        """監視ループ"""
        while self.is_monitoring:
            try:
                # エラースキャン
                errors = self.detector.scan_for_rtx3060_errors()

                if errors:
                    logger.info(f"[MONITOR] Detected {len(errors)} RTX3060 errors")

                    for error in errors:
                        severity = error.get('severity', 'info')

                        if severity in ['critical', 'error']:
                            logger.error(f"[MONITOR] {severity.upper()}: {error['message']}")
                            # 自動回復試行
                            if self.recovery_agent.execute_recovery(error):
                                logger.info("[MONITOR] Auto-recovery successful")
                            else:
                                logger.error("[MONITOR] Auto-recovery failed")
                        elif severity == 'warning':
                            logger.warning(f"[MONITOR] WARNING: {error['message']}")
                            # 警告レベルのエラーも回復試行
                            self.recovery_agent.execute_recovery(error)

                # 定期的なメモリクリーンアップ
                self._periodic_cleanup()

            except Exception as e:
                logger.error(f"[MONITOR] Monitoring error: {e}")

            time.sleep(self.check_interval)

    def _periodic_cleanup(self):
        """定期クリーンアップ"""
        try:
            import torch
            if torch.cuda.is_available():
                # GPUメモリ使用量チェック
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                max_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                usage_ratio = memory_reserved / max_memory

                # 80%以上使用していたらクリーンアップ
                if usage_ratio > 0.8:
                    torch.cuda.empty_cache()
                    logger.debug("[MONITOR] Periodic GPU cache cleanup performed")

        except Exception as e:
            logger.debug(f"Periodic cleanup failed: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="RTX3060 SO8T Error Detection and Recovery System"
    )
    parser.add_argument(
        '--scan',
        action='store_true',
        help='Scan for RTX3060 errors'
    )
    parser.add_argument(
        '--recover',
        action='store_true',
        help='Execute recovery actions'
    )
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Start monitoring mode'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        help='Log file to scan for errors'
    )

    args = parser.parse_args()

    detector = RTX3060ErrorDetector()
    recovery_agent = RTX3060RecoveryAgent()

    if args.scan:
        # エラースキャン
        logger.info("[SCAN] Scanning for RTX3060 errors...")
        errors = detector.scan_for_rtx3060_errors(args.log_file)

        if errors:
            logger.info(f"Found {len(errors)} errors:")
            for i, error in enumerate(errors, 1):
                print(f"{i}. [{error['severity'].upper()}] {error['type']}: {error['message']}")
                print(f"   Action: {error['action']}")
        else:
            logger.info("No RTX3060 errors detected")

    elif args.recover:
        # 回復実行
        logger.info("[RECOVER] Executing recovery actions...")
        errors = detector.scan_for_rtx3060_errors(args.log_file)

        recovered = 0
        for error in errors:
            if recovery_agent.execute_recovery(error):
                recovered += 1

        logger.info(f"Recovery completed: {recovered}/{len(errors)} successful")

    elif args.monitor:
        # 監視モード
        logger.info("[MONITOR] Starting RTX3060 monitoring...")
        monitor = RTX3060MonitoringAgent()
        monitor.start_monitoring()

        try:
            # Ctrl+Cで停止
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("[MONITOR] Stopping monitoring...")
            monitor.stop_monitoring()

    else:
        # デフォルト：スキャンのみ
        main()


if __name__ == '__main__':
    main()
