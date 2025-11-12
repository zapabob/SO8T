#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リソースバランス管理スクリプト

GPU/メモリ/CPU使用率を監視し、閾値超過時に自動調整を行います。

Usage:
    python scripts/utils/resource_balancer.py --config configs/complete_automated_ab_pipeline.yaml
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/resource_balancer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# psutilインポート
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. Install with: pip install psutil")


@dataclass
class ResourceMetrics:
    """リソースメトリクス"""
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ResourceBalancer:
    """リソースバランス管理クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.resource_config = config.get('resource_balance', {})
        
        # 閾値設定
        self.gpu_threshold = self.resource_config.get('gpu_threshold', 0.9)
        self.memory_threshold = self.resource_config.get('memory_threshold', 0.9)
        self.cpu_threshold = self.resource_config.get('cpu_threshold', 0.8)
        self.auto_adjust = self.resource_config.get('auto_adjust', True)
        
        # 監視設定
        self.monitor_interval = self.resource_config.get('monitor_interval', 5.0)  # 秒
        self.monitoring = False
        self.monitor_thread = None
        
        # 調整コールバック
        self.adjustment_callbacks: Dict[str, Callable] = {}
        
        # メトリクス履歴
        self.metrics_history: list[ResourceMetrics] = []
        self.max_history = 100
        
        # 出力ディレクトリ
        self.output_dir = Path("logs/resource_balancer")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("Resource Balancer Initialized")
        logger.info("="*80)
        logger.info(f"GPU threshold: {self.gpu_threshold}")
        logger.info(f"Memory threshold: {self.memory_threshold}")
        logger.info(f"CPU threshold: {self.cpu_threshold}")
        logger.info(f"Auto adjust: {self.auto_adjust}")
        logger.info(f"Monitor interval: {self.monitor_interval}s")
    
    def get_gpu_metrics(self) -> tuple[float, float]:
        """
        GPU使用率とメモリ使用率を取得
        
        Returns:
            (gpu_usage, gpu_memory_usage): GPU使用率(0-1), GPUメモリ使用率(0-1)
        """
        try:
            # nvidia-smiを使用
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    # 最初のGPUの情報を取得
                    parts = lines[0].split(', ')
                    if len(parts) >= 3:
                        gpu_usage = float(parts[0]) / 100.0
                        memory_used = float(parts[1])
                        memory_total = float(parts[2])
                        gpu_memory_usage = memory_used / memory_total if memory_total > 0 else 0.0
                        return gpu_usage, gpu_memory_usage
            
            return 0.0, 0.0
            
        except FileNotFoundError:
            logger.warning("nvidia-smi not found. GPU metrics unavailable.")
            return 0.0, 0.0
        except Exception as e:
            logger.warning(f"Failed to get GPU metrics: {e}")
            return 0.0, 0.0
    
    def get_cpu_metrics(self) -> float:
        """
        CPU使用率を取得
        
        Returns:
            cpu_usage: CPU使用率(0-1)
        """
        if not PSUTIL_AVAILABLE:
            return 0.0
        
        try:
            cpu_usage = psutil.cpu_percent(interval=1.0) / 100.0
            return cpu_usage
        except Exception as e:
            logger.warning(f"Failed to get CPU metrics: {e}")
            return 0.0
    
    def get_memory_metrics(self) -> float:
        """
        メモリ使用率を取得
        
        Returns:
            memory_usage: メモリ使用率(0-1)
        """
        if not PSUTIL_AVAILABLE:
            return 0.0
        
        try:
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            return memory_usage
        except Exception as e:
            logger.warning(f"Failed to get memory metrics: {e}")
            return 0.0
    
    def get_current_metrics(self) -> ResourceMetrics:
        """現在のリソースメトリクスを取得"""
        gpu_usage, gpu_memory_usage = self.get_gpu_metrics()
        cpu_usage = self.get_cpu_metrics()
        memory_usage = self.get_memory_metrics()
        
        metrics = ResourceMetrics(
            gpu_usage=gpu_usage,
            gpu_memory_usage=gpu_memory_usage,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage
        )
        
        # 履歴に追加
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        return metrics
    
    def check_thresholds(self, metrics: ResourceMetrics) -> Dict[str, bool]:
        """
        閾値チェック
        
        Args:
            metrics: リソースメトリクス
        
        Returns:
            threshold_violations: 閾値違反の辞書
        """
        violations = {
            'gpu': metrics.gpu_usage > self.gpu_threshold,
            'gpu_memory': metrics.gpu_memory_usage > self.memory_threshold,
            'cpu': metrics.cpu_usage > self.cpu_threshold,
            'memory': metrics.memory_usage > self.memory_threshold
        }
        
        return violations
    
    def adjust_batch_size(self, current_batch_size: int, reduction_factor: float = 0.5) -> int:
        """
        バッチサイズを削減
        
        Args:
            current_batch_size: 現在のバッチサイズ
            reduction_factor: 削減係数
        
        Returns:
            new_batch_size: 新しいバッチサイズ
        """
        new_batch_size = max(1, int(current_batch_size * reduction_factor))
        logger.info(f"Adjusting batch size: {current_batch_size} -> {new_batch_size}")
        return new_batch_size
    
    def enable_cpu_offload(self) -> bool:
        """CPU offloadを有効化"""
        logger.info("Enabling CPU offload")
        return True
    
    def reduce_precision(self) -> bool:
        """精度を削減（fp16 -> int8等）"""
        logger.info("Reducing precision")
        return True
    
    def adjust_resources(self, violations: Dict[str, bool], metrics: ResourceMetrics) -> Dict[str, Any]:
        """
        リソース調整を実行
        
        Args:
            violations: 閾値違反の辞書
            metrics: リソースメトリクス
        
        Returns:
            adjustments: 実行した調整の辞書
        """
        if not self.auto_adjust:
            return {}
        
        adjustments = {}
        
        # GPU使用率が高い場合
        if violations.get('gpu', False):
            logger.warning(f"GPU usage exceeds threshold: {metrics.gpu_usage:.2%} > {self.gpu_threshold:.2%}")
            
            # バッチサイズ削減
            if 'batch_size' in self.adjustment_callbacks:
                current_batch = self.adjustment_callbacks['batch_size']()
                new_batch = self.adjust_batch_size(current_batch)
                adjustments['batch_size'] = new_batch
                if 'set_batch_size' in self.adjustment_callbacks:
                    self.adjustment_callbacks['set_batch_size'](new_batch)
        
        # GPUメモリ使用率が高い場合
        if violations.get('gpu_memory', False):
            logger.warning(f"GPU memory usage exceeds threshold: {metrics.gpu_memory_usage:.2%} > {self.memory_threshold:.2%}")
            
            # CPU offloadを有効化
            if 'enable_cpu_offload' in self.adjustment_callbacks:
                self.adjustment_callbacks['enable_cpu_offload']()
                adjustments['cpu_offload'] = True
            
            # 精度削減
            if 'reduce_precision' in self.adjustment_callbacks:
                self.adjustment_callbacks['reduce_precision']()
                adjustments['precision_reduced'] = True
        
        # CPU使用率が高い場合
        if violations.get('cpu', False):
            logger.warning(f"CPU usage exceeds threshold: {metrics.cpu_usage:.2%} > {self.cpu_threshold:.2%}")
            
            # バッチサイズ削減
            if 'batch_size' in self.adjustment_callbacks:
                current_batch = self.adjustment_callbacks['batch_size']()
                new_batch = self.adjust_batch_size(current_batch, reduction_factor=0.75)
                adjustments['batch_size'] = new_batch
                if 'set_batch_size' in self.adjustment_callbacks:
                    self.adjustment_callbacks['set_batch_size'](new_batch)
        
        # メモリ使用率が高い場合
        if violations.get('memory', False):
            logger.warning(f"Memory usage exceeds threshold: {metrics.memory_usage:.2%} > {self.memory_threshold:.2%}")
            
            # CPU offloadを有効化
            if 'enable_cpu_offload' in self.adjustment_callbacks:
                self.adjustment_callbacks['enable_cpu_offload']()
                adjustments['cpu_offload'] = True
        
        return adjustments
    
    def register_adjustment_callback(self, name: str, callback: Callable):
        """
        調整コールバックを登録
        
        Args:
            name: コールバック名
            callback: コールバック関数
        """
        self.adjustment_callbacks[name] = callback
        logger.info(f"Registered adjustment callback: {name}")
    
    def monitor_loop(self):
        """監視ループ"""
        logger.info("Starting resource monitoring loop...")
        
        while self.monitoring:
            try:
                # メトリクス取得
                metrics = self.get_current_metrics()
                
                # 閾値チェック
                violations = self.check_thresholds(metrics)
                
                # ログ出力
                logger.info(
                    f"Resources - GPU: {metrics.gpu_usage:.2%}, "
                    f"GPU Mem: {metrics.gpu_memory_usage:.2%}, "
                    f"CPU: {metrics.cpu_usage:.2%}, "
                    f"Mem: {metrics.memory_usage:.2%}"
                )
                
                # 閾値違反がある場合
                if any(violations.values()):
                    logger.warning(f"Threshold violations detected: {violations}")
                    
                    # 自動調整
                    adjustments = self.adjust_resources(violations, metrics)
                    
                    if adjustments:
                        logger.info(f"Applied adjustments: {adjustments}")
                
                # 待機
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                logger.exception(e)
                time.sleep(self.monitor_interval)
    
    def start_monitoring(self):
        """監視を開始"""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """監視を停止"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def save_metrics_history(self, filepath: Optional[Path] = None):
        """メトリクス履歴を保存"""
        if filepath is None:
            filepath = self.output_dir / f"metrics_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        history_data = [
            {
                'gpu_usage': m.gpu_usage,
                'gpu_memory_usage': m.gpu_memory_usage,
                'cpu_usage': m.cpu_usage,
                'memory_usage': m.memory_usage,
                'timestamp': m.timestamp
            }
            for m in self.metrics_history
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metrics history saved to {filepath}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """メトリクスサマリーを取得"""
        if not self.metrics_history:
            return {}
        
        gpu_usages = [m.gpu_usage for m in self.metrics_history]
        gpu_memory_usages = [m.gpu_memory_usage for m in self.metrics_history]
        cpu_usages = [m.cpu_usage for m in self.metrics_history]
        memory_usages = [m.memory_usage for m in self.metrics_history]
        
        return {
            'gpu': {
                'mean': sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0.0,
                'max': max(gpu_usages) if gpu_usages else 0.0,
                'min': min(gpu_usages) if gpu_usages else 0.0
            },
            'gpu_memory': {
                'mean': sum(gpu_memory_usages) / len(gpu_memory_usages) if gpu_memory_usages else 0.0,
                'max': max(gpu_memory_usages) if gpu_memory_usages else 0.0,
                'min': min(gpu_memory_usages) if gpu_memory_usages else 0.0
            },
            'cpu': {
                'mean': sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0.0,
                'max': max(cpu_usages) if cpu_usages else 0.0,
                'min': min(cpu_usages) if cpu_usages else 0.0
            },
            'memory': {
                'mean': sum(memory_usages) / len(memory_usages) if memory_usages else 0.0,
                'max': max(memory_usages) if memory_usages else 0.0,
                'min': min(memory_usages) if memory_usages else 0.0
            },
            'sample_count': len(self.metrics_history)
        }


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Resource Balancer for SO8T Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/complete_automated_ab_pipeline.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Monitoring duration in seconds (0 = infinite)"
    )
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # リソースバランサー初期化
    balancer = ResourceBalancer(config)
    
    try:
        # 監視開始
        balancer.start_monitoring()
        
        # 指定時間監視
        if args.duration > 0:
            time.sleep(args.duration)
            balancer.stop_monitoring()
        else:
            # 無限ループ
            logger.info("Monitoring indefinitely. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping monitoring...")
                balancer.stop_monitoring()
        
        # メトリクス履歴保存
        balancer.save_metrics_history()
        
        # サマリー表示
        summary = balancer.get_metrics_summary()
        logger.info("="*80)
        logger.info("Resource Metrics Summary")
        logger.info("="*80)
        logger.info(json.dumps(summary, indent=2, ensure_ascii=False))
        
        return 0
        
    except Exception as e:
        logger.error(f"Resource balancer failed: {e}")
        logger.exception(e)
        balancer.stop_monitoring()
        return 1


if __name__ == "__main__":
    sys.exit(main())

