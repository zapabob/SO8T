#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
クローラー健康状態監視モジュール

クローラーの健康状態を監視し、問題を検出・報告します。

Usage:
    from scripts.data.crawler_health_monitor import CrawlerHealthMonitor
    monitor = CrawlerHealthMonitor()
    health_status = monitor.check_health()
"""

import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """健康状態"""
    status: str  # "healthy", "warning", "critical"
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    error_rate: float
    timestamp: str
    issues: List[str]
    
    def to_dict(self):
        return asdict(self)


class CrawlerHealthMonitor:
    """クローラー健康状態監視クラス"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 設定辞書
        """
        self.config = config or {
            "cpu_threshold_warning": 0.8,
            "cpu_threshold_critical": 0.95,
            "memory_threshold_warning": 0.85,
            "memory_threshold_critical": 0.95,
            "disk_threshold_warning": 0.9,
            "disk_threshold_critical": 0.95,
            "error_rate_threshold_warning": 0.1,
            "error_rate_threshold_critical": 0.2,
        }
        
        self.health_history: List[HealthStatus] = []
        self.max_history = 100
    
    def check_health(
        self,
        error_count: int = 0,
        total_requests: int = 0
    ) -> HealthStatus:
        """
        健康状態をチェック
        
        Args:
            error_count: エラー数
            total_requests: 総リクエスト数
        
        Returns:
            health_status: 健康状態
        """
        issues = []
        status = "healthy"
        
        # CPU使用率
        cpu_usage = 0.0
        if PSUTIL_AVAILABLE:
            cpu_usage = psutil.cpu_percent(interval=1.0) / 100.0
            if cpu_usage > self.config["cpu_threshold_critical"]:
                status = "critical"
                issues.append(f"CPU usage critical: {cpu_usage:.1%}")
            elif cpu_usage > self.config["cpu_threshold_warning"]:
                if status == "healthy":
                    status = "warning"
                issues.append(f"CPU usage high: {cpu_usage:.1%}")
        
        # メモリ使用率
        memory_usage = 0.0
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            if memory_usage > self.config["memory_threshold_critical"]:
                status = "critical"
                issues.append(f"Memory usage critical: {memory_usage:.1%}")
            elif memory_usage > self.config["memory_threshold_warning"]:
                if status == "healthy":
                    status = "warning"
                issues.append(f"Memory usage high: {memory_usage:.1%}")
        
        # ディスク使用率
        disk_usage = 0.0
        if PSUTIL_AVAILABLE:
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent / 100.0
            if disk_usage > self.config["disk_threshold_critical"]:
                status = "critical"
                issues.append(f"Disk usage critical: {disk_usage:.1%}")
            elif disk_usage > self.config["disk_threshold_warning"]:
                if status == "healthy":
                    status = "warning"
                issues.append(f"Disk usage high: {disk_usage:.1%}")
        
        # エラー率
        error_rate = 0.0
        if total_requests > 0:
            error_rate = error_count / total_requests
            if error_rate > self.config["error_rate_threshold_critical"]:
                status = "critical"
                issues.append(f"Error rate critical: {error_rate:.1%}")
            elif error_rate > self.config["error_rate_threshold_warning"]:
                if status == "healthy":
                    status = "warning"
                issues.append(f"Error rate high: {error_rate:.1%}")
        
        health_status = HealthStatus(
            status=status,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            error_rate=error_rate,
            timestamp=datetime.now().isoformat(),
            issues=issues
        )
        
        # 履歴に追加
        self.health_history.append(health_status)
        if len(self.health_history) > self.max_history:
            self.health_history.pop(0)
        
        return health_status
    
    def get_health_summary(self) -> Dict:
        """健康状態サマリーを取得"""
        if not self.health_history:
            return {"status": "unknown", "message": "No health data available"}
        
        latest = self.health_history[-1]
        
        return {
            "status": latest.status,
            "cpu_usage": latest.cpu_usage,
            "memory_usage": latest.memory_usage,
            "disk_usage": latest.disk_usage,
            "error_rate": latest.error_rate,
            "issues": latest.issues,
            "timestamp": latest.timestamp
        }


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crawler health monitoring")
    parser.add_argument("--error-count", type=int, default=0,
                        help="Error count")
    parser.add_argument("--total-requests", type=int, default=0,
                        help="Total request count")
    args = parser.parse_args()
    
    monitor = CrawlerHealthMonitor()
    health = monitor.check_health(args.error_count, args.total_requests)
    
    print(f"[HEALTH] Status: {health.status}")
    print(f"[HEALTH] CPU: {health.cpu_usage:.1%}")
    print(f"[HEALTH] Memory: {health.memory_usage:.1%}")
    print(f"[HEALTH] Disk: {health.disk_usage:.1%}")
    print(f"[HEALTH] Error rate: {health.error_rate:.1%}")
    
    if health.issues:
        print("[HEALTH] Issues:")
        for issue in health.issues:
            print(f"  - {issue}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()







