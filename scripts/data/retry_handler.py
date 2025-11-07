#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リトライ機能モジュール

指数バックオフリトライ、最大リトライ回数の設定、リトライ統計の記録を行う。

Usage:
    from scripts.data.retry_handler import RetryHandler
    handler = RetryHandler(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    result = await handler.retry_async(some_async_function, *args, **kwargs)
"""

import asyncio
import logging
from typing import Callable, Any, Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RetryStats:
    """リトライ統計"""
    total_attempts: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    total_delay_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return asdict(self)


class RetryHandler:
    """リトライハンドラー"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 10.0,
        log_retries: bool = True
    ):
        """
        Args:
            max_retries: 最大リトライ回数
            initial_delay: 初期遅延（秒）
            backoff_factor: バックオフ倍率
            max_delay: 最大遅延（秒）
            log_retries: リトライをログに記録するか
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.log_retries = log_retries
        
        # リトライ統計
        self.retry_stats: Dict[str, RetryStats] = defaultdict(RetryStats)
    
    async def retry_async(
        self,
        func: Callable,
        *args,
        operation_name: Optional[str] = None,
        retryable_exceptions: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """
        非同期関数をリトライ実行
        
        Args:
            func: 実行する非同期関数
            *args: 関数の位置引数
            operation_name: 操作名（統計記録用）
            retryable_exceptions: リトライ可能な例外のタプル
            **kwargs: 関数のキーワード引数
        
        Returns:
            関数の戻り値
        
        Raises:
            最後の試行で発生した例外
        """
        if operation_name is None:
            operation_name = func.__name__ if hasattr(func, '__name__') else "unknown"
        
        last_exception = None
        delay = self.initial_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                
                # 成功
                if attempt > 0:
                    # リトライ成功
                    self.retry_stats[operation_name].successful_retries += 1
                    if self.log_retries:
                        logger.info(
                            f"[RETRY_SUCCESS] {operation_name} succeeded after {attempt} retries"
                        )
                
                self.retry_stats[operation_name].total_attempts += attempt + 1
                return result
            
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    # リトライ可能
                    if self.log_retries:
                        logger.warning(
                            f"[RETRY] {operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                    
                    await asyncio.sleep(delay)
                    self.retry_stats[operation_name].total_delay_time += delay
                    
                    # 指数バックオフ
                    delay = min(delay * self.backoff_factor, self.max_delay)
                else:
                    # リトライ上限に達した
                    self.retry_stats[operation_name].failed_retries += 1
                    self.retry_stats[operation_name].total_attempts += attempt + 1
                    
                    if self.log_retries:
                        logger.error(
                            f"[RETRY_FAILED] {operation_name} failed after {self.max_retries + 1} attempts: {str(e)}"
                        )
        
        # すべてのリトライが失敗
        raise last_exception
    
    def retry_sync(
        self,
        func: Callable,
        *args,
        operation_name: Optional[str] = None,
        retryable_exceptions: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """
        同期関数をリトライ実行
        
        Args:
            func: 実行する同期関数
            *args: 関数の位置引数
            operation_name: 操作名（統計記録用）
            retryable_exceptions: リトライ可能な例外のタプル
            **kwargs: 関数のキーワード引数
        
        Returns:
            関数の戻り値
        
        Raises:
            最後の試行で発生した例外
        """
        if operation_name is None:
            operation_name = func.__name__ if hasattr(func, '__name__') else "unknown"
        
        last_exception = None
        delay = self.initial_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                # 成功
                if attempt > 0:
                    # リトライ成功
                    self.retry_stats[operation_name].successful_retries += 1
                    if self.log_retries:
                        logger.info(
                            f"[RETRY_SUCCESS] {operation_name} succeeded after {attempt} retries"
                        )
                
                self.retry_stats[operation_name].total_attempts += attempt + 1
                return result
            
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    # リトライ可能
                    if self.log_retries:
                        logger.warning(
                            f"[RETRY] {operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                    
                    import time
                    time.sleep(delay)
                    self.retry_stats[operation_name].total_delay_time += delay
                    
                    # 指数バックオフ
                    delay = min(delay * self.backoff_factor, self.max_delay)
                else:
                    # リトライ上限に達した
                    self.retry_stats[operation_name].failed_retries += 1
                    self.retry_stats[operation_name].total_attempts += attempt + 1
                    
                    if self.log_retries:
                        logger.error(
                            f"[RETRY_FAILED] {operation_name} failed after {self.max_retries + 1} attempts: {str(e)}"
                        )
        
        # すべてのリトライが失敗
        raise last_exception
    
    def get_retry_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        リトライ統計を取得
        
        Args:
            operation_name: 操作名（Noneの場合は全統計）
        
        Returns:
            リトライ統計
        """
        if operation_name:
            stats = self.retry_stats.get(operation_name)
            if stats:
                return stats.to_dict()
            return {}
        
        return {
            name: stats.to_dict()
            for name, stats in self.retry_stats.items()
        }
    
    def get_retry_summary(self) -> Dict[str, Any]:
        """リトライサマリーを取得"""
        total_operations = len(self.retry_stats)
        total_attempts = sum(stats.total_attempts for stats in self.retry_stats.values())
        total_successful = sum(stats.successful_retries for stats in self.retry_stats.values())
        total_failed = sum(stats.failed_retries for stats in self.retry_stats.values())
        total_delay = sum(stats.total_delay_time for stats in self.retry_stats.values())
        
        return {
            "total_operations": total_operations,
            "total_attempts": total_attempts,
            "total_successful_retries": total_successful,
            "total_failed_retries": total_failed,
            "total_delay_time": total_delay,
            "success_rate": (total_successful / total_attempts * 100) if total_attempts > 0 else 0,
            "operations": {
                name: stats.to_dict()
                for name, stats in self.retry_stats.items()
            }
        }
    
    def save_retry_report(self, output_file: Optional[Path] = None) -> Path:
        """
        リトライレポートを保存
        
        Args:
            output_file: 出力ファイルパス（Noneの場合は自動生成）
        
        Returns:
            保存されたファイルパス
        """
        if output_file is None:
            output_file = Path(f"retry_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "max_retries": self.max_retries,
                "initial_delay": self.initial_delay,
                "backoff_factor": self.backoff_factor,
                "max_delay": self.max_delay,
            },
            "summary": self.get_retry_summary()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[REPORT] Retry report saved to {output_file}")
        return output_file
    
    def reset_stats(self):
        """統計をリセット"""
        self.retry_stats.clear()
        logger.info("[RESET] Retry statistics reset")

