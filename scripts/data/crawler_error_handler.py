#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
クローラーエラーハンドリングモジュール

エラータイプ別の分類、ロギング、統計記録、エラーレポート生成を行う。

Usage:
    from scripts.data.crawler_error_handler import CrawlerErrorHandler, ErrorType
    handler = CrawlerErrorHandler()
    handler.handle_error(ErrorType.NETWORK_ERROR, url, exception)
"""

import logging
import traceback
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """エラータイプ"""
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    PARSE_ERROR = "parse_error"
    DOMAIN_CLASSIFICATION_ERROR = "domain_classification_error"
    LABELING_ERROR = "labeling_error"
    ROBOTS_TXT_ERROR = "robots_txt_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ErrorRecord:
    """エラー記録"""
    error_type: str
    url: str
    error_message: str
    timestamp: str
    traceback: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return asdict(self)


class CrawlerErrorHandler:
    """クローラーエラーハンドラー"""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Args:
            log_dir: エラーログ保存ディレクトリ（Noneの場合はログファイルを作成しない）
        """
        self.log_dir = log_dir
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # エラー統計
        self.error_stats: Dict[str, int] = defaultdict(int)
        self.error_records: List[ErrorRecord] = []
        self.max_records = 1000  # 最大記録数
        
        # エラータイプ別の処理方法
        self.error_handlers = {
            ErrorType.NETWORK_ERROR: self._handle_network_error,
            ErrorType.TIMEOUT_ERROR: self._handle_timeout_error,
            ErrorType.PARSE_ERROR: self._handle_parse_error,
            ErrorType.DOMAIN_CLASSIFICATION_ERROR: self._handle_domain_classification_error,
            ErrorType.LABELING_ERROR: self._handle_labeling_error,
            ErrorType.ROBOTS_TXT_ERROR: self._handle_robots_txt_error,
            ErrorType.UNKNOWN_ERROR: self._handle_unknown_error,
        }
    
    def handle_error(
        self,
        error_type: ErrorType,
        url: str,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        log_traceback: bool = True
    ) -> bool:
        """
        エラーを処理
        
        Args:
            error_type: エラータイプ
            url: エラーが発生したURL
            exception: 例外オブジェクト
            context: 追加コンテキスト情報
            log_traceback: トレースバックをログに記録するか
        
        Returns:
            処理を継続できるかどうか（True: 継続可能、False: 中断推奨）
        """
        error_message = str(exception)
        error_type_str = error_type.value
        
        # エラー記録を作成
        error_record = ErrorRecord(
            error_type=error_type_str,
            url=url,
            error_message=error_message,
            timestamp=datetime.now().isoformat(),
            traceback=traceback.format_exc() if log_traceback else None,
            context=context
        )
        
        # 記録を追加
        self.error_records.append(error_record)
        if len(self.error_records) > self.max_records:
            self.error_records.pop(0)  # 古い記録を削除
        
        # 統計を更新
        self.error_stats[error_type_str] += 1
        
        # エラータイプ別の処理
        handler = self.error_handlers.get(error_type, self._handle_unknown_error)
        should_continue = handler(error_record)
        
        # ログ出力
        logger.warning(
            f"[ERROR] {error_type_str} at {url}: {error_message}",
            extra={"error_type": error_type_str, "url": url}
        )
        
        if log_traceback and error_record.traceback:
            logger.debug(f"[TRACEBACK] {error_record.traceback}")
        
        return should_continue
    
    def _handle_network_error(self, error_record: ErrorRecord) -> bool:
        """ネットワークエラーの処理"""
        logger.warning(f"[NETWORK_ERROR] {error_record.url}: {error_record.error_message}")
        return True  # リトライ可能なので継続
    
    def _handle_timeout_error(self, error_record: ErrorRecord) -> bool:
        """タイムアウトエラーの処理"""
        logger.warning(f"[TIMEOUT_ERROR] {error_record.url}: {error_record.error_message}")
        return True  # リトライ可能なので継続
    
    def _handle_parse_error(self, error_record: ErrorRecord) -> bool:
        """HTML解析エラーの処理"""
        logger.error(f"[PARSE_ERROR] {error_record.url}: {error_record.error_message}")
        return True  # 解析エラーはスキップして継続
    
    def _handle_domain_classification_error(self, error_record: ErrorRecord) -> bool:
        """ドメイン分類エラーの処理"""
        logger.warning(f"[DOMAIN_CLASSIFICATION_ERROR] {error_record.url}: {error_record.error_message}")
        return True  # 分類失敗はスキップして継続
    
    def _handle_labeling_error(self, error_record: ErrorRecord) -> bool:
        """ラベル付けエラーの処理"""
        logger.error(f"[LABELING_ERROR] {error_record.url}: {error_record.error_message}")
        return True  # ラベル付けエラーはスキップして継続
    
    def _handle_robots_txt_error(self, error_record: ErrorRecord) -> bool:
        """robots.txtエラーの処理"""
        logger.debug(f"[ROBOTS_TXT_ERROR] {error_record.url}: {error_record.error_message}")
        return False  # robots.txt違反はスキップ
    
    def _handle_unknown_error(self, error_record: ErrorRecord) -> bool:
        """未知のエラーの処理"""
        logger.error(f"[UNKNOWN_ERROR] {error_record.url}: {error_record.error_message}")
        return True  # とりあえず継続
    
    def get_error_stats(self) -> Dict[str, int]:
        """エラー統計を取得"""
        return dict(self.error_stats)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """エラーサマリーを取得"""
        total_errors = sum(self.error_stats.values())
        
        return {
            "total_errors": total_errors,
            "error_types": dict(self.error_stats),
            "error_rate": {
                error_type: (count / total_errors * 100) if total_errors > 0 else 0
                for error_type, count in self.error_stats.items()
            },
            "recent_errors": [
                record.to_dict()
                for record in self.error_records[-10:]  # 最新10件
            ]
        }
    
    def save_error_report(self, output_file: Optional[Path] = None) -> Path:
        """
        エラーレポートを保存
        
        Args:
            output_file: 出力ファイルパス（Noneの場合は自動生成）
        
        Returns:
            保存されたファイルパス
        """
        if output_file is None:
            if self.log_dir:
                output_file = self.log_dir / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            else:
                output_file = Path(f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_error_summary(),
            "all_errors": [record.to_dict() for record in self.error_records]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[REPORT] Error report saved to {output_file}")
        return output_file
    
    def reset_stats(self):
        """統計をリセット"""
        self.error_stats.clear()
        self.error_records.clear()
        logger.info("[RESET] Error statistics reset")


def classify_exception(exception: Exception) -> ErrorType:
    """
    例外をエラータイプに分類
    
    Args:
        exception: 例外オブジェクト
    
    Returns:
        エラータイプ
    """
    exception_type = type(exception).__name__
    exception_message = str(exception).lower()
    
    # タイムアウトエラー
    if "timeout" in exception_message or "Timeout" in exception_type:
        return ErrorType.TIMEOUT_ERROR
    
    # ネットワークエラー
    if any(keyword in exception_message for keyword in ["connection", "network", "dns", "refused"]):
        return ErrorType.NETWORK_ERROR
    
    # 解析エラー
    if any(keyword in exception_type for keyword in ["Parse", "HTML", "BeautifulSoup"]):
        return ErrorType.PARSE_ERROR
    
    # ドメイン分類エラー
    if "domain" in exception_message or "classification" in exception_message:
        return ErrorType.DOMAIN_CLASSIFICATION_ERROR
    
    # ラベル付けエラー
    if "label" in exception_message or "labeling" in exception_message:
        return ErrorType.LABELING_ERROR
    
    # robots.txtエラー
    if "robots" in exception_message:
        return ErrorType.ROBOTS_TXT_ERROR
    
    # 未知のエラー
    return ErrorType.UNKNOWN_ERROR





















