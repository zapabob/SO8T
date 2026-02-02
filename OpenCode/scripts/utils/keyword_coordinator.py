#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
キーワード共有メカニズム

キーワードキュー管理と共有メモリ（JSONファイル）によるキーワード状態管理を実装。
ブラウザ間でキーワードの重複を回避するための協調機能を提供します。

Usage:
    from scripts.utils.keyword_coordinator import KeywordCoordinator
    
    coordinator = KeywordCoordinator(keyword_queue_file="D:/webdataset/checkpoints/keyword_queue.json")
    coordinator.add_keywords(["Python", "Rust", "TypeScript"])
    keyword = coordinator.get_next_keyword(browser_id=0)
    coordinator.mark_keyword_completed(keyword, browser_id=0)
"""

import sys
import json
import logging
import asyncio
import fcntl
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class KeywordStatus(Enum):
    """キーワード状態"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class KeywordPriority(Enum):
    """キーワード優先度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    
    @classmethod
    def get_priority_value(cls, priority: str) -> int:
        """優先度を数値に変換"""
        priority_map = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'urgent': 4
        }
        return priority_map.get(priority.lower(), 2)  # デフォルト: medium


class KeywordCoordinator:
    """キーワード共有メカニズム"""
    
    def __init__(
        self,
        keyword_queue_file: str = "D:/webdataset/checkpoints/keyword_queue.json",
        assignment_timeout: int = 3600,  # 1時間
        lock_timeout: float = 5.0,  # 5秒
        cache_file: Optional[str] = None,
        cache_expiration_hours: int = 24
    ):
        """
        初期化
        
        Args:
            keyword_queue_file: キーワードキューファイルのパス
            assignment_timeout: キーワード割り当てタイムアウト（秒）
            lock_timeout: ロック取得タイムアウト（秒）
            cache_file: キャッシュファイルのパス（Noneの場合は自動生成）
            cache_expiration_hours: キャッシュ有効期限（時間）
        """
        self.keyword_queue_file = Path(keyword_queue_file)
        self.assignment_timeout = assignment_timeout
        self.lock_timeout = lock_timeout
        
        # キューファイルの親ディレクトリを作成
        self.keyword_queue_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 初期化時にキューを読み込み
        self._load_queue()
        
        # キャッシュ設定
        if cache_file is None:
            cache_file = str(self.keyword_queue_file.parent / "keyword_classification_cache.json")
        self.cache_file = Path(cache_file)
        self.cache_expiration_hours = cache_expiration_hours
        self.cache: Dict[str, Any] = {}
        self._load_cache()
        
        logger.info("="*80)
        logger.info("Keyword Coordinator Initialized")
        logger.info("="*80)
        logger.info(f"Queue file: {self.keyword_queue_file}")
        logger.info(f"Cache file: {self.cache_file}")
        logger.info(f"Assignment timeout: {self.assignment_timeout}s")
        logger.info(f"Cache expiration: {self.cache_expiration_hours}h")
        logger.info(f"Total keywords: {len(self.queue.get('keywords', {}))}")
        logger.info(f"Cached classifications: {len(self.cache.get('classifications', {}))}")
    
    def _load_queue(self) -> Dict[str, Any]:
        """
        キューを読み込み
        
        Returns:
            queue: キュー辞書
        """
        if self.keyword_queue_file.exists():
            try:
                with open(self.keyword_queue_file, 'r', encoding='utf-8') as f:
                    self.queue = json.load(f)
            except Exception as e:
                logger.warning(f"[QUEUE] Failed to load queue: {e}, initializing new queue")
                self.queue = {
                    'keywords': {},
                    'browser_assignments': {},
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'last_updated': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                }
        else:
            self.queue = {
                'keywords': {},
                'browser_assignments': {},
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
        
        return self.queue
    
    def _save_queue(self) -> bool:
        """
        キューを保存
        
        Returns:
            success: 成功フラグ
        """
        try:
            # ロックを取得して保存
            with open(self.keyword_queue_file, 'w', encoding='utf-8') as f:
                # Windowsではファイルロックが異なるため、try-exceptで処理
                try:
                    if sys.platform != 'win32':
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass  # Windowsではスキップ
                
                self.queue['metadata']['last_updated'] = datetime.now().isoformat()
                json.dump(self.queue, f, ensure_ascii=False, indent=2)
                
                try:
                    if sys.platform != 'win32':
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass  # Windowsではスキップ
            
            return True
        except Exception as e:
            logger.error(f"[QUEUE] Failed to save queue: {e}")
            return False
    
    def add_keywords(
        self, 
        keywords: List[str], 
        source: str = "manual",
        priority: str = "medium"
    ) -> int:
        """
        キーワードを追加
        
        Args:
            keywords: 追加するキーワードのリスト
            source: キーワードのソース（manual, streamlit, api等）
            priority: 優先度（low, medium, high, urgent）
        
        Returns:
            added_count: 追加されたキーワード数
        """
        # 優先度の検証
        priority_lower = priority.lower()
        if priority_lower not in ['low', 'medium', 'high', 'urgent']:
            logger.warning(f"[QUEUE] Invalid priority '{priority}', using 'medium'")
            priority_lower = 'medium'
        
        added_count = 0
        
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if not keyword_lower:
                continue
            
            # 既に存在するキーワードはスキップ
            if keyword_lower in self.queue['keywords']:
                logger.debug(f"[QUEUE] Keyword already exists: {keyword}")
                continue
            
            # 新しいキーワードを追加
            self.queue['keywords'][keyword_lower] = {
                'keyword': keyword,
                'status': KeywordStatus.PENDING.value,
                'priority': priority_lower,
                'priority_value': KeywordPriority.get_priority_value(priority_lower),
                'browser_id': None,
                'assigned_at': None,
                'completed_at': None,
                'failed_at': None,
                'source': source,
                'added_at': datetime.now().isoformat(),
                'retry_count': 0,
                'max_retries': 3,
                # 進捗追跡情報
                'progress': {
                    'started_at': None,
                    'finished_at': None,
                    'samples_collected': 0,
                    'urls_processed': 0,
                    'urls_failed': 0,
                    'processing_times': [],  # 各URLの処理時間（秒）
                    'success_rate': 0.0,
                    'by_browser': {}  # ブラウザ別の処理状況
                }
            }
            added_count += 1
            logger.info(f"[QUEUE] Added keyword: {keyword} (source: {source}, priority: {priority_lower})")
        
        # キューを保存
        self._save_queue()
        
        logger.info(f"[QUEUE] Added {added_count} keywords (total: {len(self.queue['keywords'])})")
        return added_count
    
    def get_next_keyword(self, browser_id: int, priority_filter: Optional[str] = None) -> Optional[str]:
        """
        次のキーワードを取得（重複回避、優先度順）
        
        Args:
            browser_id: ブラウザID
            priority_filter: 優先度フィルタ（low, medium, high, urgent、Noneの場合はすべて）
        
        Returns:
            keyword: 割り当てられたキーワード（Noneの場合は利用可能なキーワードなし）
        """
        # タイムアウトした割り当てをリセット
        self._reset_timed_out_assignments()
        
        # pending状態のキーワードを取得
        pending_keywords = []
        for keyword_lower, keyword_data in self.queue['keywords'].items():
            status = keyword_data.get('status', KeywordStatus.PENDING.value)
            priority = keyword_data.get('priority', 'medium')
            
            # pending状態のキーワードをフィルタ
            if status == KeywordStatus.PENDING.value:
                # 優先度フィルタを適用
                if priority_filter and priority.lower() != priority_filter.lower():
                    continue
                
                # 他のブラウザに割り当てられていないことを確認
                assigned_browser = keyword_data.get('browser_id')
                if assigned_browser is None or assigned_browser == browser_id:
                    priority_value = keyword_data.get('priority_value', KeywordPriority.get_priority_value(priority))
                    pending_keywords.append((priority_value, keyword_lower, keyword_data))
        
        # 優先度順にソート（urgent > high > medium > low）
        pending_keywords.sort(key=lambda x: x[0], reverse=True)
        
        # 最も優先度の高いキーワードを割り当て
        if pending_keywords:
            _, keyword_lower, keyword_data = pending_keywords[0]
            
            # キーワードを割り当て
            keyword_data['status'] = KeywordStatus.ASSIGNED.value
            keyword_data['browser_id'] = browser_id
            keyword_data['assigned_at'] = datetime.now().isoformat()
            
            # ブラウザ割り当てを記録
            if browser_id not in self.queue['browser_assignments']:
                self.queue['browser_assignments'][browser_id] = []
            self.queue['browser_assignments'][browser_id].append(keyword_lower)
            
            # キューを保存
            self._save_queue()
            
            keyword = keyword_data.get('keyword', keyword_lower)
            priority = keyword_data.get('priority', 'medium')
            logger.info(f"[QUEUE] Assigned keyword '{keyword}' (priority: {priority}) to browser {browser_id}")
            return keyword
        
        logger.debug(f"[QUEUE] No available keywords for browser {browser_id}")
        return None
    
    def mark_keyword_processing(self, keyword: str, browser_id: int) -> bool:
        """
        キーワードを処理中にマーク
        
        Args:
            keyword: キーワード
            browser_id: ブラウザID
        
        Returns:
            success: 成功フラグ
        """
        keyword_lower = keyword.lower().strip()
        
        if keyword_lower not in self.queue['keywords']:
            logger.warning(f"[QUEUE] Keyword not found: {keyword}")
            return False
        
        keyword_data = self.queue['keywords'][keyword_lower]
        
        # ブラウザIDが一致することを確認
        if keyword_data.get('browser_id') != browser_id:
            logger.warning(f"[QUEUE] Keyword '{keyword}' not assigned to browser {browser_id}")
            return False
        
        # 状態を更新
        keyword_data['status'] = KeywordStatus.PROCESSING.value
        
        # 進捗情報を初期化（初回処理開始時）
        progress = keyword_data.setdefault('progress', {})
        if not progress.get('started_at'):
            progress['started_at'] = datetime.now().isoformat()
            progress['samples_collected'] = 0
            progress['urls_processed'] = 0
            progress['urls_failed'] = 0
            progress['processing_times'] = []
            progress['by_browser'] = {}
        
        # ブラウザ別の処理状況を初期化
        if browser_id not in progress['by_browser']:
            progress['by_browser'][browser_id] = {
                'samples_collected': 0,
                'urls_processed': 0,
                'urls_failed': 0,
                'started_at': datetime.now().isoformat()
            }
        
        self._save_queue()
        
        logger.info(f"[QUEUE] Marked keyword '{keyword}' as processing (browser {browser_id})")
        return True
    
    def mark_keyword_completed(self, keyword: str, browser_id: int) -> bool:
        """
        キーワードを完了にマーク
        
        Args:
            keyword: キーワード
            browser_id: ブラウザID
        
        Returns:
            success: 成功フラグ
        """
        keyword_lower = keyword.lower().strip()
        
        if keyword_lower not in self.queue['keywords']:
            logger.warning(f"[QUEUE] Keyword not found: {keyword}")
            return False
        
        keyword_data = self.queue['keywords'][keyword_lower]
        
        # ブラウザIDが一致することを確認
        if keyword_data.get('browser_id') != browser_id:
            logger.warning(f"[QUEUE] Keyword '{keyword}' not assigned to browser {browser_id}")
            return False
        
        # 状態を更新
        keyword_data['status'] = KeywordStatus.COMPLETED.value
        keyword_data['completed_at'] = datetime.now().isoformat()
        
        # 進捗情報を更新
        progress = keyword_data.setdefault('progress', {})
        progress['finished_at'] = datetime.now().isoformat()
        
        # 成功率を計算
        total_urls = progress.get('urls_processed', 0) + progress.get('urls_failed', 0)
        if total_urls > 0:
            progress['success_rate'] = progress.get('urls_processed', 0) / total_urls
        
        self._save_queue()
        
        logger.info(f"[QUEUE] Marked keyword '{keyword}' as completed (browser {browser_id})")
        return True
    
    def mark_keyword_failed(self, keyword: str, browser_id: int, error: Optional[str] = None) -> bool:
        """
        キーワードを失敗にマーク
        
        Args:
            keyword: キーワード
            browser_id: ブラウザID
            error: エラーメッセージ（オプション）
        
        Returns:
            success: 成功フラグ
        """
        keyword_lower = keyword.lower().strip()
        
        if keyword_lower not in self.queue['keywords']:
            logger.warning(f"[QUEUE] Keyword not found: {keyword}")
            return False
        
        keyword_data = self.queue['keywords'][keyword_lower]
        
        # リトライ回数を増やす
        retry_count = keyword_data.get('retry_count', 0) + 1
        keyword_data['retry_count'] = retry_count
        
        # 最大リトライ回数を超えた場合は失敗、そうでなければpendingに戻す
        max_retries = keyword_data.get('max_retries', 3)
        if retry_count >= max_retries:
            keyword_data['status'] = KeywordStatus.FAILED.value
            keyword_data['failed_at'] = datetime.now().isoformat()
            keyword_data['error'] = error
            logger.warning(f"[QUEUE] Marked keyword '{keyword}' as failed after {retry_count} retries")
        else:
            # pendingに戻して再試行可能にする
            keyword_data['status'] = KeywordStatus.PENDING.value
            keyword_data['browser_id'] = None
            keyword_data['assigned_at'] = None
            logger.info(f"[QUEUE] Reset keyword '{keyword}' to pending for retry ({retry_count}/{max_retries})")
        
        self._save_queue()
        return True
    
    def _reset_timed_out_assignments(self) -> int:
        """
        タイムアウトした割り当てをリセット
        
        Returns:
            reset_count: リセットされたキーワード数
        """
        reset_count = 0
        current_time = datetime.now()
        
        for keyword_lower, keyword_data in self.queue['keywords'].items():
            status = keyword_data.get('status', KeywordStatus.PENDING.value)
            assigned_at_str = keyword_data.get('assigned_at')
            
            # assignedまたはprocessing状態でタイムアウトしている場合
            if status in [KeywordStatus.ASSIGNED.value, KeywordStatus.PROCESSING.value] and assigned_at_str:
                try:
                    assigned_at = datetime.fromisoformat(assigned_at_str)
                    elapsed = (current_time - assigned_at).total_seconds()
                    
                    if elapsed > self.assignment_timeout:
                        # タイムアウトしたのでpendingに戻す
                        keyword_data['status'] = KeywordStatus.PENDING.value
                        keyword_data['browser_id'] = None
                        keyword_data['assigned_at'] = None
                        reset_count += 1
                        logger.warning(f"[QUEUE] Reset timed-out keyword: {keyword_data.get('keyword', keyword_lower)}")
                except Exception as e:
                    logger.warning(f"[QUEUE] Failed to parse assigned_at for keyword {keyword_lower}: {e}")
        
        if reset_count > 0:
            self._save_queue()
        
        return reset_count
    
    def get_keyword_status(self, keyword: str) -> Optional[Dict[str, Any]]:
        """
        キーワードの状態を取得
        
        Args:
            keyword: キーワード
        
        Returns:
            status: キーワード状態辞書（Noneの場合はキーワードが見つからない）
        """
        keyword_lower = keyword.lower().strip()
        
        if keyword_lower not in self.queue['keywords']:
            return None
        
        keyword_data = self.queue['keywords'][keyword_lower].copy()
        return keyword_data
    
    def get_all_keywords(
        self, 
        status_filter: Optional[KeywordStatus] = None,
        priority_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        すべてのキーワードを取得（オプションで状態フィルタ、優先度フィルタ）
        
        Args:
            status_filter: 状態フィルタ（Noneの場合はすべて）
            priority_filter: 優先度フィルタ（low, medium, high, urgent、Noneの場合はすべて）
        
        Returns:
            keywords: キーワードリスト（優先度順にソート）
        """
        keywords = []
        
        for keyword_lower, keyword_data in self.queue['keywords'].items():
            status = keyword_data.get('status')
            priority = keyword_data.get('priority', 'medium')
            
            # 状態フィルタを適用
            if status_filter and status != status_filter.value:
                continue
            
            # 優先度フィルタを適用
            if priority_filter and priority.lower() != priority_filter.lower():
                continue
            
            keywords.append(keyword_data.copy())
        
        # 優先度順にソート（urgent > high > medium > low）
        keywords.sort(key=lambda x: x.get('priority_value', KeywordPriority.get_priority_value(x.get('priority', 'medium'))), reverse=True)
        
        return keywords
    
    def get_browser_keywords(self, browser_id: int) -> List[Dict[str, Any]]:
        """
        ブラウザに割り当てられたキーワードを取得
        
        Args:
            browser_id: ブラウザID
        
        Returns:
            keywords: キーワードリスト
        """
        keywords = []
        
        for keyword_lower, keyword_data in self.queue['keywords'].items():
            if keyword_data.get('browser_id') == browser_id:
                keywords.append(keyword_data.copy())
        
        return keywords
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        統計情報を取得
        
        Returns:
            stats: 統計情報辞書
        """
        stats = {
            'total': len(self.queue['keywords']),
            'pending': 0,
            'assigned': 0,
            'processing': 0,
            'completed': 0,
            'failed': 0,
            'by_browser': {},
            'by_priority': {
                'low': 0,
                'medium': 0,
                'high': 0,
                'urgent': 0
            },
            'progress_stats': {
                'total_samples': 0,
                'total_urls_processed': 0,
                'total_urls_failed': 0,
                'avg_processing_time': 0.0,
                'min_processing_time': 0.0,
                'max_processing_time': 0.0,
                'success_rate': 0.0
            },
            'by_time': {},  # 時間別統計（時間ごとの処理数）
            'by_category': {}  # カテゴリ別統計（将来の拡張用）
        }
        
        all_processing_times = []
        completed_keywords = []
        
        for keyword_data in self.queue['keywords'].values():
            status = keyword_data.get('status', KeywordStatus.PENDING.value)
            priority = keyword_data.get('priority', 'medium')
            
            stats[status] = stats.get(status, 0) + 1
            
            # 優先度別統計
            if priority in stats['by_priority']:
                stats['by_priority'][priority] += 1
            
            browser_id = keyword_data.get('browser_id')
            if browser_id is not None:
                if browser_id not in stats['by_browser']:
                    stats['by_browser'][browser_id] = 0
                stats['by_browser'][browser_id] += 1
            
            # 進捗統計
            progress = keyword_data.get('progress', {})
            if progress:
                samples = progress.get('samples_collected', 0)
                urls_processed = progress.get('urls_processed', 0)
                urls_failed = progress.get('urls_failed', 0)
                processing_times = progress.get('processing_times', [])
                
                stats['progress_stats']['total_samples'] += samples
                stats['progress_stats']['total_urls_processed'] += urls_processed
                stats['progress_stats']['total_urls_failed'] += urls_failed
                all_processing_times.extend(processing_times)
                
                # 時間別統計（追加時刻から時間単位で集計）
                added_at_str = keyword_data.get('added_at')
                if added_at_str:
                    try:
                        added_at = datetime.fromisoformat(added_at_str)
                        hour_key = added_at.strftime('%Y-%m-%d %H:00')
                        if hour_key not in stats['by_time']:
                            stats['by_time'][hour_key] = 0
                        stats['by_time'][hour_key] += 1
                    except Exception:
                        pass
                
                # 完了したキーワードの統計
                if status == KeywordStatus.COMPLETED.value:
                    completed_keywords.append(keyword_data)
        
        # 処理時間の統計
        if all_processing_times:
            stats['progress_stats']['avg_processing_time'] = sum(all_processing_times) / len(all_processing_times)
            stats['progress_stats']['min_processing_time'] = min(all_processing_times)
            stats['progress_stats']['max_processing_time'] = max(all_processing_times)
        
        # 成功率の計算
        total_urls = stats['progress_stats']['total_urls_processed'] + stats['progress_stats']['total_urls_failed']
        if total_urls > 0:
            stats['progress_stats']['success_rate'] = stats['progress_stats']['total_urls_processed'] / total_urls
        
        return stats
    
    def clear_completed_keywords(self, older_than_hours: int = 24) -> int:
        """
        完了したキーワードをクリア（一定時間経過後）
        
        Args:
            older_than_hours: クリアする時間（時間）
        
        Returns:
            cleared_count: クリアされたキーワード数
        """
        cleared_count = 0
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        keywords_to_remove = []
        for keyword_lower, keyword_data in self.queue['keywords'].items():
            status = keyword_data.get('status', KeywordStatus.PENDING.value)
            completed_at_str = keyword_data.get('completed_at')
            
            if status == KeywordStatus.COMPLETED.value and completed_at_str:
                try:
                    completed_at = datetime.fromisoformat(completed_at_str)
                    if completed_at < cutoff_time:
                        keywords_to_remove.append(keyword_lower)
                except Exception:
                    pass
        
        for keyword_lower in keywords_to_remove:
            del self.queue['keywords'][keyword_lower]
            cleared_count += 1
        
        if cleared_count > 0:
            self._save_queue()
            logger.info(f"[QUEUE] Cleared {cleared_count} completed keywords older than {older_than_hours} hours")
        
        return cleared_count
    
    def _load_cache(self) -> Dict[str, Any]:
        """
        キャッシュを読み込み
        
        Returns:
            cache: キャッシュ辞書
        """
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                
                # 期限切れのキャッシュを削除
                self._cleanup_expired_cache()
                
                logger.info(f"[CACHE] Loaded {len(self.cache.get('classifications', {}))} cached classifications")
            else:
                self.cache = {
                    'classifications': {},
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'expiration_hours': self.cache_expiration_hours
                    }
                }
                logger.info("[CACHE] Created new cache")
        except Exception as e:
            logger.warning(f"[CACHE] Failed to load cache: {e}")
            self.cache = {
                'classifications': {},
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'expiration_hours': self.cache_expiration_hours
                }
            }
        
        return self.cache
    
    def _save_cache(self) -> bool:
        """
        キャッシュを保存
        
        Returns:
            success: 成功フラグ
        """
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"[CACHE] Failed to save cache: {e}")
            return False
    
    def _cleanup_expired_cache(self) -> int:
        """
        期限切れのキャッシュを削除
        
        Returns:
            removed_count: 削除されたキャッシュ数
        """
        removed_count = 0
        current_time = datetime.now()
        classifications = self.cache.get('classifications', {})
        
        keywords_to_remove = []
        for keyword_lower, cache_data in classifications.items():
            cached_at_str = cache_data.get('cached_at')
            if cached_at_str:
                try:
                    cached_at = datetime.fromisoformat(cached_at_str)
                    elapsed_hours = (current_time - cached_at).total_seconds() / 3600
                    if elapsed_hours > self.cache_expiration_hours:
                        keywords_to_remove.append(keyword_lower)
                except Exception:
                    keywords_to_remove.append(keyword_lower)
        
        for keyword_lower in keywords_to_remove:
            del classifications[keyword_lower]
            removed_count += 1
        
        if removed_count > 0:
            self._save_cache()
            logger.info(f"[CACHE] Removed {removed_count} expired cache entries")
        
        return removed_count
    
    def get_cached_classification(self, keyword: str) -> Optional[Dict[str, Any]]:
        """
        キーワードの分類結果をキャッシュから取得
        
        Args:
            keyword: キーワード
        
        Returns:
            classification: 分類結果辞書（Noneの場合はキャッシュに存在しない）
        """
        keyword_lower = keyword.lower().strip()
        classifications = self.cache.get('classifications', {})
        
        if keyword_lower not in classifications:
            return None
        
        cache_data = classifications[keyword_lower]
        cached_at_str = cache_data.get('cached_at')
        
        # 期限切れチェック
        if cached_at_str:
            try:
                cached_at = datetime.fromisoformat(cached_at_str)
                elapsed_hours = (datetime.now() - cached_at).total_seconds() / 3600
                if elapsed_hours > self.cache_expiration_hours:
                    # 期限切れなので削除
                    del classifications[keyword_lower]
                    self._save_cache()
                    return None
            except Exception:
                return None
        
        logger.debug(f"[CACHE] Cache hit for keyword: {keyword}")
        return cache_data.get('classification')
    
    def cache_classification(
        self,
        keyword: str,
        classification: Dict[str, Any]
    ) -> bool:
        """
        キーワードの分類結果をキャッシュに保存
        
        Args:
            keyword: キーワード
            classification: 分類結果辞書（カテゴリ、言語、ジャンルなど）
        
        Returns:
            success: 成功フラグ
        """
        keyword_lower = keyword.lower().strip()
        classifications = self.cache.setdefault('classifications', {})
        
        classifications[keyword_lower] = {
            'keyword': keyword,
            'classification': classification,
            'cached_at': datetime.now().isoformat()
        }
        
        success = self._save_cache()
        if success:
            logger.debug(f"[CACHE] Cached classification for keyword: {keyword}")
        
        return success
    
    def invalidate_cache(self, keyword: Optional[str] = None) -> int:
        """
        キャッシュを無効化
        
        Args:
            keyword: キーワード（Noneの場合はすべてのキャッシュを無効化）
        
        Returns:
            invalidated_count: 無効化されたキャッシュ数
        """
        classifications = self.cache.get('classifications', {})
        
        if keyword is None:
            # すべてのキャッシュを無効化
            invalidated_count = len(classifications)
            classifications.clear()
        else:
            # 特定のキーワードのキャッシュを無効化
            keyword_lower = keyword.lower().strip()
            if keyword_lower in classifications:
                del classifications[keyword_lower]
                invalidated_count = 1
            else:
                invalidated_count = 0
        
        if invalidated_count > 0:
            self._save_cache()
            logger.info(f"[CACHE] Invalidated {invalidated_count} cache entries")
        
        return invalidated_count


def main():
    """メイン関数（テスト用）"""
    coordinator = KeywordCoordinator()
    
    # テストキーワードを追加
    test_keywords = ["Python", "Rust", "TypeScript", "JavaScript", "Java"]
    coordinator.add_keywords(test_keywords, source="test")
    
    # キーワードを取得
    browser_id = 0
    keyword = coordinator.get_next_keyword(browser_id)
    print(f"Browser {browser_id} got keyword: {keyword}")
    
    # 処理中にマーク
    coordinator.mark_keyword_processing(keyword, browser_id)
    
    # 完了にマーク
    coordinator.mark_keyword_completed(keyword, browser_id)
    
    # 統計を表示
    stats = coordinator.get_statistics()
    print(f"Statistics: {stats}")


if __name__ == "__main__":
    main()

