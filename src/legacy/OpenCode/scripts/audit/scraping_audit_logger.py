#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webスクレイピング専用SQL（ISO基準監査ログ）統合管理

Webスクレイピング動作の完全監査ログ記録、ISO基準に準拠した監査ログスキーマ実装

Usage:
    from scripts.audit.scraping_audit_logger import ScrapingAuditLogger
    
    logger = ScrapingAuditLogger()
    logger.log_scraping_session(session_id="session_001", keyword="Python", url="https://example.com")
"""

import sys
import json
import sqlite3
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraping_audit_logger.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ScrapingSession:
    """スクレイピングセッション"""
    session_id: str
    started_at: str
    browser_index: int
    keyword: str
    status: str  # active/completed/denied/error
    samples_collected: int
    last_activity: str
    quadruple_thinking: Optional[Dict] = None
    four_class_classification: Optional[Dict] = None
    ended_at: Optional[str] = None


@dataclass
class ScrapingEvent:
    """スクレイピングイベント"""
    event_id: str
    session_id: str
    timestamp: str
    event_type: str  # scraping_start/scraping_end/url_access/data_collected/so8t_decision/power_failure/recovery
    url: Optional[str] = None
    keyword: Optional[str] = None
    data_hash: Optional[str] = None
    quadruple_thinking: Optional[Dict] = None
    four_class_classification: Optional[Dict] = None
    decision: Optional[str] = None  # ALLOW/ESCALATE/DENY/REFUSE
    details: Optional[Dict] = None


@dataclass
class DataCleaningEvent:
    """データクレンジングイベント"""
    event_id: str
    timestamp: str
    session_id: Optional[str] = None
    input_samples: int = 0
    output_samples: int = 0
    cleaning_method: Optional[str] = None
    quality_score: Optional[float] = None
    details: Optional[Dict] = None


@dataclass
class DatasetCreationEvent:
    """データセット作成イベント"""
    event_id: str
    timestamp: str
    dataset_name: str
    dataset_type: str  # train/val/test
    samples_count: int
    source_sessions: List[str]
    details: Optional[Dict] = None


class ScrapingAuditLogger:
    """Webスクレイピング専用SQL（ISO基準監査ログ）統合管理クラス"""
    
    def __init__(self, db_path: Path = Path("database/so8t_scraping_audit.db")):
        """
        初期化
        
        Args:
            db_path: データベースパス
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        logger.info("="*80)
        logger.info("Scraping Audit Logger Initialized")
        logger.info("="*80)
        logger.info(f"Database path: {self.db_path}")
    
    def _init_database(self):
        """データベース初期化（ISO基準準拠）"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # スクレイピングセッションテーブル（ISO基準: セッション追跡）
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS scraping_sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            browser_index INTEGER NOT NULL,
            keyword TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('active', 'completed', 'denied', 'error')),
            samples_collected INTEGER DEFAULT 0,
            last_activity TEXT NOT NULL,
            quadruple_thinking TEXT,
            four_class_classification TEXT,
            ended_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # スクレイピングイベントテーブル（ISO基準: 完全監査証跡）
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS scraping_events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL CHECK(event_type IN (
                'scraping_start', 'scraping_end', 'url_access', 'data_collected',
                'so8t_decision', 'power_failure', 'recovery', 'duplicate_detected',
                'error', 'checkpoint_saved'
            )),
            url TEXT,
            keyword TEXT,
            data_hash TEXT,
            quadruple_thinking TEXT,
            four_class_classification TEXT,
            decision TEXT CHECK(decision IN ('ALLOW', 'ESCALATE', 'DENY', 'REFUSE')),
            details TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES scraping_sessions(session_id)
        )
        """)
        
        # データクレンジングイベントテーブル（ISO基準: データ処理追跡）
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_cleaning_events (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            session_id TEXT,
            input_samples INTEGER DEFAULT 0,
            output_samples INTEGER DEFAULT 0,
            cleaning_method TEXT,
            quality_score REAL,
            details TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES scraping_sessions(session_id)
        )
        """)
        
        # データセット作成イベントテーブル（ISO基準: データセット追跡）
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_creation_events (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            dataset_type TEXT NOT NULL CHECK(dataset_type IN ('train', 'val', 'test')),
            samples_count INTEGER NOT NULL,
            source_sessions TEXT,
            details TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 重複防止テーブル（ISO基準: データ整合性）
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS duplicate_prevention (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url_hash TEXT NOT NULL,
            keyword TEXT NOT NULL,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            data_hash TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(url_hash, keyword, session_id)
        )
        """)
        
        # 電源断・復旧情報テーブル（ISO基準: 障害追跡）
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS power_failure_recovery (
            recovery_id TEXT PRIMARY KEY,
            failure_timestamp TEXT NOT NULL,
            recovery_timestamp TEXT NOT NULL,
            session_id TEXT NOT NULL,
            checkpoint_path TEXT,
            samples_lost INTEGER DEFAULT 0,
            samples_recovered INTEGER DEFAULT 0,
            recovery_status TEXT CHECK(recovery_status IN ('success', 'partial', 'failed')),
            details TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES scraping_sessions(session_id)
        )
        """)
        
        # インデックス作成（ISO基準: 検索性能）
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON scraping_sessions(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_keyword ON scraping_sessions(keyword)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_browser ON scraping_sessions(browser_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON scraping_events(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON scraping_events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON scraping_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_duplicate_url_hash ON duplicate_prevention(url_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_duplicate_keyword ON duplicate_prevention(keyword)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recovery_session ON power_failure_recovery(session_id)")
        
        conn.commit()
        conn.close()
        
        logger.info("[OK] Database initialized with ISO-compliant schema")
    
    @contextmanager
    def connect(self):
        """データベース接続コンテキストマネージャー"""
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def log_scraping_session(self, session: ScrapingSession) -> bool:
        """
        スクレイピングセッションを記録
        
        Args:
            session: スクレイピングセッション
            
        Returns:
            success: 成功フラグ
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                
                # 既存セッションの更新または新規作成
                cursor.execute("""
                INSERT OR REPLACE INTO scraping_sessions
                (session_id, started_at, browser_index, keyword, status, samples_collected,
                 last_activity, quadruple_thinking, four_class_classification, ended_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.started_at,
                    session.browser_index,
                    session.keyword,
                    session.status,
                    session.samples_collected,
                    session.last_activity,
                    json.dumps(session.quadruple_thinking, ensure_ascii=False) if session.quadruple_thinking else None,
                    json.dumps(session.four_class_classification, ensure_ascii=False) if session.four_class_classification else None,
                    session.ended_at,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                logger.debug(f"[AUDIT] Session logged: {session.session_id}")
                return True
                
        except Exception as e:
            logger.error(f"[AUDIT] Failed to log session: {e}")
            return False
    
    def log_scraping_event(self, event: ScrapingEvent) -> bool:
        """
        スクレイピングイベントを記録
        
        Args:
            event: スクレイピングイベント
            
        Returns:
            success: 成功フラグ
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                INSERT INTO scraping_events
                (event_id, session_id, timestamp, event_type, url, keyword, data_hash,
                 quadruple_thinking, four_class_classification, decision, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.session_id,
                    event.timestamp,
                    event.event_type,
                    event.url,
                    event.keyword,
                    event.data_hash,
                    json.dumps(event.quadruple_thinking, ensure_ascii=False) if event.quadruple_thinking else None,
                    json.dumps(event.four_class_classification, ensure_ascii=False) if event.four_class_classification else None,
                    event.decision,
                    json.dumps(event.details, ensure_ascii=False) if event.details else None
                ))
                
                conn.commit()
                logger.debug(f"[AUDIT] Event logged: {event.event_id} ({event.event_type})")
                return True
                
        except Exception as e:
            logger.error(f"[AUDIT] Failed to log event: {e}")
            return False
    
    def log_data_cleaning_event(self, event: DataCleaningEvent) -> bool:
        """
        データクレンジングイベントを記録
        
        Args:
            event: データクレンジングイベント
            
        Returns:
            success: 成功フラグ
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                INSERT INTO data_cleaning_events
                (event_id, timestamp, session_id, input_samples, output_samples,
                 cleaning_method, quality_score, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp,
                    event.session_id,
                    event.input_samples,
                    event.output_samples,
                    event.cleaning_method,
                    event.quality_score,
                    json.dumps(event.details, ensure_ascii=False) if event.details else None
                ))
                
                conn.commit()
                logger.debug(f"[AUDIT] Data cleaning event logged: {event.event_id}")
                return True
                
        except Exception as e:
            logger.error(f"[AUDIT] Failed to log data cleaning event: {e}")
            return False
    
    def log_dataset_creation_event(self, event: DatasetCreationEvent) -> bool:
        """
        データセット作成イベントを記録
        
        Args:
            event: データセット作成イベント
            
        Returns:
            success: 成功フラグ
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                INSERT INTO dataset_creation_events
                (event_id, timestamp, dataset_name, dataset_type, samples_count, source_sessions, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp,
                    event.dataset_name,
                    event.dataset_type,
                    event.samples_count,
                    json.dumps(event.source_sessions, ensure_ascii=False),
                    json.dumps(event.details, ensure_ascii=False) if event.details else None
                ))
                
                conn.commit()
                logger.debug(f"[AUDIT] Dataset creation event logged: {event.event_id}")
                return True
                
        except Exception as e:
            logger.error(f"[AUDIT] Failed to log dataset creation event: {e}")
            return False
    
    def check_duplicate(self, url: str, keyword: str, session_id: str) -> bool:
        """
        重複チェック
        
        Args:
            url: URL
            keyword: キーワード
            session_id: セッションID
            
        Returns:
            is_duplicate: 重複しているかどうか
        """
        try:
            url_hash = hashlib.sha256(url.encode()).hexdigest()
            
            with self.connect() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                SELECT COUNT(*) FROM duplicate_prevention
                WHERE url_hash = ? AND keyword = ? AND session_id = ?
                """, (url_hash, keyword, session_id))
                
                count = cursor.fetchone()[0]
                return count > 0
                
        except Exception as e:
            logger.error(f"[AUDIT] Failed to check duplicate: {e}")
            return False
    
    def register_duplicate(self, url: str, keyword: str, session_id: str, data_hash: Optional[str] = None) -> bool:
        """
        重複を登録
        
        Args:
            url: URL
            keyword: キーワード
            session_id: セッションID
            data_hash: データハッシュ（オプション）
            
        Returns:
            success: 成功フラグ
        """
        try:
            url_hash = hashlib.sha256(url.encode()).hexdigest()
            
            with self.connect() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                INSERT OR IGNORE INTO duplicate_prevention
                (url_hash, keyword, session_id, timestamp, data_hash)
                VALUES (?, ?, ?, ?, ?)
                """, (
                    url_hash,
                    keyword,
                    session_id,
                    datetime.now().isoformat(),
                    data_hash
                ))
                
                conn.commit()
                logger.debug(f"[AUDIT] Duplicate registered: {url_hash[:16]}...")
                return True
                
        except Exception as e:
            logger.error(f"[AUDIT] Failed to register duplicate: {e}")
            return False
    
    def log_power_failure_recovery(
        self,
        recovery_id: str,
        failure_timestamp: str,
        recovery_timestamp: str,
        session_id: str,
        checkpoint_path: Optional[str] = None,
        samples_lost: int = 0,
        samples_recovered: int = 0,
        recovery_status: str = "success",
        details: Optional[Dict] = None
    ) -> bool:
        """
        電源断・復旧情報を記録
        
        Args:
            recovery_id: 復旧ID
            failure_timestamp: 障害発生時刻
            recovery_timestamp: 復旧時刻
            session_id: セッションID
            checkpoint_path: チェックポイントパス
            samples_lost: 失われたサンプル数
            samples_recovered: 復旧したサンプル数
            recovery_status: 復旧ステータス
            details: 詳細情報
            
        Returns:
            success: 成功フラグ
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                INSERT INTO power_failure_recovery
                (recovery_id, failure_timestamp, recovery_timestamp, session_id,
                 checkpoint_path, samples_lost, samples_recovered, recovery_status, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    recovery_id,
                    failure_timestamp,
                    recovery_timestamp,
                    session_id,
                    checkpoint_path,
                    samples_lost,
                    samples_recovered,
                    recovery_status,
                    json.dumps(details, ensure_ascii=False) if details else None
                ))
                
                conn.commit()
                logger.info(f"[AUDIT] Power failure recovery logged: {recovery_id}")
                return True
                
        except Exception as e:
            logger.error(f"[AUDIT] Failed to log power failure recovery: {e}")
            return False
    
    def get_active_sessions(self) -> List[Dict]:
        """
        アクティブなセッションを取得
        
        Returns:
            sessions: セッションのリスト
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                SELECT * FROM scraping_sessions
                WHERE status = 'active'
                ORDER BY last_activity DESC
                """)
                
                sessions = []
                for row in cursor.fetchall():
                    session = dict(row)
                    if session.get('quadruple_thinking'):
                        session['quadruple_thinking'] = json.loads(session['quadruple_thinking'])
                    if session.get('four_class_classification'):
                        session['four_class_classification'] = json.loads(session['four_class_classification'])
                    sessions.append(session)
                
                return sessions
                
        except Exception as e:
            logger.error(f"[AUDIT] Failed to get active sessions: {e}")
            return []
    
    def get_session_events(self, session_id: str, limit: int = 100) -> List[Dict]:
        """
        セッションのイベントを取得
        
        Args:
            session_id: セッションID
            limit: 取得件数
            
        Returns:
            events: イベントのリスト
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                SELECT * FROM scraping_events
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """, (session_id, limit))
                
                events = []
                for row in cursor.fetchall():
                    event = dict(row)
                    if event.get('quadruple_thinking'):
                        event['quadruple_thinking'] = json.loads(event['quadruple_thinking'])
                    if event.get('four_class_classification'):
                        event['four_class_classification'] = json.loads(event['four_class_classification'])
                    if event.get('details'):
                        event['details'] = json.loads(event['details'])
                    events.append(event)
                
                return events
                
        except Exception as e:
            logger.error(f"[AUDIT] Failed to get session events: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        統計情報を取得
        
        Returns:
            statistics: 統計情報の辞書
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # セッション統計
                cursor.execute("SELECT COUNT(*) FROM scraping_sessions")
                stats['total_sessions'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM scraping_sessions WHERE status = 'active'")
                stats['active_sessions'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM scraping_sessions WHERE status = 'completed'")
                stats['completed_sessions'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT SUM(samples_collected) FROM scraping_sessions")
                stats['total_samples'] = cursor.fetchone()[0] or 0
                
                # イベント統計
                cursor.execute("SELECT COUNT(*) FROM scraping_events")
                stats['total_events'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM scraping_events WHERE event_type = 'so8t_decision'")
                stats['so8t_decisions'] = cursor.fetchone()[0]
                
                # 重複統計
                cursor.execute("SELECT COUNT(*) FROM duplicate_prevention")
                stats['duplicates_prevented'] = cursor.fetchone()[0]
                
                # 電源断・復旧統計
                cursor.execute("SELECT COUNT(*) FROM power_failure_recovery")
                stats['power_failures'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            logger.error(f"[AUDIT] Failed to get statistics: {e}")
            return {}


def main():
    """テスト実行"""
    logger.info("="*80)
    logger.info("Scraping Audit Logger Test")
    logger.info("="*80)
    
    audit_logger = ScrapingAuditLogger()
    
    # テストセッション記録
    session = ScrapingSession(
        session_id="test_session_001",
        started_at=datetime.now().isoformat(),
        browser_index=0,
        keyword="Python",
        status="active",
        samples_collected=10,
        last_activity=datetime.now().isoformat()
    )
    
    audit_logger.log_scraping_session(session)
    logger.info("[OK] Test session logged")
    
    # テストイベント記録
    event = ScrapingEvent(
        event_id="test_event_001",
        session_id="test_session_001",
        timestamp=datetime.now().isoformat(),
        event_type="url_access",
        url="https://example.com",
        keyword="Python"
    )
    
    audit_logger.log_scraping_event(event)
    logger.info("[OK] Test event logged")
    
    # 統計情報取得
    stats = audit_logger.get_statistics()
    logger.info(f"[OK] Statistics: {stats}")
    
    logger.info("[OK] Test completed")


if __name__ == "__main__":
    main()

