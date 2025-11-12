"""
SQL監査ログシステム（本番環境対応版）
防衛・航空宇宙・運輸向けセキュアLLMOpsの完全監査

記録内容:
- 入力者情報
- 入力内容
- 出力内容
- 判定結果（ALLOW/ESCALATION/DENY）
- タイムスタンプ
- セッションID
- エビングハウス忘却曲線用メタデータ

Author: SO8T Project Team
Date: 2024-11-06
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """監査ログエントリ"""
    session_id: str
    user_id: str
    input_text: str
    output_text: str
    judgment: str
    confidence: float
    domain: str
    timestamp: str
    processing_time_ms: float
    model_version: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """
    SQL監査ログシステム
    
    すべての入出力を監査データベースに記録し、
    コンプライアンスと事後分析を可能にする。
    """
    
    def __init__(self, db_path: str = "database/so8t_compliance.db"):
        """
        Args:
            db_path: データベースパス
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self._initialize_db()
        
        logger.info(f"[AuditLogger] Initialized with database: {self.db_path}")
    
    def _initialize_db(self):
        """データベースを初期化"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        
        # 監査ログテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                input_text TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                output_text TEXT NOT NULL,
                output_hash TEXT NOT NULL,
                judgment TEXT NOT NULL,
                confidence REAL NOT NULL,
                domain TEXT,
                timestamp TEXT NOT NULL,
                processing_time_ms REAL,
                model_version TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # エビングハウス忘却曲線用テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forgetting_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_log_id INTEGER NOT NULL,
                importance_score REAL DEFAULT 0.5,
                review_count INTEGER DEFAULT 0,
                last_review TIMESTAMP,
                next_review TIMESTAMP,
                retention_rate REAL DEFAULT 1.0,
                FOREIGN KEY (audit_log_id) REFERENCES audit_logs(id)
            )
        """)
        
        # ユーザー統計テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id TEXT PRIMARY KEY,
                total_queries INTEGER DEFAULT 0,
                allow_count INTEGER DEFAULT 0,
                escalation_count INTEGER DEFAULT 0,
                deny_count INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.0,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # インデックス作成
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp 
            ON audit_logs(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id 
            ON audit_logs(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_judgment 
            ON audit_logs(judgment)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_forgetting_curve_next_review 
            ON forgetting_curve(next_review)
        """)
        
        self.conn.commit()
        logger.info("[AuditLogger] Database initialized")
    
    def log(self, entry: AuditEntry) -> int:
        """
        監査ログを記録
        
        Args:
            entry: 監査エントリ
            
        Returns:
            ログID
        """
        cursor = self.conn.cursor()
        
        # ハッシュ計算
        input_hash = hashlib.sha256(entry.input_text.encode()).hexdigest()
        output_hash = hashlib.sha256(entry.output_text.encode()).hexdigest()
        
        # ログ挿入
        cursor.execute("""
            INSERT INTO audit_logs (
                session_id, user_id, input_text, input_hash,
                output_text, output_hash, judgment, confidence,
                domain, timestamp, processing_time_ms, model_version, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.session_id,
            entry.user_id,
            entry.input_text,
            input_hash,
            entry.output_text,
            output_hash,
            entry.judgment,
            entry.confidence,
            entry.domain,
            entry.timestamp,
            entry.processing_time_ms,
            entry.model_version,
            json.dumps(entry.metadata, ensure_ascii=False)
        ))
        
        log_id = cursor.lastrowid
        
        # 忘却曲線エントリを作成
        importance = self._calculate_importance(entry)
        next_review = self._calculate_next_review(datetime.now(), 0)
        
        cursor.execute("""
            INSERT INTO forgetting_curve (
                audit_log_id, importance_score, next_review
            ) VALUES (?, ?, ?)
        """, (log_id, importance, next_review))
        
        # ユーザー統計を更新
        self._update_user_stats(entry.user_id, entry.judgment, entry.confidence)
        
        self.conn.commit()
        
        logger.debug(f"[AuditLogger] Logged entry {log_id}: "
                    f"user={entry.user_id}, judgment={entry.judgment}")
        
        return log_id
    
    def _calculate_importance(self, entry: AuditEntry) -> float:
        """重要度を計算"""
        importance = 0.5
        
        # DENY判定は重要度高
        if entry.judgment == "DENY":
            importance += 0.3
        elif entry.judgment == "ESCALATION":
            importance += 0.2
        
        # 低信頼度は重要（レビューが必要）
        if entry.confidence < 0.7:
            importance += 0.2
        
        # ドメイン特化は重要
        if entry.domain in ["defense", "aerospace", "transport"]:
            importance += 0.1
        
        return min(importance, 1.0)
    
    def _calculate_next_review(
        self,
        last_review: datetime,
        review_count: int
    ) -> str:
        """
        次回復習日を計算（エビングハウス忘却曲線）
        
        Args:
            last_review: 最終復習日
            review_count: 復習回数
            
        Returns:
            次回復習日（ISO形式）
        """
        # エビングハウス間隔: 1日, 3日, 7日, 14日, 30日...
        intervals = [1, 3, 7, 14, 30, 60, 120]
        
        if review_count < len(intervals):
            days = intervals[review_count]
        else:
            days = intervals[-1] * (2 ** (review_count - len(intervals)))
        
        next_review = last_review + timedelta(days=days)
        return next_review.isoformat()
    
    def _update_user_stats(self, user_id: str, judgment: str, confidence: float):
        """ユーザー統計を更新"""
        cursor = self.conn.cursor()
        
        # 既存レコード確認
        cursor.execute("""
            SELECT * FROM user_stats WHERE user_id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        if row:
            # 更新
            cursor.execute(f"""
                UPDATE user_stats SET
                    total_queries = total_queries + 1,
                    {judgment.lower()}_count = {judgment.lower()}_count + 1,
                    avg_confidence = (avg_confidence * total_queries + ?) / (total_queries + 1),
                    last_seen = ?
                WHERE user_id = ?
            """, (confidence, datetime.now().isoformat(), user_id))
        else:
            # 新規作成
            cursor.execute(f"""
                INSERT INTO user_stats (
                    user_id, total_queries, {judgment.lower()}_count, avg_confidence
                ) VALUES (?, 1, 1, ?)
            """, (user_id, confidence))
    
    def get_reviews_due(self, limit: int = 100) -> List[Dict[str, Any]]:
        """復習が必要なエントリを取得"""
        cursor = self.conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            SELECT 
                al.*, fc.importance_score, fc.review_count, fc.next_review
            FROM audit_logs al
            JOIN forgetting_curve fc ON al.id = fc.audit_log_id
            WHERE fc.next_review <= ?
            ORDER BY fc.importance_score DESC, fc.next_review ASC
            LIMIT ?
        """, (now, limit))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def mark_reviewed(self, audit_log_id: int):
        """復習完了をマーク"""
        cursor = self.conn.cursor()
        
        # 現在の復習回数を取得
        cursor.execute("""
            SELECT review_count FROM forgetting_curve
            WHERE audit_log_id = ?
        """, (audit_log_id,))
        
        row = cursor.fetchone()
        if not row:
            return
        
        review_count = row[0]
        now = datetime.now()
        next_review = self._calculate_next_review(now, review_count + 1)
        
        # 更新
        cursor.execute("""
            UPDATE forgetting_curve SET
                review_count = review_count + 1,
                last_review = ?,
                next_review = ?,
                retention_rate = retention_rate * 0.95
            WHERE audit_log_id = ?
        """, (now.isoformat(), next_review, audit_log_id))
        
        self.conn.commit()
    
    def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """ユーザー統計を取得"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM user_stats WHERE user_id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def query_logs(
        self,
        user_id: Optional[str] = None,
        judgment: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """ログをクエリ"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if judgment:
            query += " AND judgment = ?"
            params.append(judgment)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def close(self):
        """接続をクローズ"""
        if self.conn:
            self.conn.close()
            logger.info("[AuditLogger] Database connection closed")


# テスト用
if __name__ == "__main__":
    print("=" * 80)
    print("Audit Logger Test")
    print("=" * 80)
    
    # テスト用DB
    logger_inst = AuditLogger(db_path="test_audit.db")
    
    # テストエントリ
    entry = AuditEntry(
        session_id="test_session_001",
        user_id="user_001",
        input_text="防衛システムについて教えてください",
        output_text="防衛システムは...",
        judgment="ALLOW",
        confidence=0.92,
        domain="defense",
        timestamp=datetime.now().isoformat(),
        processing_time_ms=150.5,
        model_version="so8t-phi4-v1.0",
        metadata={"test": True}
    )
    
    log_id = logger_inst.log(entry)
    print(f"\n[Test] Logged entry with ID: {log_id}")
    
    # ユーザー統計
    stats = logger_inst.get_user_stats("user_001")
    print(f"\n[Test] User stats: {stats}")
    
    # ログクエリ
    logs = logger_inst.query_logs(user_id="user_001", limit=10)
    print(f"\n[Test] Found {len(logs)} logs")
    
    logger_inst.close()
    
    print("\n" + "=" * 80)
    print("[AuditLogger] Test complete!")
    print("=" * 80)

