"""
SQL監査ログシステム
完全監査・ユーザー統計・エビングハウス忘却曲線統合
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    SQL監査ログシステム
    
    主要機能:
    - 入出力完全記録
    - ユーザー統計
    - エビングハウス忘却曲線統合
    - 重要度計算
    - 復習スケジュール
    """
    
    def __init__(self, db_path: str = "database/so8t_compliance.db"):
        """
        Args:
            db_path: SQLiteデータベースパス
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """データベース初期化"""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        
        # 監査ログテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                session_id TEXT,
                query TEXT,
                response TEXT,
                judgment TEXT,
                confidence REAL,
                model_name TEXT,
                processing_time_ms INTEGER,
                metadata TEXT
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
                first_access TIMESTAMP,
                last_access TIMESTAMP
            )
        """)
        
        # 忘却曲線テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forgetting_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_id INTEGER,
                importance_score REAL,
                next_review_date TIMESTAMP,
                review_count INTEGER DEFAULT 0,
                retention_rate REAL DEFAULT 1.0,
                FOREIGN KEY (log_id) REFERENCES audit_logs(id)
            )
        """)
        
        self.conn.commit()
        logger.info(f"[AUDIT] Database initialized: {self.db_path}")
    
    def log(
        self,
        user_id: str,
        session_id: str,
        query: str,
        response: str,
        judgment: str,
        confidence: float,
        model_name: str = "phi4-so8t-ja",
        processing_time_ms: int = 0,
        metadata: Optional[str] = None,
    ) -> int:
        """
        監査ログを記録
        
        Args:
            user_id: ユーザーID
            session_id: セッションID
            query: クエリ
            response: 応答
            judgment: 判定（ALLOW/ESCALATION/DENY）
            confidence: 信頼度
            model_name: モデル名
            processing_time_ms: 処理時間（ミリ秒）
            metadata: メタデータ（JSON文字列）
        
        Returns:
            log_id: ログID
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO audit_logs (
                user_id, session_id, query, response, judgment,
                confidence, model_name, processing_time_ms, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, session_id, query, response, judgment,
            confidence, model_name, processing_time_ms, metadata
        ))
        
        log_id = cursor.lastrowid
        
        # ユーザー統計更新
        self._update_user_stats(user_id, judgment, confidence)
        
        # 重要度計算＆忘却曲線登録
        importance_score = self._calculate_importance(query, judgment, confidence)
        self._register_forgetting_curve(log_id, importance_score)
        
        self.conn.commit()
        
        logger.debug(f"[AUDIT] Logged: log_id={log_id}, user={user_id}, judgment={judgment}")
        
        return log_id
    
    def _update_user_stats(self, user_id: str, judgment: str, confidence: float):
        """ユーザー統計更新"""
        cursor = self.conn.cursor()
        
        # 既存統計を取得
        cursor.execute("SELECT * FROM user_stats WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        
        if row is None:
            # 新規ユーザー
            cursor.execute("""
                INSERT INTO user_stats (
                    user_id, total_queries, allow_count, escalation_count,
                    deny_count, avg_confidence, first_access, last_access
                ) VALUES (?, 1, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                1 if judgment == 'ALLOW' else 0,
                1 if judgment == 'ESCALATION' else 0,
                1 if judgment == 'DENY' else 0,
                confidence,
                datetime.now(),
                datetime.now()
            ))
        else:
            # 既存ユーザー更新
            total = row[1] + 1
            allow = row[2] + (1 if judgment == 'ALLOW' else 0)
            escalation = row[3] + (1 if judgment == 'ESCALATION' else 0)
            deny = row[4] + (1 if judgment == 'DENY' else 0)
            avg_conf = (row[5] * row[1] + confidence) / total
            
            cursor.execute("""
                UPDATE user_stats
                SET total_queries = ?, allow_count = ?, escalation_count = ?,
                    deny_count = ?, avg_confidence = ?, last_access = ?
                WHERE user_id = ?
            """, (total, allow, escalation, deny, avg_conf, datetime.now(), user_id))
    
    def _calculate_importance(self, query: str, judgment: str, confidence: float) -> float:
        """
        重要度計算
        
        重要度 = 判定重み × (1 - 信頼度) × クエリ長補正
        """
        # 判定重み
        judgment_weight = {
            'ALLOW': 0.3,
            'ESCALATION': 0.7,
            'DENY': 1.0,
        }.get(judgment, 0.5)
        
        # 信頼度補正（低信頼度 = 高重要度）
        confidence_factor = 1.0 - confidence
        
        # クエリ長補正（長いクエリ = 高重要度）
        length_factor = min(1.0, len(query) / 100)
        
        importance = judgment_weight * confidence_factor * (0.5 + 0.5 * length_factor)
        
        return importance
    
    def _register_forgetting_curve(self, log_id: int, importance_score: float):
        """
        エビングハウス忘却曲線に登録
        
        復習スケジュール: 1日, 3日, 7日, 14日, 30日, 60日, 120日...
        """
        cursor = self.conn.cursor()
        
        # 次回復習日（重要度に応じて調整）
        if importance_score >= 0.8:
            next_review_days = 1  # 高重要度: 1日後
        elif importance_score >= 0.5:
            next_review_days = 3  # 中重要度: 3日後
        else:
            next_review_days = 7  # 低重要度: 7日後
        
        next_review_date = datetime.now() + timedelta(days=next_review_days)
        
        cursor.execute("""
            INSERT INTO forgetting_curve (
                log_id, importance_score, next_review_date, review_count, retention_rate
            ) VALUES (?, ?, ?, 0, 1.0)
        """, (log_id, importance_score, next_review_date))
    
    def get_reviews_due(self, limit: int = 10) -> List[Dict]:
        """
        復習期限が来たログを取得
        
        Args:
            limit: 最大取得数
        
        Returns:
            reviews: 復習対象ログリスト
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT
                al.id, al.query, al.response, al.judgment,
                fc.importance_score, fc.review_count, fc.retention_rate
            FROM audit_logs al
            JOIN forgetting_curve fc ON al.id = fc.log_id
            WHERE fc.next_review_date <= ?
            ORDER BY fc.importance_score DESC, fc.next_review_date ASC
            LIMIT ?
        """, (datetime.now(), limit))
        
        reviews = []
        for row in cursor.fetchall():
            reviews.append({
                'log_id': row[0],
                'query': row[1],
                'response': row[2],
                'judgment': row[3],
                'importance_score': row[4],
                'review_count': row[5],
                'retention_rate': row[6],
            })
        
        return reviews
    
    def mark_reviewed(self, log_id: int):
        """
        復習完了マーク
        
        次回復習日を更新（間隔を2倍に）
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT review_count, next_review_date
            FROM forgetting_curve
            WHERE log_id = ?
        """, (log_id,))
        
        row = cursor.fetchone()
        if row:
            review_count = row[0] + 1
            current_next_review = datetime.fromisoformat(row[1])
            
            # 次回復習間隔（倍々に増加）
            interval_days = 2 ** review_count
            next_review_date = datetime.now() + timedelta(days=interval_days)
            
            # 保持率低下（80%ずつ）
            retention_rate = 0.8 ** review_count
            
            cursor.execute("""
                UPDATE forgetting_curve
                SET review_count = ?, next_review_date = ?, retention_rate = ?
                WHERE log_id = ?
            """, (review_count, next_review_date, retention_rate, log_id))
            
            self.conn.commit()
            
            logger.info(f"[REVIEW] Marked reviewed: log_id={log_id}, next_review={next_review_date.date()}")
    
    def get_user_stats(self, user_id: str) -> Optional[Dict]:
        """ユーザー統計取得"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT * FROM user_stats WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                'user_id': row[0],
                'total_queries': row[1],
                'allow_count': row[2],
                'escalation_count': row[3],
                'deny_count': row[4],
                'avg_confidence': row[5],
                'first_access': row[6],
                'last_access': row[7],
            }
        
        return None
    
    def close(self):
        """データベース接続を閉じる"""
        if self.conn:
            self.conn.close()
            logger.info("[AUDIT] Database connection closed")

