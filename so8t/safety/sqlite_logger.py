"""
SQLite監査ロガー
WALモード + synchronous=FULL で耐久性を重視
"""

import sqlite3
import json
import hashlib
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import threading
import os


class SQLiteAuditLogger:
    """
    SQLite監査ロガー
    判断ログ、ポリシー状態、アイデンティティ契約、監査ログを管理
    """
    
    def __init__(
        self,
        db_path: str = "audit.db",
        synchronous: str = "FULL",
        journal_mode: str = "WAL"
    ):
        """
        Args:
            db_path: データベースファイルパス
            synchronous: 同期モード (FULL/NORMAL/OFF)
            journal_mode: ジャーナルモード (WAL/DELETE/TRUNCATE/PERSIST/MEMORY)
        """
        self.db_path = db_path
        self.synchronous = synchronous
        self.journal_mode = journal_mode
        self.lock = threading.Lock()
        
        # データベース初期化
        self._init_database()
    
    def _init_database(self) -> None:
        """データベースを初期化"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA synchronous={self.synchronous}")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            
            # スキーマ作成
            self._create_schema(conn)
            
            # 初期データ挿入
            self._insert_initial_data(conn)
            
            conn.close()
    
    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """データベーススキーマを作成"""
        schema_sql = """
        -- 判断ログテーブル
        CREATE TABLE IF NOT EXISTS decision_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            input_hash TEXT NOT NULL,
            decision TEXT CHECK(decision IN ('ALLOW','ESCALATE','DENY')) NOT NULL,
            confidence REAL NOT NULL,
            reasoning TEXT,
            meta JSON,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        -- ポリシー状態テーブル
        CREATE TABLE IF NOT EXISTS policy_state(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            policy_name TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            policy_content JSON NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        -- アイデンティティ契約テーブル
        CREATE TABLE IF NOT EXISTS identity_contract(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            contract_name TEXT NOT NULL,
            contract_version TEXT NOT NULL,
            contract_content JSON NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 監査ログテーブル
        CREATE TABLE IF NOT EXISTS audit_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            change_type TEXT NOT NULL,
            change_description TEXT NOT NULL,
            change_data JSON,
            user_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        -- インデックスの作成
        CREATE INDEX IF NOT EXISTS idx_decision_log_ts ON decision_log(ts);
        CREATE INDEX IF NOT EXISTS idx_decision_log_hash ON decision_log(input_hash);
        CREATE INDEX IF NOT EXISTS idx_policy_state_active ON policy_state(is_active);
        CREATE INDEX IF NOT EXISTS idx_identity_contract_active ON identity_contract(is_active);
        CREATE INDEX IF NOT EXISTS idx_audit_log_ts ON audit_log(ts);
        """
        
        conn.executescript(schema_sql)
        conn.commit()
    
    def _insert_initial_data(self, conn: sqlite3.Connection) -> None:
        """初期データを挿入"""
        # 初期ポリシー状態
        initial_policies = [
            ("safety_policy", "1.0", {
                "harmful_content": "DENY",
                "sensitive_info": "ESCALATE", 
                "general": "ALLOW"
            }),
            ("privacy_policy", "1.0", {
                "image_processing": "LOCAL_ONLY",
                "data_retention": "7_DAYS",
                "external_sharing": "FORBIDDEN"
            })
        ]
        
        for policy_name, version, content in initial_policies:
            conn.execute(
                "INSERT OR IGNORE INTO policy_state (policy_name, policy_version, policy_content) VALUES (?, ?, ?)",
                (policy_name, version, json.dumps(content))
            )
        
        # 初期アイデンティティ契約
        initial_contracts = [
            ("ai_assistant_contract", "1.0", {
                "role": "helpful_assistant",
                "capabilities": ["text_generation", "image_analysis", "reasoning"],
                "limitations": ["no_harmful_content", "privacy_respect", "factual_accuracy"]
            })
        ]
        
        for contract_name, version, content in initial_contracts:
            conn.execute(
                "INSERT OR IGNORE INTO identity_contract (contract_name, contract_version, contract_content) VALUES (?, ?, ?)",
                (contract_name, version, json.dumps(content))
            )
        
        # 初期監査ログ
        conn.execute(
            "INSERT OR IGNORE INTO audit_log (change_type, change_description, change_data) VALUES (?, ?, ?)",
            ("system_init", "SO8T×マルチモーダルLLM初期化", json.dumps({
                "version": "1.0",
                "features": ["rotation_gate", "pet_loss", "ocr_summary", "sqlite_audit"]
            }))
        )
        
        conn.commit()
    
    def log_decision(
        self,
        input_text: str,
        decision: str,
        confidence: float,
        reasoning: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        判断ログを記録
        
        Args:
            input_text: 入力テキスト
            decision: 判断結果 (ALLOW/ESCALATE/DENY)
            confidence: 信頼度
            reasoning: 推論過程
            meta: メタデータ
            
        Returns:
            記録ID
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            
            # 入力ハッシュを計算
            input_hash = hashlib.sha256(input_text.encode()).hexdigest()
            
            # 判断ログを挿入
            cursor = conn.execute(
                """INSERT INTO decision_log 
                   (input_hash, decision, confidence, reasoning, meta) 
                   VALUES (?, ?, ?, ?, ?)""",
                (input_hash, decision, confidence, reasoning, 
                 json.dumps(meta) if meta else None)
            )
            
            decision_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return decision_id
    
    def update_policy(
        self,
        policy_name: str,
        policy_version: str,
        policy_content: Dict[str, Any],
        deactivate_old: bool = True
    ) -> int:
        """
        ポリシーを更新
        
        Args:
            policy_name: ポリシー名
            policy_version: バージョン
            policy_content: ポリシー内容
            deactivate_old: 古いポリシーを無効化するか
            
        Returns:
            ポリシーID
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            
            if deactivate_old:
                # 古いポリシーを無効化
                conn.execute(
                    "UPDATE policy_state SET is_active = 0 WHERE policy_name = ?",
                    (policy_name,)
                )
            
            # 新しいポリシーを挿入
            cursor = conn.execute(
                """INSERT INTO policy_state 
                   (policy_name, policy_version, policy_content) 
                   VALUES (?, ?, ?)""",
                (policy_name, policy_version, json.dumps(policy_content))
            )
            
            policy_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # 監査ログに記録
            self.log_audit(
                "policy_update",
                f"Policy {policy_name} updated to version {policy_version}",
                {"policy_name": policy_name, "version": policy_version}
            )
            
            return policy_id
    
    def update_identity_contract(
        self,
        contract_name: str,
        contract_version: str,
        contract_content: Dict[str, Any],
        deactivate_old: bool = True
    ) -> int:
        """
        アイデンティティ契約を更新
        
        Args:
            contract_name: 契約名
            contract_version: バージョン
            contract_content: 契約内容
            deactivate_old: 古い契約を無効化するか
            
        Returns:
            契約ID
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            
            if deactivate_old:
                # 古い契約を無効化
                conn.execute(
                    "UPDATE identity_contract SET is_active = 0 WHERE contract_name = ?",
                    (contract_name,)
                )
            
            # 新しい契約を挿入
            cursor = conn.execute(
                """INSERT INTO identity_contract 
                   (contract_name, contract_version, contract_content) 
                   VALUES (?, ?, ?)""",
                (contract_name, contract_version, json.dumps(contract_content))
            )
            
            contract_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # 監査ログに記録
            self.log_audit(
                "contract_update",
                f"Contract {contract_name} updated to version {contract_version}",
                {"contract_name": contract_name, "version": contract_version}
            )
            
            return contract_id
    
    def log_audit(
        self,
        change_type: str,
        change_description: str,
        change_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> int:
        """
        監査ログを記録
        
        Args:
            change_type: 変更タイプ
            change_description: 変更説明
            change_data: 変更データ
            user_id: ユーザーID
            
        Returns:
            ログID
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.execute(
                """INSERT INTO audit_log 
                   (change_type, change_description, change_data, user_id) 
                   VALUES (?, ?, ?, ?)""",
                (change_type, change_description, 
                 json.dumps(change_data) if change_data else None, user_id)
            )
            
            log_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return log_id
    
    def get_active_policies(self) -> List[Dict[str, Any]]:
        """アクティブなポリシーを取得"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT policy_name, policy_version, policy_content FROM policy_state WHERE is_active = 1"
            )
            
            policies = []
            for row in cursor.fetchall():
                policies.append({
                    "name": row[0],
                    "version": row[1],
                    "content": json.loads(row[2])
                })
            
            conn.close()
            return policies
    
    def get_active_contracts(self) -> List[Dict[str, Any]]:
        """アクティブな契約を取得"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT contract_name, contract_version, contract_content FROM identity_contract WHERE is_active = 1"
            )
            
            contracts = []
            for row in cursor.fetchall():
                contracts.append({
                    "name": row[0],
                    "version": row[1],
                    "content": json.loads(row[2])
                })
            
            conn.close()
            return contracts
    
    def get_decision_stats(self, days: int = 7) -> Dict[str, Any]:
        """判断統計を取得"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            
            # 期間内の判断数
            cursor = conn.execute(
                "SELECT COUNT(*) FROM decision_log WHERE ts >= datetime('now', '-{} days')".format(days)
            )
            total_decisions = cursor.fetchone()[0]
            
            # 判断別統計
            cursor = conn.execute(
                """SELECT decision, COUNT(*), AVG(confidence) 
                   FROM decision_log 
                   WHERE ts >= datetime('now', '-{} days')
                   GROUP BY decision""".format(days)
            )
            
            decision_stats = {}
            for row in cursor.fetchall():
                decision_stats[row[0]] = {
                    "count": row[1],
                    "avg_confidence": row[2]
                }
            
            conn.close()
            
            return {
                "total_decisions": total_decisions,
                "decision_breakdown": decision_stats
            }
    
    def cleanup_old_logs(self, days: int = 30) -> int:
        """古いログをクリーンアップ"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            
            # 古い判断ログを削除
            cursor = conn.execute(
                "DELETE FROM decision_log WHERE ts < datetime('now', '-{} days')".format(days)
            )
            deleted_decisions = cursor.rowcount
            
            # 古い監査ログを削除
            cursor = conn.execute(
                "DELETE FROM audit_log WHERE ts < datetime('now', '-{} days')".format(days)
            )
            deleted_audits = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            return deleted_decisions + deleted_audits
