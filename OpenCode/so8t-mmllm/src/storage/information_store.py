#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情報ストア機能
- ローカルベクトルDB（FAISS/ChromaDB）
- セキュア検索API
- バージョン管理
- 情報漏洩防止
"""

import os
import json
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Document:
    """ドキュメント"""
    doc_id: str
    title: str
    content: str
    domain: str
    classification: str  # 公開/取扱注意/機密/極秘
    version: int
    created_at: str
    updated_at: str
    metadata: Dict


@dataclass
class SearchResult:
    """検索結果"""
    doc_id: str
    title: str
    snippet: str
    score: float
    classification: str


class SimpleVectorStore:
    """
    簡易ベクトルストア（FAISS/ChromaDB代替）
    実際の本番環境では、FAISSまたはChromaDBを使用推奨
    """
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        self.documents: Dict[str, Document] = {}
    
    def add_document(self, doc_id: str, vector: np.ndarray, document: Document):
        """
        ドキュメント追加
        
        Args:
            doc_id: ドキュメントID
            vector: ベクトル表現
            document: ドキュメント
        """
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vector.shape[0]}")
        
        self.vectors[doc_id] = vector / np.linalg.norm(vector)  # 正規化
        self.documents[doc_id] = document
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        類似度検索
        
        Args:
            query_vector: クエリベクトル
            top_k: 上位K件
        
        Returns:
            results: [(doc_id, score), ...]
        """
        if not self.vectors:
            return []
        
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # コサイン類似度計算
        scores = {}
        for doc_id, vector in self.vectors.items():
            similarity = np.dot(query_vector, vector)
            scores[doc_id] = similarity
        
        # スコア順ソート
        sorted_results = sorted(scores.items(), key=lambda x: -x[1])
        
        return sorted_results[:top_k]


class InformationStore:
    """情報ストア"""
    
    def __init__(self, db_path: Path = Path("database/so8t_information_store.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # ベクトルストア
        self.vector_store = SimpleVectorStore(dimension=768)
        
        # アクセス制御
        self.classification_levels = ["公開", "取扱注意", "機密", "極秘"]
    
    def _init_database(self):
        """データベース初期化"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # ドキュメントテーブル
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            domain TEXT NOT NULL,
            classification TEXT NOT NULL,
            version INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT
        )
        """)
        
        # バージョン履歴テーブル
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS version_history (
            history_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            content TEXT NOT NULL,
            updated_by TEXT,
            updated_at TEXT NOT NULL,
            change_description TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
        """)
        
        # アクセスログテーブル
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS access_logs (
            log_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            access_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            allowed BOOLEAN NOT NULL,
            reason TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
        """)
        
        # インデックス
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain ON documents(domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_classification ON documents(classification)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_version ON version_history(doc_id, version)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_timestamp ON access_logs(timestamp)")
        
        conn.commit()
        conn.close()
    
    def add_document(self, document: Document, user_id: str) -> bool:
        """
        ドキュメント追加
        
        Args:
            document: ドキュメント
            user_id: ユーザーID
        
        Returns:
            success: 成功フラグ
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
            INSERT INTO documents 
            (doc_id, title, content, domain, classification, version, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                document.doc_id,
                document.title,
                document.content,
                document.domain,
                document.classification,
                document.version,
                document.created_at,
                document.updated_at,
                json.dumps(document.metadata, ensure_ascii=False)
            ))
            
            # バージョン履歴
            history_id = hashlib.md5(f"{document.doc_id}_{document.version}_{datetime.now()}".encode()).hexdigest()[:16]
            cursor.execute("""
            INSERT INTO version_history 
            (history_id, doc_id, version, content, updated_by, updated_at, change_description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                history_id,
                document.doc_id,
                document.version,
                document.content,
                user_id,
                document.updated_at,
                "Initial version"
            ))
            
            conn.commit()
            
            # ベクトルストアに追加（簡易的なベクトル化）
            vector = self._simple_vectorize(document.content)
            self.vector_store.add_document(document.doc_id, vector, document)
            
            print(f"[OK] Document added: {document.doc_id}")
            return True
        
        except Exception as e:
            print(f"[ERROR] Failed to add document: {e}")
            conn.rollback()
            return False
        
        finally:
            conn.close()
    
    def _simple_vectorize(self, text: str) -> np.ndarray:
        """
        簡易ベクトル化（実際は埋め込みモデル使用推奨）
        
        Args:
            text: テキスト
        
        Returns:
            vector: ベクトル
        """
        # ハッシュベースの簡易ベクトル化
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # 768次元に拡張
        vector = np.zeros(768)
        for i in range(min(len(hash_bytes), 768)):
            vector[i] = hash_bytes[i] / 255.0
        
        return vector
    
    def search_documents(self, 
                         query: str, 
                         user_id: str, 
                         user_clearance: str = "公開",
                         top_k: int = 5) -> List[SearchResult]:
        """
        セキュア検索
        
        Args:
            query: クエリ
            user_id: ユーザーID
            user_clearance: ユーザーのクリアランスレベル
            top_k: 上位K件
        
        Returns:
            results: 検索結果リスト
        """
        # クエリベクトル化
        query_vector = self._simple_vectorize(query)
        
        # ベクトル検索
        vector_results = self.vector_store.search(query_vector, top_k=top_k * 2)
        
        # アクセス制御フィルタリング
        filtered_results = []
        for doc_id, score in vector_results:
            document = self.vector_store.documents.get(doc_id)
            if not document:
                continue
            
            # アクセス権確認
            if self._check_access(user_clearance, document.classification):
                snippet = document.content[:200] + "..." if len(document.content) > 200 else document.content
                
                result = SearchResult(
                    doc_id=document.doc_id,
                    title=document.title,
                    snippet=snippet,
                    score=float(score),
                    classification=document.classification
                )
                
                filtered_results.append(result)
                
                # アクセスログ記録
                self._log_access(doc_id, user_id, "search", True, "")
            else:
                # アクセス拒否ログ
                self._log_access(doc_id, user_id, "search", False, "Insufficient clearance")
            
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
    
    def _check_access(self, user_clearance: str, doc_classification: str) -> bool:
        """
        アクセス権確認
        
        Args:
            user_clearance: ユーザークリアランス
            doc_classification: ドキュメント機密レベル
        
        Returns:
            allowed: アクセス許可フラグ
        """
        user_level = self.classification_levels.index(user_clearance) if user_clearance in self.classification_levels else 0
        doc_level = self.classification_levels.index(doc_classification) if doc_classification in self.classification_levels else 0
        
        return user_level >= doc_level
    
    def _log_access(self, doc_id: str, user_id: str, access_type: str, allowed: bool, reason: str):
        """アクセスログ記録"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        log_id = hashlib.md5(f"{doc_id}_{user_id}_{datetime.now()}".encode()).hexdigest()[:16]
        
        cursor.execute("""
        INSERT INTO access_logs 
        (log_id, doc_id, user_id, access_type, timestamp, allowed, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            log_id,
            doc_id,
            user_id,
            access_type,
            datetime.now().isoformat(),
            allowed,
            reason
        ))
        
        conn.commit()
        conn.close()
    
    def get_document_versions(self, doc_id: str) -> List[Dict]:
        """ドキュメントバージョン履歴取得"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT history_id, version, updated_by, updated_at, change_description
        FROM version_history
        WHERE doc_id = ?
        ORDER BY version DESC
        """, (doc_id,))
        
        versions = []
        for row in cursor.fetchall():
            version = {
                "history_id": row[0],
                "version": row[1],
                "updated_by": row[2],
                "updated_at": row[3],
                "change_description": row[4]
            }
            versions.append(version)
        
        conn.close()
        return versions
    
    def generate_report(self, output_path: Path = None):
        """情報ストアレポート生成"""
        if output_path is None:
            output_path = Path("_docs") / f"{datetime.now().strftime('%Y-%m-%d')}_information_store_report.md"
        
        output_path.parent.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 統計
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        
        cursor.execute("SELECT classification, COUNT(*) FROM documents GROUP BY classification")
        class_breakdown = dict(cursor.fetchall())
        
        cursor.execute("SELECT COUNT(*) FROM access_logs WHERE allowed = 1")
        allowed_access = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM access_logs WHERE allowed = 0")
        denied_access = cursor.fetchone()[0]
        
        conn.close()
        
        report = f"""# 情報ストアレポート

## ストレージ概要
- **レポート日時**: {datetime.now().isoformat()}
- **総ドキュメント数**: {total_docs:,}
- **ベクトル次元**: {self.vector_store.dimension}

## 機密レベル別統計

| 機密レベル | ドキュメント数 |
|-----------|--------------|
"""
        
        for classification in self.classification_levels:
            count = class_breakdown.get(classification, 0)
            report += f"| {classification} | {count:,} |\n"
        
        report += f"""
## アクセス統計
- **許可されたアクセス**: {allowed_access:,}
- **拒否されたアクセス**: {denied_access:,}
- **拒否率**: {(denied_access / max(allowed_access + denied_access, 1) * 100):.1f}%

## セキュリティ機能
- ローカルベクトルDB（情報漏洩防止）
- アクセス制御（クリアランスレベル）
- バージョン管理（完全履歴）
- アクセス監査ログ

## 次のステップ
- [READY] 本番環境配備
- [READY] FAISS/ChromaDB統合（高速化）
- [READY] 埋め込みモデル統合（精度向上）
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[OK] Report saved to {output_path}")


def test_information_store():
    """テスト実行"""
    print("\n[TEST] Information Store Test")
    print("="*60)
    
    store = InformationStore()
    
    # テストドキュメント追加
    doc = Document(
        doc_id="TEST001",
        title="防衛システム概要",
        content="防衛システムの基本的な概要について説明します。公開情報です。",
        domain="defense",
        classification="公開",
        version=1,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        metadata={"author": "test_user"}
    )
    
    store.add_document(doc, user_id="test_user")
    
    # 検索テスト
    results = store.search_documents(
        query="防衛システム",
        user_id="test_user",
        user_clearance="公開"
    )
    
    print(f"\n[SEARCH] Found {len(results)} results")
    for result in results:
        print(f"- {result.title} (score: {result.score:.3f})")
    
    # レポート生成
    store.generate_report()
    
    print("\n[OK] Test completed")


if __name__ == "__main__":
    test_information_store()
