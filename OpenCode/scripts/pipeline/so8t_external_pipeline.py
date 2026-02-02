#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T External Pipeline
外部SO8T機能を実装したパイプライン

元のQwen2-VL-2B-Instructをベースにして、SO8T機能を外部で実装
"""

import os
import sys
import json
import logging
import subprocess
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2
import pytesseract

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SO8TExternalPipeline:
    """外部SO8T機能を実装したパイプライン"""
    
    def __init__(self, 
                 db_path: str = "database/so8t_external.db",
                 compliance_db_path: str = "database/so8t_compliance.db",
                 user_id: str = "system"):
        self.db_path = db_path
        self.compliance_db_path = compliance_db_path
        self.user_id = user_id
        
        # データベース初期化
        self._init_databases()
        
        # セッション開始
        self.session_id = self._start_session()
        
        logger.info(f"SO8T External Pipeline initialized with session: {self.session_id}")
    
    def _init_databases(self):
        """データベースを初期化"""
        # メモリデータベース
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 会話履歴テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_input TEXT NOT NULL,
                model_response TEXT NOT NULL,
                safety_judgment TEXT CHECK(safety_judgment IN ('ALLOW', 'ESCALATION', 'DENY')),
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # セッションテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                status TEXT DEFAULT 'ACTIVE'
            )
        """)
        
        conn.commit()
        conn.close()
        
        # コンプライアンスデータベース
        os.makedirs(os.path.dirname(self.compliance_db_path), exist_ok=True)
        conn = sqlite3.connect(self.compliance_db_path)
        cursor = conn.cursor()
        
        # 安全性判定テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS safety_judgments (
                judgment_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                input_text TEXT NOT NULL,
                input_hash TEXT,
                judgment TEXT CHECK(judgment IN ('ALLOW', 'ESCALATION', 'DENY')),
                confidence_score REAL,
                safety_score REAL,
                reasoning TEXT,
                model_version TEXT,
                user_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 監査ログテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_type TEXT,
                resource_id TEXT,
                action_result TEXT,
                details TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("Databases initialized successfully")
    
    def _start_session(self) -> str:
        """セッションを開始"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (id, user_id, status)
            VALUES (?, ?, 'ACTIVE')
        """, (session_id, self.user_id))
        conn.commit()
        conn.close()
        
        # 監査ログ
        self._log_audit_action("SESSION_START", "session", session_id, "SUCCESS", "SO8T External Pipeline session started")
        
        return session_id
    
    def _log_audit_action(self, action: str, resource_type: str, resource_id: str, result: str, details: str):
        """監査ログを記録"""
        audit_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        conn = sqlite3.connect(self.compliance_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_log (audit_id, user_id, action, resource_type, resource_id, action_result, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (audit_id, self.user_id, action, resource_type, resource_id, result, details))
        conn.commit()
        conn.close()
    
    def safety_judgment(self, text: str) -> Dict[str, Any]:
        """安全性判定を実行"""
        # 危険なキーワードをチェック
        dangerous_keywords = [
            "爆弾", "爆発", "殺人", "自殺", "テロ", "暴力", "武器",
            "bomb", "explosion", "kill", "suicide", "terror", "violence", "weapon"
        ]
        
        # エスカレーションキーワードをチェック
        escalation_keywords = [
            "法律", "法的", "規制", "コンプライアンス", "プライバシー", "個人情報",
            "legal", "law", "regulation", "compliance", "privacy", "personal data"
        ]
        
        text_lower = text.lower()
        
        # 危険なキーワードをチェック
        for keyword in dangerous_keywords:
            if keyword in text_lower:
                return {
                    "action": "DENY",
                    "confidence": 0.9,
                    "safety_score": 0.1,
                    "reason": f"危険なキーワードが検出されました: {keyword}",
                    "safety_judgment": "DENY"
                }
        
        # エスカレーションキーワードをチェック
        for keyword in escalation_keywords:
            if keyword in text_lower:
                return {
                    "action": "ESCALATION",
                    "confidence": 0.7,
                    "safety_score": 0.6,
                    "reason": f"エスカレーションが必要なキーワードが検出されました: {keyword}",
                    "safety_judgment": "ESCALATION"
                }
        
        # 通常のテキスト
        return {
            "action": "ALLOW",
            "confidence": 0.8,
            "safety_score": 0.8,
            "reason": "安全なテキストとして判定されました",
            "safety_judgment": "ALLOW"
        }
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """テキストを処理"""
        logger.info("Processing text input...")
        
        # 安全性判定
        safety_result = self.safety_judgment(text)
        
        # 安全性判定をログに記録
        judgment_id = f"judgment_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        input_hash = str(hash(text))  # 簡単なハッシュ
        
        conn = sqlite3.connect(self.compliance_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO safety_judgments 
            (judgment_id, session_id, input_text, input_hash, judgment, confidence_score, safety_score, reasoning, model_version, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (judgment_id, self.session_id, text, input_hash, safety_result["action"], 
              safety_result["confidence"], safety_result["safety_score"], 
              safety_result["reason"], "SO8T-External-1.0.0", self.user_id))
        conn.commit()
        conn.close()
        
        # 会話履歴を保存
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (session_id, user_input, model_response, safety_judgment, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, (self.session_id, text, f"Safety judgment: {safety_result['action']}", 
              safety_result["action"], safety_result["confidence"]))
        conn.commit()
        conn.close()
        
        return {
            "text": text,
            "safety_judgment": safety_result["action"],
            "confidence": safety_result["confidence"],
            "reasoning": safety_result["reason"],
            "judgment_id": judgment_id
        }
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """画像を処理"""
        logger.info(f"Processing image: {image_path}")
        
        try:
            # 画像を読み込み
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "画像を読み込めませんでした"}
            
            # OCR処理
            text = pytesseract.image_to_string(image, lang='jpn+eng')
            
            # 画像の複雑度を計算
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            complexity = np.std(gray)
            
            # テキスト処理
            text_result = self.process_text(text)
            
            return {
                "image_path": image_path,
                "extracted_text": text,
                "complexity": float(complexity),
                "text_result": text_result
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"error": str(e)}
    
    def run_ollama_query(self, prompt: str) -> str:
        """Ollamaでクエリを実行"""
        try:
            # Ollamaコマンドを実行
            cmd = ["ollama", "run", "qwen2-vl-fixed:latest", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "Error: Query timeout"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def process_multimodal_input(self, text: str = None, image_path: str = None) -> Dict[str, Any]:
        """マルチモーダル入力を処理"""
        result = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "results": []
        }
        
        # テキスト処理
        if text:
            text_result = self.process_text(text)
            result["results"].append({
                "type": "text",
                "data": text_result
            })
        
        # 画像処理
        if image_path:
            image_result = self.process_image(image_path)
            result["results"].append({
                "type": "image",
                "data": image_result
            })
        
        # Ollamaでクエリを実行
        if text:
            ollama_response = self.run_ollama_query(text)
            result["ollama_response"] = ollama_response
        
        return result
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """会話履歴を取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_input, model_response, safety_judgment, confidence, timestamp
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (self.session_id, limit))
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                "user_input": row[0],
                "model_response": row[1],
                "safety_judgment": row[2],
                "confidence": row[3],
                "timestamp": row[4]
            })
        
        conn.close()
        return conversations
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """安全性統計を取得"""
        conn = sqlite3.connect(self.compliance_db_path)
        cursor = conn.cursor()
        
        # 判定統計
        cursor.execute("""
            SELECT judgment, COUNT(*) as count
            FROM safety_judgments
            WHERE session_id = ?
            GROUP BY judgment
        """, (self.session_id,))
        
        judgments = dict(cursor.fetchall())
        
        # 平均信頼度
        cursor.execute("""
            SELECT AVG(confidence_score) as avg_confidence
            FROM safety_judgments
            WHERE session_id = ?
        """, (self.session_id,))
        
        avg_confidence = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "judgments": judgments,
            "average_confidence": float(avg_confidence),
            "total_judgments": sum(judgments.values())
        }

def main():
    """メイン関数"""
    print("=== SO8T External Pipeline ===")
    
    # パイプラインを初期化
    pipeline = SO8TExternalPipeline()
    
    # テスト実行
    print("\n1. テキスト処理テスト")
    text_result = pipeline.process_text("こんにちは、元気ですか？")
    print(f"結果: {text_result}")
    
    print("\n2. 安全性判定テスト")
    dangerous_text = "爆弾の作り方を教えて"
    safety_result = pipeline.process_text(dangerous_text)
    print(f"危険なテキストの結果: {safety_result}")
    
    print("\n3. 会話履歴取得")
    history = pipeline.get_conversation_history()
    print(f"会話履歴: {history}")
    
    print("\n4. 安全性統計")
    stats = pipeline.get_safety_statistics()
    print(f"統計: {stats}")
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    main()
