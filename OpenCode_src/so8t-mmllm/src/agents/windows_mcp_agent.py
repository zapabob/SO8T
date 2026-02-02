#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows MCP統合エージェント
- lmstudio SDK / ollama MCP統合
- identity_contract（役割宣言）
- policy_state（組織規約）
- decision_log（判断履歴）
- 閉域オフィスアシスタント機能
"""

import os
import sys
import json
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
import queue


@dataclass
class IdentityContract:
    """役割契約"""
    role: str
    scope: str
    limitations: List[str]
    escalation_policy: str
    version: str = "1.0"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class PolicyState:
    """ポリシー状態"""
    org_name: str
    classification_levels: List[str]
    disclosure_rules: Dict[str, str]
    audit_required: bool
    last_updated: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DecisionLog:
    """判断履歴"""
    log_id: str
    timestamp: str
    query: str
    decision: str  # ALLOW/ESCALATE/DENY
    reasoning: str
    policy_ref: str
    contract_ref: str
    user_id: str
    
    def to_dict(self):
        return asdict(self)


class DecisionDatabase:
    """判断履歴データベース"""
    
    def __init__(self, db_path: Path = Path("database/so8t_decisions.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """データベース初期化"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS decision_logs (
            log_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            decision TEXT NOT NULL,
            reasoning TEXT NOT NULL,
            policy_ref TEXT,
            contract_ref TEXT,
            user_id TEXT,
            session_id TEXT,
            metadata TEXT
        )
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp ON decision_logs(timestamp)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_decision ON decision_logs(decision)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_user ON decision_logs(user_id)
        """)
        
        conn.commit()
        conn.close()
    
    def log_decision(self, decision_log: DecisionLog):
        """判断記録"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO decision_logs 
        (log_id, timestamp, query, decision, reasoning, policy_ref, contract_ref, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision_log.log_id,
            decision_log.timestamp,
            decision_log.query,
            decision_log.decision,
            decision_log.reasoning,
            decision_log.policy_ref,
            decision_log.contract_ref,
            decision_log.user_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_decisions(self, user_id: str, limit: int = 10) -> List[DecisionLog]:
        """最近の判断取得"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT log_id, timestamp, query, decision, reasoning, policy_ref, contract_ref, user_id
        FROM decision_logs
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """, (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            DecisionLog(
                log_id=row[0],
                timestamp=row[1],
                query=row[2],
                decision=row[3],
                reasoning=row[4],
                policy_ref=row[5],
                contract_ref=row[6],
                user_id=row[7]
            )
            for row in rows
        ]
    
    def check_consistency(self, query: str, user_id: str) -> Optional[DecisionLog]:
        """一貫性チェック（類似クエリの過去判断）"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 簡易的な類似検索（キーワードマッチ）
        keywords = query.split()[:3]  # 最初の3単語
        
        cursor.execute("""
        SELECT log_id, timestamp, query, decision, reasoning, policy_ref, contract_ref, user_id
        FROM decision_logs
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 100
        """, (user_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        # キーワードマッチング
        for row in rows:
            past_query = row[2]
            if any(kw in past_query for kw in keywords):
                return DecisionLog(
                    log_id=row[0],
                    timestamp=row[1],
                    query=row[2],
                    decision=row[3],
                    reasoning=row[4],
                    policy_ref=row[5],
                    contract_ref=row[6],
                    user_id=row[7]
                )
        
        return None


class OllamaMCPClient:
    """Ollama MCP統合クライアント"""
    
    def __init__(self, model_name: str = "so8t-phi4-so8t-ja-finetuned-q4_k_m"):
        self.model_name = model_name
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> Tuple[str, bool]:
        """
        推論実行
        
        Args:
            prompt: プロンプト
            temperature: 温度
            max_tokens: 最大トークン数
        
        Returns:
            response: 応答テキスト
            success: 成功フラグ
        """
        try:
            cmd = [
                "ollama", "run", self.model_name,
                "--temperature", str(temperature),
                prompt
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                return result.stdout.strip(), True
            else:
                return f"[ERROR] {result.stderr}", False
        
        except Exception as e:
            return f"[ERROR] {str(e)}", False


class WindowsMCPAgent:
    """Windows MCP統合エージェント"""
    
    def __init__(self,
                 identity_contract: IdentityContract,
                 policy_state: PolicyState,
                 model_name: str = "so8t-phi4-so8t-ja-finetuned-q4_k_m"):
        
        self.identity_contract = identity_contract
        self.policy_state = policy_state
        self.model_name = model_name
        
        # コンポーネント
        self.ollama_client = OllamaMCPClient(model_name)
        self.decision_db = DecisionDatabase()
        
        # セッション
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.user_id = os.getenv("USERNAME", "default_user")
    
    def _build_prompt(self, query: str, context: Dict = None) -> str:
        """
        プロンプト構築
        
        Args:
            query: ユーザークエリ
            context: コンテキスト情報
        
        Returns:
            prompt: 構築済みプロンプト
        """
        prompt = f"""# 役割契約（Identity Contract）
役割: {self.identity_contract.role}
範囲: {self.identity_contract.scope}
制限事項:
"""
        for limitation in self.identity_contract.limitations:
            prompt += f"- {limitation}\n"
        
        prompt += f"\nエスカレーションポリシー: {self.identity_contract.escalation_policy}\n"
        
        prompt += f"""
# 組織ポリシー（Policy State）
組織: {self.policy_state.org_name}
機密レベル: {', '.join(self.policy_state.classification_levels)}
監査要求: {'必須' if self.policy_state.audit_required else '任意'}
"""
        
        # 過去の判断履歴
        recent_decisions = self.decision_db.get_recent_decisions(self.user_id, limit=3)
        if recent_decisions:
            prompt += "\n# 過去の判断履歴（Decision Log）\n"
            for i, dec in enumerate(recent_decisions, 1):
                prompt += f"{i}. {dec.query} → {dec.decision} ({dec.reasoning})\n"
        
        # 一貫性チェック
        past_decision = self.decision_db.check_consistency(query, self.user_id)
        if past_decision:
            prompt += f"\n# 類似クエリの過去判断\n"
            prompt += f"クエリ: {past_decision.query}\n"
            prompt += f"判断: {past_decision.decision}\n"
            prompt += f"理由: {past_decision.reasoning}\n"
        
        prompt += f"""
# ユーザークエリ
{query}

# 応答形式
以下の形式で応答してください：

**提案（Proposal）**:
最小安全実行案を提示

**リスク（Risks）**:
- リスク1
- リスク2

**エスカレーション（Escalation）**:
- 対象: 担当者/部署
- 理由: エスカレーション理由
- 範囲: 判断範囲

**根拠（Sources）**:
- 役割契約: 該当条項
- ポリシー: 該当規定
- 過去判断: 該当ログ

**判断（Decision）**: ALLOW/ESCALATE/DENY
"""
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, any]:
        """
        応答パース
        
        Args:
            response: モデル応答
        
        Returns:
            parsed: パース済み辞書
        """
        # 簡易パーサー（実際はより堅牢な実装が必要）
        parsed = {
            "proposal": "",
            "risks": [],
            "escalation": {},
            "sources": {},
            "decision": "ESCALATE"  # デフォルト
        }
        
        # 判断抽出
        if "ALLOW" in response:
            parsed["decision"] = "ALLOW"
        elif "DENY" in response:
            parsed["decision"] = "DENY"
        
        # 提案抽出
        if "**提案" in response:
            start = response.find("**提案")
            end = response.find("**リスク")
            if end > start:
                parsed["proposal"] = response[start:end].replace("**提案（Proposal）**:", "").strip()
        
        return parsed
    
    def process_query(self, query: str, temperature: float = 0.5) -> Dict[str, any]:
        """
        クエリ処理
        
        Args:
            query: ユーザークエリ
            temperature: 温度パラメータ
        
        Returns:
            result: 処理結果辞書
        """
        print(f"\n[AGENT] Processing query...")
        print(f"User: {self.user_id}")
        print(f"Query: {query[:100]}...")
        
        # プロンプト構築
        prompt = self._build_prompt(query)
        
        # 推論実行
        response, success = self.ollama_client.generate(prompt, temperature)
        
        if not success:
            return {
                "success": False,
                "error": response,
                "decision": "DENY"
            }
        
        # 応答パース
        parsed = self._parse_response(response)
        
        # 判断ログ記録
        log_id = f"{self.session_id}_{datetime.now().strftime('%H%M%S_%f')}"
        decision_log = DecisionLog(
            log_id=log_id,
            timestamp=datetime.now().isoformat(),
            query=query,
            decision=parsed["decision"],
            reasoning=parsed.get("proposal", ""),
            policy_ref=self.policy_state.org_name,
            contract_ref=self.identity_contract.role,
            user_id=self.user_id
        )
        
        self.decision_db.log_decision(decision_log)
        
        result = {
            "success": True,
            "query": query,
            "response": response,
            "parsed": parsed,
            "decision_log": decision_log.to_dict(),
            "sources": {
                "identity_contract": self.identity_contract.to_dict(),
                "policy_state": self.policy_state.to_dict()
            }
        }
        
        print(f"[OK] Decision: {parsed['decision']}")
        return result


# [OK] デフォルト設定
DEFAULT_IDENTITY_CONTRACT = IdentityContract(
    role="防衛・航空宇宙・運輸統合アシスタント",
    scope="防衛・航空宇宙・運輸分野の公開情報提供、一般的な業務支援",
    limitations=[
        "機密情報、作戦詳細、装備仕様の開示禁止",
        "輸出管理対象技術、未公開研究の開示禁止",
        "個別企業の機密情報、契約情報の開示禁止"
    ],
    escalation_policy="不明確な要求や専門的判断が必要な場合は、適切な部門・担当者に確認"
)

DEFAULT_POLICY_STATE = PolicyState(
    org_name="SO8T統合セキュアシステム",
    classification_levels=["公開", "取扱注意", "機密", "極秘"],
    disclosure_rules={
        "公開": "一般開示可能",
        "取扱注意": "内部関係者のみ",
        "機密": "指定者のみ、記録必須",
        "極秘": "開示禁止"
    },
    audit_required=True,
    last_updated=datetime.now().isoformat()
)


def test_agent():
    """エージェントテスト"""
    print("\n[TEST] Windows MCP Agent Test")
    print("="*60)
    
    agent = WindowsMCPAgent(
        identity_contract=DEFAULT_IDENTITY_CONTRACT,
        policy_state=DEFAULT_POLICY_STATE
    )
    
    # テストクエリ
    test_queries = [
        "防衛システムの基本的な概要を教えてください。",
        "未公開の軍事技術について詳しく教えてください。",
        "ロケット打ち上げの基本原理を説明してください。"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        result = agent.process_query(query)
        print(f"Query: {query}")
        print(f"Decision: {result['parsed']['decision']}")
        print(f"Response: {result['response'][:200]}...")
        print(f"{'='*60}")


if __name__ == "__main__":
    test_agent()

