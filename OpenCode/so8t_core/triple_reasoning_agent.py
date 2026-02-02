"""
三重推論エージェント
ALLOW / ESCALATION / DENY 判定システム
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class JudgmentType(Enum):
    """判定タイプ"""
    ALLOW = "ALLOW"  # 応答可能
    ESCALATION = "ESCALATION"  # 人間確認必要
    DENY = "DENY"  # 応答拒否


@dataclass
class JudgmentResult:
    """判定結果"""
    judgment: JudgmentType
    confidence: float  # 信頼度 [0, 1]
    reason: str
    matched_rules: List[str]


class TripleReasoningAgent:
    """
    三重推論エージェント
    
    主要機能:
    - ルールベース判定
    - キーワードマッチング
    - モデル信頼度補正
    - 説明可能性
    """
    
    def __init__(
        self,
        deny_keywords: Optional[List[str]] = None,
        escalation_keywords: Optional[List[str]] = None,
        confidence_threshold_deny: float = 0.7,
        confidence_threshold_escalation: float = 0.5,
    ):
        """
        Args:
            deny_keywords: 拒否キーワードリスト
            escalation_keywords: エスカレーションキーワードリスト
            confidence_threshold_deny: 拒否判定の信頼度閾値
            confidence_threshold_escalation: エスカレーション判定の信頼度閾値
        """
        # デフォルトキーワード
        self.deny_keywords = deny_keywords or [
            '機密', '秘密', '未公開', '内部', '機密情報',
            '秘密情報', 'パスワード', '認証情報', 'API キー',
            'トークン', '秘密鍵', 'プライベートキー',
        ]
        
        self.escalation_keywords = escalation_keywords or [
            '詳細な仕様', '具体的な運用', '専門的な判断',
            '内部構造', '設計', '実装', '運用方法',
            '技術仕様', '詳細仕様', '性能データ',
        ]
        
        self.confidence_threshold_deny = confidence_threshold_deny
        self.confidence_threshold_escalation = confidence_threshold_escalation
    
    def judge(
        self,
        query: str,
        model_confidence: Optional[float] = None,
        context: Optional[Dict] = None,
    ) -> JudgmentResult:
        """
        クエリを判定
        
        Args:
            query: ユーザークエリ
            model_confidence: モデルの信頼度（オプション）
            context: 追加コンテキスト（オプション）
        
        Returns:
            JudgmentResult: 判定結果
        """
        matched_rules = []
        
        # ルール1: 拒否キーワードチェック
        for keyword in self.deny_keywords:
            if keyword in query:
                matched_rules.append(f"DENY_KEYWORD: {keyword}")
        
        if matched_rules:
            return JudgmentResult(
                judgment=JudgmentType.DENY,
                confidence=1.0,
                reason="機密情報に関わる内容が含まれています。",
                matched_rules=matched_rules,
            )
        
        # ルール2: エスカレーションキーワードチェック
        for keyword in self.escalation_keywords:
            if keyword in query:
                matched_rules.append(f"ESCALATION_KEYWORD: {keyword}")
        
        if matched_rules:
            return JudgmentResult(
                judgment=JudgmentType.ESCALATION,
                confidence=0.8,
                reason="専門的な判断が必要な内容です。",
                matched_rules=matched_rules,
            )
        
        # ルール3: モデル信頼度による判定
        if model_confidence is not None:
            if model_confidence < self.confidence_threshold_escalation:
                matched_rules.append(f"LOW_CONFIDENCE: {model_confidence:.2f}")
                return JudgmentResult(
                    judgment=JudgmentType.ESCALATION,
                    confidence=model_confidence,
                    reason="モデルの信頼度が低いため、確認が必要です。",
                    matched_rules=matched_rules,
                )
        
        # ルール4: デフォルトはALLOW
        matched_rules.append("DEFAULT_ALLOW")
        return JudgmentResult(
            judgment=JudgmentType.ALLOW,
            confidence=model_confidence or 0.9,
            reason="一般的な情報であり、応答可能です。",
            matched_rules=matched_rules,
        )
    
    def add_deny_keyword(self, keyword: str):
        """拒否キーワードを追加"""
        if keyword not in self.deny_keywords:
            self.deny_keywords.append(keyword)
            logger.info(f"[AGENT] Added DENY keyword: {keyword}")
    
    def add_escalation_keyword(self, keyword: str):
        """エスカレーションキーワードを追加"""
        if keyword not in self.escalation_keywords:
            self.escalation_keywords.append(keyword)
            logger.info(f"[AGENT] Added ESCALATION keyword: {keyword}")
    
    def remove_deny_keyword(self, keyword: str):
        """拒否キーワードを削除"""
        if keyword in self.deny_keywords:
            self.deny_keywords.remove(keyword)
            logger.info(f"[AGENT] Removed DENY keyword: {keyword}")
    
    def remove_escalation_keyword(self, keyword: str):
        """エスカレーションキーワードを削除"""
        if keyword in self.escalation_keywords:
            self.escalation_keywords.remove(keyword)
            logger.info(f"[AGENT] Removed ESCALATION keyword: {keyword}")
    
    def get_statistics(self, judgments: List[JudgmentResult]) -> Dict:
        """
        判定統計を取得
        
        Args:
            judgments: 判定結果リスト
        
        Returns:
            statistics: 統計情報
        """
        total = len(judgments)
        if total == 0:
            return {}
        
        allow_count = sum(1 for j in judgments if j.judgment == JudgmentType.ALLOW)
        escalation_count = sum(1 for j in judgments if j.judgment == JudgmentType.ESCALATION)
        deny_count = sum(1 for j in judgments if j.judgment == JudgmentType.DENY)
        
        avg_confidence = sum(j.confidence for j in judgments) / total
        
        return {
            'total': total,
            'allow_count': allow_count,
            'escalation_count': escalation_count,
            'deny_count': deny_count,
            'allow_rate': allow_count / total,
            'escalation_rate': escalation_count / total,
            'deny_rate': deny_count / total,
            'avg_confidence': avg_confidence,
        }

