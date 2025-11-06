"""
三重推論エージェント（ALLOW/ESCALATION/DENY）
防衛・航空宇宙・運輸向けセキュアLLMOps用判定システム

判定基準:
- ALLOW: 安全かつ適切な応答が可能
- ESCALATION: 人間の確認が必要
- DENY: 危険・不適切な要求で応答不可

本番環境要件:
- リアルタイム判定（低レイテンシ）
- 監査ログ統合
- 説明可能性
- 設定可能な閾値

Author: SO8T Project Team
Date: 2024-11-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class Judgment(Enum):
    """判定結果"""
    ALLOW = "ALLOW"
    ESCALATION = "ESCALATION"
    DENY = "DENY"


@dataclass
class ReasoningResult:
    """推論結果"""
    judgment: Judgment
    confidence: float
    reason: str
    evidence: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'judgment': self.judgment.value,
            'confidence': self.confidence,
            'reason': self.reason,
            'evidence': self.evidence,
            'timestamp': self.timestamp,
        }


class TripleReasoningAgent:
    """
    三重推論エージェント
    
    入力クエリに対して安全性・適切性を判定し、
    ALLOW/ESCALATION/DENYの3段階で応答方針を決定する。
    """
    
    def __init__(
        self,
        allow_threshold: float = 0.8,
        deny_threshold: float = 0.9,
        escalation_default: bool = True
    ):
        """
        Args:
            allow_threshold: ALLOW判定の信頼度閾値
            deny_threshold: DENY判定の信頼度閾値
            escalation_default: デフォルトでESCALATION
        """
        self.allow_threshold = allow_threshold
        self.deny_threshold = deny_threshold
        self.escalation_default = escalation_default
        
        # 危険キーワード
        self.deny_keywords = [
            '機密', '秘密', 'パスワード', '認証情報',
            '脆弱性', '攻撃', 'ハッキング', 'バイパス',
            '個人情報', 'プライバシー', '位置情報'
        ]
        
        # ESCALATION キーワード
        self.escalation_keywords = [
            '新型', '試験中', '開発中', '未公開',
            '評価', '詳細仕様', 'プロトコル詳細'
        ]
        
        # ALLOW キーワード
        self.allow_keywords = [
            '一般', '公開', '基本', '概要', '教育',
            '標準', '公式', 'ドキュメント'
        ]
        
        logger.info(f"[TripleReasoning] Initialized with "
                   f"allow_threshold={allow_threshold}, "
                   f"deny_threshold={deny_threshold}")
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        クエリを分析
        
        Args:
            query: 入力クエリ
            
        Returns:
            分析結果
        """
        analysis = {
            'has_deny_keywords': False,
            'has_escalation_keywords': False,
            'has_allow_keywords': False,
            'deny_matches': [],
            'escalation_matches': [],
            'allow_matches': [],
            'length': len(query),
            'question_marks': query.count('?') + query.count('？'),
        }
        
        query_lower = query.lower()
        
        # キーワードマッチング
        for kw in self.deny_keywords:
            if kw in query_lower:
                analysis['has_deny_keywords'] = True
                analysis['deny_matches'].append(kw)
        
        for kw in self.escalation_keywords:
            if kw in query_lower:
                analysis['has_escalation_keywords'] = True
                analysis['escalation_matches'].append(kw)
        
        for kw in self.allow_keywords:
            if kw in query_lower:
                analysis['has_allow_keywords'] = True
                analysis['allow_matches'].append(kw)
        
        return analysis
    
    def compute_judgment(
        self,
        query: str,
        logits: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None
    ) -> ReasoningResult:
        """
        判定を計算
        
        Args:
            query: 入力クエリ
            logits: モデル出力ロジット（オプション）
            hidden_states: 隠れ状態（オプション）
            
        Returns:
            ReasoningResult
        """
        # クエリ分析
        analysis = self.analyze_query(query)
        
        # ルールベース判定
        if analysis['has_deny_keywords']:
            judgment = Judgment.DENY
            confidence = 0.95
            reason = f"危険なキーワードを検出: {', '.join(analysis['deny_matches'])}"
            evidence = {
                'rule': 'deny_keywords',
                'matches': analysis['deny_matches']
            }
        
        elif analysis['has_escalation_keywords']:
            judgment = Judgment.ESCALATION
            confidence = 0.85
            reason = f"人間の確認が必要: {', '.join(analysis['escalation_matches'])}"
            evidence = {
                'rule': 'escalation_keywords',
                'matches': analysis['escalation_matches']
            }
        
        elif analysis['has_allow_keywords']:
            judgment = Judgment.ALLOW
            confidence = 0.90
            reason = "一般的な情報として応答可能"
            evidence = {
                'rule': 'allow_keywords',
                'matches': analysis['allow_matches']
            }
        
        else:
            # キーワードマッチしない場合
            if self.escalation_default:
                judgment = Judgment.ESCALATION
                confidence = 0.70
                reason = "明確な判定基準がないため人間の確認を推奨"
                evidence = {'rule': 'default_escalation'}
            else:
                judgment = Judgment.ALLOW
                confidence = 0.75
                reason = "特に問題が検出されなかったため応答可能"
                evidence = {'rule': 'default_allow'}
        
        # モデルベース補正（ロジットがあれば）
        if logits is not None:
            model_confidence = self._compute_model_confidence(logits)
            evidence['model_confidence'] = model_confidence
            
            # 低信頼度の場合はESCALATIONに変更
            if model_confidence < 0.5 and judgment == Judgment.ALLOW:
                judgment = Judgment.ESCALATION
                reason += " (モデル信頼度が低いため)"
                confidence = min(confidence, 0.75)
        
        return ReasoningResult(
            judgment=judgment,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            timestamp=datetime.now().isoformat()
        )
    
    def _compute_model_confidence(self, logits: torch.Tensor) -> float:
        """
        モデル出力から信頼度を計算
        
        Args:
            logits: ロジット [B, T, V]
            
        Returns:
            信頼度 (0.0-1.0)
        """
        # 最大確率の平均を信頼度とする
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)
        confidence = max_probs.mean().item()
        return confidence
    
    def should_respond(self, result: ReasoningResult) -> bool:
        """
        応答すべきか判定
        
        Args:
            result: 推論結果
            
        Returns:
            True if should respond
        """
        return result.judgment == Judgment.ALLOW


# テスト用
if __name__ == "__main__":
    print("=" * 80)
    print("Triple Reasoning Agent Test")
    print("=" * 80)
    
    agent = TripleReasoningAgent()
    
    test_queries = [
        "防衛システムの一般的な概要を教えてください。",
        "特定施設の詳細な配置図を教えてください。",
        "新型システムの性能評価について教えてください。",
        "公開されている技術標準について説明してください。",
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n[Test {i+1}] Query: {query}")
        result = agent.compute_judgment(query)
        print(f"  Judgment: {result.judgment.value}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Reason: {result.reason}")
        print(f"  Evidence: {result.evidence}")
        print(f"  Should respond: {agent.should_respond(result)}")
    
    print("\n" + "=" * 80)

