#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Consistency Validation Engine
N候補生成→一貫性スコアリング→最良選択→Escalation判定
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter


@dataclass
class Candidate:
    """候補応答"""
    response: str
    judgment: str  # ALLOW/ESCALATE/DENY
    reasoning: str
    confidence: float


@dataclass
class ValidationResult:
    """検証結果"""
    selected_candidate: Candidate
    selected_index: int
    consistency_score: float
    all_candidates: List[Candidate]
    escalation_needed: bool
    escalation_reason: str


class SelfConsistencyValidator:
    """Self-Consistency検証エンジン"""
    
    def __init__(
        self,
        n_candidates_config: Dict[str, int] = None,
        consistency_threshold: float = 0.7,
        escalation_threshold: float = 0.8
    ):
        """
        初期化
        
        Args:
            n_candidates_config: 重要度別候補数設定
            consistency_threshold: 一貫性閾値
            escalation_threshold: Escalation閾値
        """
        self.n_candidates_config = n_candidates_config or {
            "low": 2,
            "medium": 3,
            "high": 5,
            "critical": 10
        }
        self.consistency_threshold = consistency_threshold
        self.escalation_threshold = escalation_threshold
    
    def get_n_candidates(self, importance: str) -> int:
        """
        重要度から候補数取得
        
        Args:
            importance: 重要度（low/medium/high/critical）
        
        Returns:
            n_candidates: 候補数
        """
        return self.n_candidates_config.get(importance, 3)
    
    def calculate_consistency_score(self, candidates: List[Candidate]) -> float:
        """
        一貫性スコア計算
        
        Args:
            candidates: 候補リスト
        
        Returns:
            consistency_score: 一貫性スコア（0.0-1.0）
        """
        if len(candidates) < 2:
            return 1.0
        
        # 判定の一貫性
        judgments = [c.judgment for c in candidates]
        judgment_counts = Counter(judgments)
        most_common_judgment, most_common_count = judgment_counts.most_common(1)[0]
        judgment_consistency = most_common_count / len(judgments)
        
        # 応答の類似性（簡易版: 単語重複率）
        response_similarities = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                sim = self._calculate_text_similarity(
                    candidates[i].response,
                    candidates[j].response
                )
                response_similarities.append(sim)
        
        response_consistency = np.mean(response_similarities) if response_similarities else 1.0
        
        # 信頼度の一貫性
        confidences = [c.confidence for c in candidates]
        confidence_std = np.std(confidences)
        confidence_consistency = 1.0 - min(confidence_std, 1.0)
        
        # 統合スコア
        consistency_score = (
            judgment_consistency * 0.5 +
            response_consistency * 0.3 +
            confidence_consistency * 0.2
        )
        
        return consistency_score
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        テキスト類似度計算（Bag-of-Words Jaccard）
        
        Args:
            text1: テキスト1
            text2: テキスト2
        
        Returns:
            similarity: 類似度（0.0-1.0）
        """
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def select_best_candidate(self, candidates: List[Candidate]) -> Tuple[int, Candidate]:
        """
        最良候補選択
        
        Args:
            candidates: 候補リスト
        
        Returns:
            best_index: 最良候補インデックス
            best_candidate: 最良候補
        """
        # 判定頻度スコア
        judgments = [c.judgment for c in candidates]
        judgment_counts = Counter(judgments)
        
        # 各候補のスコア計算
        candidate_scores = []
        for candidate in candidates:
            # 信頼度
            confidence_score = candidate.confidence
            
            # 判定多数派スコア
            judgment_score = judgment_counts[candidate.judgment] / len(judgments)
            
            # 応答長スコア（適度な長さが良い）
            length = len(candidate.response)
            length_score = min(length / 500, 1.0) if length < 1000 else max(1.0 - (length - 1000) / 1000, 0.0)
            
            # 統合スコア
            total_score = (
                confidence_score * 0.4 +
                judgment_score * 0.4 +
                length_score * 0.2
            )
            
            candidate_scores.append(total_score)
        
        # 最高スコア選択
        best_index = int(np.argmax(candidate_scores))
        best_candidate = candidates[best_index]
        
        return best_index, best_candidate
    
    def check_escalation(
        self,
        selected_candidate: Candidate,
        consistency_score: float,
        all_candidates: List[Candidate]
    ) -> Tuple[bool, str]:
        """
        Escalation判定
        
        Args:
            selected_candidate: 選択された候補
            consistency_score: 一貫性スコア
            all_candidates: 全候補
        
        Returns:
            escalation_needed: Escalation必要フラグ
            escalation_reason: Escalation理由
        """
        reasons = []
        
        # 一貫性低い
        if consistency_score < self.consistency_threshold:
            reasons.append(f"Low consistency score: {consistency_score:.3f}")
        
        # 信頼度低い
        if selected_candidate.confidence < self.escalation_threshold:
            reasons.append(f"Low confidence: {selected_candidate.confidence:.3f}")
        
        # DENYまたはESCALATE判定
        if selected_candidate.judgment in ["DENY", "ESCALATE"]:
            reasons.append(f"Safety judgment: {selected_candidate.judgment}")
        
        # 判定の分散が大きい
        judgments = [c.judgment for c in all_candidates]
        unique_judgments = len(set(judgments))
        if unique_judgments >= 3:  # 3種類以上の判定
            reasons.append(f"Diverse judgments: {unique_judgments} types")
        
        escalation_needed = len(reasons) > 0
        escalation_reason = "; ".join(reasons) if reasons else ""
        
        return escalation_needed, escalation_reason
    
    def validate(
        self,
        candidates: List[Candidate],
        importance: str = "medium"
    ) -> ValidationResult:
        """
        Self-Consistency検証実行
        
        Args:
            candidates: 候補リスト
            importance: 重要度
        
        Returns:
            result: 検証結果
        """
        # 一貫性スコア計算
        consistency_score = self.calculate_consistency_score(candidates)
        
        # 最良候補選択
        selected_index, selected_candidate = self.select_best_candidate(candidates)
        
        # Escalation判定
        escalation_needed, escalation_reason = self.check_escalation(
            selected_candidate,
            consistency_score,
            candidates
        )
        
        # 結果作成
        result = ValidationResult(
            selected_candidate=selected_candidate,
            selected_index=selected_index,
            consistency_score=consistency_score,
            all_candidates=candidates,
            escalation_needed=escalation_needed,
            escalation_reason=escalation_reason
        )
        
        return result


# 使用例
if __name__ == "__main__":
    # テスト候補
    test_candidates = [
        Candidate(
            response="防衛装備品の調達は防衛装備庁が担当します。",
            judgment="ALLOW",
            reasoning="一般的な情報提供",
            confidence=0.85
        ),
        Candidate(
            response="防衛装備品の調達手続きについて説明します。",
            judgment="ALLOW",
            reasoning="情報提供",
            confidence=0.82
        ),
        Candidate(
            response="機密性が高いため、詳細は上司に確認してください。",
            judgment="ESCALATE",
            reasoning="機密情報の可能性",
            confidence=0.65
        )
    ]
    
    # 検証実行
    validator = SelfConsistencyValidator()
    result = validator.validate(test_candidates, importance="high")
    
    print(f"Selected: {result.selected_candidate.response}")
    print(f"Judgment: {result.selected_candidate.judgment}")
    print(f"Consistency: {result.consistency_score:.3f}")
    print(f"Escalation: {result.escalation_needed}")
    if result.escalation_needed:
        print(f"Reason: {result.escalation_reason}")

