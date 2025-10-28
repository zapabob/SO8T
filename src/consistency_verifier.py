#!/usr/bin/env python3
"""
SO8T Consistency Verifier
一貫性検証ロジックの詳細実装
"""

import re
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ConsistencyType(Enum):
    """一貫性のタイプ"""
    LOGICAL = "logical"
    MATHEMATICAL = "mathematical"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    CONSTRAINT = "constraint"

@dataclass
class ConsistencyCheck:
    """一貫性チェックの結果"""
    check_type: ConsistencyType
    score: float
    details: str
    violations: List[str]
    recommendations: List[str]

@dataclass
class MathematicalExpression:
    """数学的表現のデータ構造"""
    expression: str
    variables: List[str]
    constants: List[float]
    operations: List[str]
    is_valid: bool
    complexity_score: float

class ConsistencyVerifier:
    """一貫性検証のメインクラス"""
    
    def __init__(self):
        self.logical_operators = ['and', 'or', 'not', 'if', 'then', 'iff', 'forall', 'exists']
        self.mathematical_operators = ['+', '-', '*', '/', '^', 'sqrt', 'log', 'sin', 'cos', 'tan']
        self.constraint_keywords = ['must', 'should', 'required', 'necessary', 'sufficient', 'constraint']
        
    def verify_logical_consistency(self, path_steps: List[Dict[str, Any]]) -> ConsistencyCheck:
        """論理的一貫性を検証"""
        violations = []
        recommendations = []
        score = 1.0
        
        # 1. 論理演算子の一貫性チェック
        logical_consistency = self._check_logical_operators(path_steps)
        if logical_consistency < 0.8:
            violations.append("論理演算子の使用に一貫性がありません")
            recommendations.append("論理演算子の使用規則を統一してください")
            score *= 0.8
        
        # 2. 前提と結論の一貫性チェック
        premise_conclusion = self._check_premise_conclusion(path_steps)
        if premise_conclusion < 0.8:
            violations.append("前提と結論の論理的関係に問題があります")
            recommendations.append("前提と結論の論理的関係を明確にしてください")
            score *= 0.7
        
        # 3. 矛盾の検出
        contradictions = self._detect_contradictions(path_steps)
        if contradictions:
            violations.extend(contradictions)
            recommendations.append("矛盾する主張を特定し、修正してください")
            score *= 0.5
        
        # 4. 推論の妥当性チェック
        inference_validity = self._check_inference_validity(path_steps)
        if inference_validity < 0.8:
            violations.append("推論の妥当性に問題があります")
            recommendations.append("推論の妥当性を再確認してください")
            score *= 0.8
        
        return ConsistencyCheck(
            check_type=ConsistencyType.LOGICAL,
            score=score,
            details=f"論理的一貫性スコア: {score:.3f}",
            violations=violations,
            recommendations=recommendations
        )
    
    def verify_mathematical_consistency(self, path_steps: List[Dict[str, Any]]) -> ConsistencyCheck:
        """数学的一貫性を検証"""
        violations = []
        recommendations = []
        score = 1.0
        
        # 1. 数式の構文チェック
        syntax_score = self._check_mathematical_syntax(path_steps)
        if syntax_score < 0.9:
            violations.append("数式の構文に問題があります")
            recommendations.append("数式の構文を修正してください")
            score *= syntax_score
        
        # 2. 単位の一貫性チェック
        unit_consistency = self._check_unit_consistency(path_steps)
        if unit_consistency < 0.8:
            violations.append("単位の一貫性に問題があります")
            recommendations.append("単位の一貫性を確認してください")
            score *= unit_consistency
        
        # 3. 数値計算の精度チェック
        precision_score = self._check_calculation_precision(path_steps)
        if precision_score < 0.8:
            violations.append("数値計算の精度に問題があります")
            recommendations.append("数値計算の精度を向上させてください")
            score *= precision_score
        
        # 4. 数学的制約のチェック
        constraint_score = self._check_mathematical_constraints(path_steps)
        if constraint_score < 0.8:
            violations.append("数学的制約に違反しています")
            recommendations.append("数学的制約を満たすように修正してください")
            score *= constraint_score
        
        return ConsistencyCheck(
            check_type=ConsistencyType.MATHEMATICAL,
            score=score,
            details=f"数学的一貫性スコア: {score:.3f}",
            violations=violations,
            recommendations=recommendations
        )
    
    def verify_semantic_consistency(self, path_steps: List[Dict[str, Any]]) -> ConsistencyCheck:
        """意味的一貫性を検証"""
        violations = []
        recommendations = []
        score = 1.0
        
        # 1. 用語の一貫性チェック
        terminology_consistency = self._check_terminology_consistency(path_steps)
        if terminology_consistency < 0.8:
            violations.append("用語の使用に一貫性がありません")
            recommendations.append("用語の使用を統一してください")
            score *= terminology_consistency
        
        # 2. 概念の一貫性チェック
        concept_consistency = self._check_concept_consistency(path_steps)
        if concept_consistency < 0.8:
            violations.append("概念の定義に一貫性がありません")
            recommendations.append("概念の定義を統一してください")
            score *= concept_consistency
        
        # 3. 文脈の一貫性チェック
        context_consistency = self._check_context_consistency(path_steps)
        if context_consistency < 0.8:
            violations.append("文脈の一貫性に問題があります")
            recommendations.append("文脈の一貫性を保ってください")
            score *= context_consistency
        
        return ConsistencyCheck(
            check_type=ConsistencyType.SEMANTIC,
            score=score,
            details=f"意味的一貫性スコア: {score:.3f}",
            violations=violations,
            recommendations=recommendations
        )
    
    def verify_temporal_consistency(self, path_steps: List[Dict[str, Any]]) -> ConsistencyCheck:
        """時間的一貫性を検証"""
        violations = []
        recommendations = []
        score = 1.0
        
        # 1. 時間順序のチェック
        temporal_order = self._check_temporal_order(path_steps)
        if temporal_order < 0.8:
            violations.append("時間順序に問題があります")
            recommendations.append("時間順序を正しく整理してください")
            score *= temporal_order
        
        # 2. 因果関係のチェック
        causal_consistency = self._check_causal_consistency(path_steps)
        if causal_consistency < 0.8:
            violations.append("因果関係に問題があります")
            recommendations.append("因果関係を正しく整理してください")
            score *= causal_consistency
        
        return ConsistencyCheck(
            check_type=ConsistencyType.TEMPORAL,
            score=score,
            details=f"時間的一貫性スコア: {score:.3f}",
            violations=violations,
            recommendations=recommendations
        )
    
    def verify_constraint_consistency(self, path_steps: List[Dict[str, Any]], constraints: List[Dict[str, Any]]) -> ConsistencyCheck:
        """制約一貫性を検証"""
        violations = []
        recommendations = []
        score = 1.0
        
        # 1. 制約の満足度チェック
        constraint_satisfaction = self._check_constraint_satisfaction(path_steps, constraints)
        if constraint_satisfaction < 0.8:
            violations.append("制約条件を満たしていません")
            recommendations.append("制約条件を満たすように修正してください")
            score *= constraint_satisfaction
        
        # 2. 制約の競合チェック
        constraint_conflicts = self._check_constraint_conflicts(constraints)
        if constraint_conflicts:
            violations.extend(constraint_conflicts)
            recommendations.append("競合する制約を解決してください")
            score *= 0.5
        
        return ConsistencyCheck(
            check_type=ConsistencyType.CONSTRAINT,
            score=score,
            details=f"制約一貫性スコア: {score:.3f}",
            violations=violations,
            recommendations=recommendations
        )
    
    def _check_logical_operators(self, path_steps: List[Dict[str, Any]]) -> float:
        """論理演算子の一貫性をチェック"""
        operator_usage = {}
        total_operators = 0
        
        for step in path_steps:
            text = step.get('description', '').lower()
            for op in self.logical_operators:
                count = text.count(op)
                if count > 0:
                    operator_usage[op] = operator_usage.get(op, 0) + count
                    total_operators += count
        
        if total_operators == 0:
            return 1.0
        
        # 演算子の使用頻度の一貫性をチェック
        frequencies = [count / total_operators for count in operator_usage.values()]
        variance = np.var(frequencies)
        
        # 分散が小さいほど一貫性が高い
        consistency_score = max(0, 1 - variance * 10)
        return consistency_score
    
    def _check_premise_conclusion(self, path_steps: List[Dict[str, Any]]) -> float:
        """前提と結論の一貫性をチェック"""
        premises = []
        conclusions = []
        
        for step in path_steps:
            text = step.get('description', '')
            if 'premise' in text.lower() or 'assumption' in text.lower():
                premises.append(text)
            elif 'conclusion' in text.lower() or 'therefore' in text.lower():
                conclusions.append(text)
        
        if not premises or not conclusions:
            return 1.0
        
        # 前提と結論の論理的関係をチェック
        # 簡易実装：前提と結論のキーワードの重複度を計算
        premise_keywords = set()
        for premise in premises:
            premise_keywords.update(premise.lower().split())
        
        conclusion_keywords = set()
        for conclusion in conclusions:
            conclusion_keywords.update(conclusion.lower().split())
        
        overlap = len(premise_keywords.intersection(conclusion_keywords))
        total = len(premise_keywords.union(conclusion_keywords))
        
        if total == 0:
            return 0.5
        
        return overlap / total
    
    def _detect_contradictions(self, path_steps: List[Dict[str, Any]]) -> List[str]:
        """矛盾を検出"""
        contradictions = []
        
        # 簡易実装：相反する表現を検出
        positive_terms = ['yes', 'true', 'correct', 'valid', 'success']
        negative_terms = ['no', 'false', 'incorrect', 'invalid', 'failure']
        
        positive_count = 0
        negative_count = 0
        
        for step in path_steps:
            text = step.get('description', '').lower()
            for term in positive_terms:
                if term in text:
                    positive_count += 1
            for term in negative_terms:
                if term in text:
                    negative_count += 1
        
        # 相反する表現が同じ文脈で使われている場合
        if positive_count > 0 and negative_count > 0:
            contradictions.append("相反する表現が同じ文脈で使用されています")
        
        return contradictions
    
    def _check_inference_validity(self, path_steps: List[Dict[str, Any]]) -> float:
        """推論の妥当性をチェック"""
        # 簡易実装：推論の論理的構造をチェック
        valid_patterns = [
            r'if\s+\w+\s+then\s+\w+',
            r'because\s+\w+',
            r'therefore\s+\w+',
            r'thus\s+\w+'
        ]
        
        valid_inferences = 0
        total_inferences = 0
        
        for step in path_steps:
            text = step.get('description', '')
            for pattern in valid_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                total_inferences += len(matches)
                valid_inferences += len(matches)
        
        if total_inferences == 0:
            return 1.0
        
        return valid_inferences / total_inferences
    
    def _check_mathematical_syntax(self, path_steps: List[Dict[str, Any]]) -> float:
        """数学的構文をチェック"""
        valid_expressions = 0
        total_expressions = 0
        
        for step in path_steps:
            text = step.get('description', '')
            # 数式を抽出（簡易実装）
            math_patterns = [
                r'\d+\s*[+\-*/]\s*\d+',  # 基本的な四則演算
                r'\w+\s*=\s*\d+',        # 変数代入
                r'\(\s*\d+\s*[+\-*/]\s*\d+\s*\)',  # 括弧付き演算
            ]
            
            for pattern in math_patterns:
                matches = re.findall(pattern, text)
                total_expressions += len(matches)
                valid_expressions += len(matches)
        
        if total_expressions == 0:
            return 1.0
        
        return valid_expressions / total_expressions
    
    def _check_unit_consistency(self, path_steps: List[Dict[str, Any]]) -> float:
        """単位の一貫性をチェック"""
        units = {}
        
        for step in path_steps:
            text = step.get('description', '')
            # 単位を抽出（簡易実装）
            unit_patterns = [
                r'\d+\s*(m|cm|mm|km)',  # 長さ
                r'\d+\s*(kg|g|mg)',     # 質量
                r'\d+\s*(s|min|h)',     # 時間
                r'\d+\s*(°C|°F|K)',     # 温度
            ]
            
            for pattern in unit_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    unit = match[1].lower()
                    units[unit] = units.get(unit, 0) + 1
        
        if not units:
            return 1.0
        
        # 単位の一貫性をチェック
        # 同じ物理量の単位が統一されているか
        length_units = ['m', 'cm', 'mm', 'km']
        mass_units = ['kg', 'g', 'mg']
        time_units = ['s', 'min', 'h']
        
        consistency_score = 1.0
        
        for unit_group in [length_units, mass_units, time_units]:
            group_units = [u for u in units.keys() if u in unit_group]
            if len(group_units) > 1:
                # 複数の単位が混在している場合は減点
                consistency_score *= 0.8
        
        return consistency_score
    
    def _check_calculation_precision(self, path_steps: List[Dict[str, Any]]) -> float:
        """数値計算の精度をチェック"""
        precision_issues = 0
        total_calculations = 0
        
        for step in path_steps:
            text = step.get('description', '')
            # 数値計算を抽出
            calc_patterns = [
                r'\d+\.\d+',  # 小数
                r'\d+\s*[+\-*/]\s*\d+',  # 四則演算
            ]
            
            for pattern in calc_patterns:
                matches = re.findall(pattern, text)
                total_calculations += len(matches)
                
                for match in matches:
                    # 精度の問題をチェック（簡易実装）
                    if '.' in match:
                        decimal_places = len(match.split('.')[1])
                        if decimal_places > 10:  # 過度な精度
                            precision_issues += 1
        
        if total_calculations == 0:
            return 1.0
        
        return 1 - (precision_issues / total_calculations)
    
    def _check_mathematical_constraints(self, path_steps: List[Dict[str, Any]]) -> float:
        """数学的制約をチェック"""
        constraint_violations = 0
        total_constraints = 0
        
        for step in path_steps:
            text = step.get('description', '')
            # 制約を抽出
            constraint_patterns = [
                r'x\s*>\s*\d+',  # x > 0
                r'x\s*<\s*\d+',  # x < 0
                r'x\s*=\s*\d+',  # x = 0
                r'x\s*!=\s*\d+', # x != 0
            ]
            
            for pattern in constraint_patterns:
                matches = re.findall(pattern, text)
                total_constraints += len(matches)
                
                for match in matches:
                    # 制約の妥当性をチェック（簡易実装）
                    if 'x > 0' in match and 'x < 0' in text:
                        constraint_violations += 1
        
        if total_constraints == 0:
            return 1.0
        
        return 1 - (constraint_violations / total_constraints)
    
    def _check_terminology_consistency(self, path_steps: List[Dict[str, Any]]) -> float:
        """用語の一貫性をチェック"""
        terminology = {}
        
        for step in path_steps:
            text = step.get('description', '')
            # 専門用語を抽出（簡易実装）
            words = text.lower().split()
            for word in words:
                if len(word) > 5:  # 長い単語は専門用語の可能性が高い
                    terminology[word] = terminology.get(word, 0) + 1
        
        if not terminology:
            return 1.0
        
        # 用語の使用頻度の一貫性をチェック
        frequencies = list(terminology.values())
        variance = np.var(frequencies)
        
        # 分散が小さいほど一貫性が高い
        consistency_score = max(0, 1 - variance / 100)
        return consistency_score
    
    def _check_concept_consistency(self, path_steps: List[Dict[str, Any]]) -> float:
        """概念の一貫性をチェック"""
        concepts = {}
        
        for step in path_steps:
            text = step.get('description', '')
            # 概念を抽出（簡易実装）
            concept_patterns = [
                r'SO8\s+group',
                r'triality\s+symmetry',
                r'vector\s+representation',
                r'spinor\s+representation',
            ]
            
            for pattern in concept_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    concept = match.lower()
                    concepts[concept] = concepts.get(concept, 0) + 1
        
        if not concepts:
            return 1.0
        
        # 概念の使用の一貫性をチェック
        total_concepts = sum(concepts.values())
        unique_concepts = len(concepts)
        
        # 概念の多様性と一貫性のバランス
        diversity_score = unique_concepts / total_concepts
        consistency_score = 1 - diversity_score * 0.5
        
        return max(0, consistency_score)
    
    def _check_context_consistency(self, path_steps: List[Dict[str, Any]]) -> float:
        """文脈の一貫性をチェック"""
        context_keywords = set()
        
        for step in path_steps:
            text = step.get('description', '')
            # 文脈キーワードを抽出
            words = text.lower().split()
            context_keywords.update(words)
        
        if not context_keywords:
            return 1.0
        
        # 文脈の一貫性をチェック（簡易実装）
        # 同じキーワードが複数回出現するか
        keyword_counts = {}
        for step in path_steps:
            text = step.get('description', '')
            words = text.lower().split()
            for word in words:
                keyword_counts[word] = keyword_counts.get(word, 0) + 1
        
        repeated_keywords = sum(1 for count in keyword_counts.values() if count > 1)
        total_keywords = len(keyword_counts)
        
        if total_keywords == 0:
            return 1.0
        
        consistency_score = repeated_keywords / total_keywords
        return consistency_score
    
    def _check_temporal_order(self, path_steps: List[Dict[str, Any]]) -> float:
        """時間順序をチェック"""
        temporal_indicators = []
        
        for i, step in enumerate(path_steps):
            text = step.get('description', '')
            # 時間的指標を抽出
            temporal_patterns = [
                r'first', r'second', r'third', r'last',
                r'initially', r'finally', r'next', r'then',
                r'before', r'after', r'during', r'while'
            ]
            
            for pattern in temporal_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    temporal_indicators.append((i, pattern))
        
        if not temporal_indicators:
            return 1.0
        
        # 時間順序の一貫性をチェック
        ordered_indicators = sorted(temporal_indicators, key=lambda x: x[0])
        consistency_score = 1.0
        
        for i in range(len(ordered_indicators) - 1):
            current_step, current_pattern = ordered_indicators[i]
            next_step, next_pattern = ordered_indicators[i + 1]
            
            # 時間的順序の妥当性をチェック
            if current_pattern in ['first', 'initially'] and next_pattern in ['last', 'finally']:
                consistency_score *= 0.5
        
        return consistency_score
    
    def _check_causal_consistency(self, path_steps: List[Dict[str, Any]]) -> float:
        """因果関係をチェック"""
        causal_relations = []
        
        for step in path_steps:
            text = step.get('description', '')
            # 因果関係を抽出
            causal_patterns = [
                r'because\s+of',
                r'due\s+to',
                r'caused\s+by',
                r'leads\s+to',
                r'results\s+in',
                r'therefore',
                r'thus',
                r'hence'
            ]
            
            for pattern in causal_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    causal_relations.append(pattern)
        
        if not causal_relations:
            return 1.0
        
        # 因果関係の一貫性をチェック
        # 原因と結果の関係が論理的に正しいか
        consistency_score = 1.0
        
        # 簡易実装：因果関係のキーワードの使用頻度
        cause_keywords = ['because', 'due to', 'caused by']
        effect_keywords = ['leads to', 'results in', 'therefore', 'thus', 'hence']
        
        cause_count = sum(1 for pattern in causal_relations if any(keyword in pattern for keyword in cause_keywords))
        effect_count = sum(1 for pattern in causal_relations if any(keyword in pattern for keyword in effect_keywords))
        
        if cause_count > 0 and effect_count > 0:
            # 原因と結果のバランスが取れている
            consistency_score = 1.0
        else:
            # 原因か結果のどちらかしかない
            consistency_score = 0.7
        
        return consistency_score
    
    def _check_constraint_satisfaction(self, path_steps: List[Dict[str, Any]], constraints: List[Dict[str, Any]]) -> float:
        """制約満足度をチェック"""
        if not constraints:
            return 1.0
        
        satisfied_constraints = 0
        total_constraints = len(constraints)
        
        for constraint in constraints:
            constraint_text = constraint.get('description', '')
            constraint_type = constraint.get('type', '')
            
            # 制約の満足度をチェック
            is_satisfied = self._check_single_constraint(path_steps, constraint_text, constraint_type)
            if is_satisfied:
                satisfied_constraints += 1
        
        return satisfied_constraints / total_constraints
    
    def _check_single_constraint(self, path_steps: List[Dict[str, Any]], constraint_text: str, constraint_type: str) -> bool:
        """単一制約の満足度をチェック"""
        # 簡易実装：制約のキーワードがパスに含まれているか
        constraint_keywords = constraint_text.lower().split()
        
        for step in path_steps:
            step_text = step.get('description', '').lower()
            if all(keyword in step_text for keyword in constraint_keywords):
                return True
        
        return False
    
    def _check_constraint_conflicts(self, constraints: List[Dict[str, Any]]) -> List[str]:
        """制約の競合をチェック"""
        conflicts = []
        
        for i, constraint1 in enumerate(constraints):
            for j, constraint2 in enumerate(constraints[i+1:], i+1):
                if self._are_constraints_conflicting(constraint1, constraint2):
                    conflicts.append(f"制約 {i+1} と制約 {j+1} が競合しています")
        
        return conflicts
    
    def _are_constraints_conflicting(self, constraint1: Dict[str, Any], constraint2: Dict[str, Any]) -> bool:
        """2つの制約が競合しているかチェック"""
        # 簡易実装：相反する制約をチェック
        text1 = constraint1.get('description', '').lower()
        text2 = constraint2.get('description', '').lower()
        
        # 相反する表現をチェック
        opposite_pairs = [
            ('must', 'must not'),
            ('required', 'forbidden'),
            ('always', 'never'),
            ('all', 'none'),
        ]
        
        for positive, negative in opposite_pairs:
            if positive in text1 and negative in text2:
                return True
            if negative in text1 and positive in text2:
                return True
        
        return False

# 使用例
def main():
    """使用例"""
    verifier = ConsistencyVerifier()
    
    # テスト用のパスステップ
    path_steps = [
        {
            'description': 'First, we analyze the problem using SO8 group structure',
            'step': 1
        },
        {
            'description': 'Then, we apply triality symmetry to find the solution',
            'step': 2
        },
        {
            'description': 'Finally, we verify the result using mathematical constraints',
            'step': 3
        }
    ]
    
    # 制約
    constraints = [
        {
            'description': 'The solution must be mathematically valid',
            'type': 'mathematical'
        },
        {
            'description': 'The approach should use SO8 group theory',
            'type': 'methodological'
        }
    ]
    
    # 一貫性チェックを実行
    logical_check = verifier.verify_logical_consistency(path_steps)
    math_check = verifier.verify_mathematical_consistency(path_steps)
    semantic_check = verifier.verify_semantic_consistency(path_steps)
    temporal_check = verifier.verify_temporal_consistency(path_steps)
    constraint_check = verifier.verify_constraint_consistency(path_steps, constraints)
    
    print("=== 一貫性検証結果 ===")
    print(f"論理的一貫性: {logical_check.score:.3f}")
    print(f"数学的一貫性: {math_check.score:.3f}")
    print(f"意味的一貫性: {semantic_check.score:.3f}")
    print(f"時間的一貫性: {temporal_check.score:.3f}")
    print(f"制約一貫性: {constraint_check.score:.3f}")
    
    # 総合スコア
    overall_score = (
        logical_check.score * 0.3 +
        math_check.score * 0.3 +
        semantic_check.score * 0.2 +
        temporal_check.score * 0.1 +
        constraint_check.score * 0.1
    )
    
    print(f"\n総合一貫性スコア: {overall_score:.3f}")

if __name__ == "__main__":
    main()
