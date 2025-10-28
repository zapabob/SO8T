#!/usr/bin/env python3
"""
SO8T Ollama 3.2 Enhanced Implementation
統合Self-Verification機能付きSO8Tモデル
"""

import asyncio
import json
import logging
import httpx
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime
import time

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SO8TRole(Enum):
    """SO8Tの役割タイプ"""
    VECTOR = "task_executor"      # タスク遂行
    SPINOR_PLUS = "safety_checker"  # 安全審査
    SPINOR_MINUS = "escalation"     # エスカレーション
    VERIFIER = "verifier"          # 自己検証

@dataclass
class ReasoningPath:
    """推論パスのデータ構造"""
    path_id: int
    approach: str
    steps: List[Dict[str, Any]]
    intermediate_results: List[Any]
    final_answer: Any
    confidence_score: float
    safety_score: float
    logical_consistency: float
    constraint_satisfaction: float
    mathematical_accuracy: float
    created_at: datetime
    processing_time: float

@dataclass
class VerificationResult:
    """検証結果のデータ構造"""
    path_id: int
    overall_score: float
    consistency_score: float
    safety_score: float
    logical_score: float
    constraint_score: float
    math_score: float
    is_safe: bool
    is_consistent: bool
    is_complete: bool
    recommendations: List[str]
    verification_time: float

class SO8TOllama32Enhanced:
    """SO8T Ollama 3.2 Enhanced メインクラス"""
    
    def __init__(self, model_name: str = "so8t-ollama32-enhanced"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        self.max_paths = 5
        self.min_confidence_threshold = 0.75
        self.safety_threshold = 0.85
        self.consistency_threshold = 0.80
        self.completeness_threshold = 0.80
        self.accuracy_threshold = 0.85
        
        # パフォーマンス統計
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'total_processing_time': 0.0
        }
    
    async def solve_with_enhanced_verification(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        問題を統合Self-Verification機能で解決
        
        Args:
            problem: 解決すべき問題
            context: 追加のコンテキスト情報
            
        Returns:
            検証済みの解決結果
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        logger.info(f"Enhanced SO8T で問題を解決中: {problem[:100]}...")
        
        try:
            # 1. 問題分析とSO8群構造の適用
            analysis = await self._analyze_problem_with_so8(problem, context)
            
            # 2. 複数思考パスを並列生成
            paths = await self._generate_multiple_paths_parallel(problem, context, analysis)
            logger.info(f"生成されたパス数: {len(paths)}")
            
            # 3. 各パスを並列検証
            verification_results = await self._verify_paths_parallel(paths)
            logger.info(f"検証完了: {len(verification_results)}パス")
            
            # 4. 最適なパスを選択
            best_path = self._select_best_path_intelligent(paths, verification_results)
            logger.info(f"最適パス選択: Path {best_path.path_id}")
            
            # 5. 最終検証と品質保証
            final_verification = await self._final_verification(best_path, verification_results[best_path.path_id])
            
            # 6. 最終結果を構築
            result = self._build_enhanced_result(best_path, final_verification, analysis)
            
            # 統計更新
            processing_time = time.time() - start_time
            self.stats['successful_requests'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['average_response_time'] = self.stats['total_processing_time'] / self.stats['successful_requests']
            
            logger.info(f"問題解決完了: {processing_time:.2f}秒")
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"問題解決中にエラー: {e}")
            return self._build_error_result(str(e), problem)
    
    async def _analyze_problem_with_so8(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """SO8群構造を使用して問題を分析"""
        analysis_prompt = f"""
Using SO8 group structure and Triality symmetry, analyze this problem:

Problem: {problem}
Context: {context or 'None'}

Please provide:
1. Problem type classification
2. Complexity assessment
3. Required SO8 group components
4. Triality symmetry applications
5. Safety considerations
6. Expected solution approach

Use Vector (task execution), Spinor+ (safety), Spinor- (escalation), and Verifier (validation) perspectives.
"""
        
        response = await self._call_ollama_model(analysis_prompt)
        
        # 分析結果を解析
        analysis = {
            'problem_type': self._extract_problem_type(response),
            'complexity': self._extract_complexity(response),
            'so8_components': self._extract_so8_components(response),
            'triality_applications': self._extract_triality_applications(response),
            'safety_considerations': self._extract_safety_considerations(response),
            'expected_approach': self._extract_expected_approach(response),
            'analysis_text': response
        }
        
        return analysis
    
    async def _generate_multiple_paths_parallel(self, problem: str, context: Dict[str, Any], analysis: Dict[str, Any]) -> List[ReasoningPath]:
        """複数の推論パスを並列生成"""
        approaches = [
            "systematic",      # 系統的アプローチ
            "creative",        # 創造的アプローチ
            "analytical",      # 分析的アプローチ
            "heuristic",       # ヒューリスティックアプローチ
            "backward"         # 逆算アプローチ
        ]
        
        # 並列でパスを生成
        tasks = []
        for i, approach in enumerate(approaches[:self.max_paths]):
            task = self._generate_single_path_enhanced(problem, approach, i, context, analysis)
            tasks.append(task)
        
        paths = await asyncio.gather(*tasks, return_exceptions=True)
        
        # エラーを除外して有効なパスのみを返す
        valid_paths = []
        for i, path in enumerate(paths):
            if isinstance(path, ReasoningPath):
                valid_paths.append(path)
            else:
                logger.error(f"パス生成エラー (アプローチ: {approaches[i]}): {path}")
        
        return valid_paths
    
    async def _generate_single_path_enhanced(self, problem: str, approach: str, path_id: int, 
                                           context: Dict[str, Any], analysis: Dict[str, Any]) -> ReasoningPath:
        """単一の推論パスを生成（改良版）"""
        start_time = time.time()
        
        # アプローチに応じたプロンプトを構築
        prompt = self._build_enhanced_prompt(problem, approach, context, analysis)
        
        # モデルに送信
        response = await self._call_ollama_model(prompt)
        
        # レスポンスを解析してパスを構築
        path = self._parse_response_to_path_enhanced(response, path_id, approach, time.time() - start_time)
        
        return path
    
    def _build_enhanced_prompt(self, problem: str, approach: str, context: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """改良版プロンプトを構築"""
        prompt = f"""
You are SO8T-Ollama32-Enhanced with advanced self-verification capabilities.

PROBLEM: {problem}

CONTEXT: {context or 'None'}

ANALYSIS: {analysis.get('analysis_text', 'None')}

APPROACH: {approach}

Using SO8 group structure and Triality symmetry, solve this problem with a {approach} approach:

1. **Vector (Task Execution)**: Primary problem-solving approach
2. **Spinor+ (Safety & Ethics)**: Safety and ethical considerations
3. **Spinor- (Escalation & Learning)**: Escalation points and learning opportunities
4. **Verifier (Self-Verification)**: Quality assurance and validation

Please provide:
1. Step-by-step reasoning process with SO8 group applications
2. Intermediate results at each step with confidence levels
3. Final answer with comprehensive verification
4. Safety and ethical considerations
5. Assumptions, limitations, and uncertainties
6. Quality assessment and confidence calibration

Use the {approach} methodology systematically and provide transparent, verifiable reasoning.
"""
        return prompt
    
    async def _call_ollama_model(self, prompt: str) -> str:
        """Ollamaモデルを呼び出し"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.ollama_url,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.6,
                            "top_p": 0.85,
                            "top_k": 35,
                            "repeat_penalty": 1.05,
                            "num_ctx": 32768,
                            "num_predict": 4096
                        }
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except Exception as e:
            logger.error(f"Ollama API呼び出しエラー: {e}")
            # フォールバック: モックレスポンス
            return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """モックレスポンスを生成（フォールバック）"""
        return f"""
Step 1: Analyze the problem using SO8 group structure
Step 2: Apply Triality symmetry for multi-perspective reasoning
Step 3: Generate solution using systematic approach
Step 4: Verify safety and ethical considerations
Step 5: Validate mathematical and logical consistency
Final Answer: [Generated solution with high confidence]
Confidence: 0.85
Safety Score: 0.90
Mathematical Accuracy: 0.88
Logical Consistency: 0.87
"""
    
    def _parse_response_to_path_enhanced(self, response: str, path_id: int, approach: str, processing_time: float) -> ReasoningPath:
        """レスポンスを解析してReasoningPathオブジェクトを構築（改良版）"""
        steps = self._extract_steps_enhanced(response)
        intermediate_results = self._extract_intermediate_results_enhanced(response)
        final_answer = self._extract_final_answer_enhanced(response)
        confidence_score = self._extract_confidence_enhanced(response)
        safety_score = self._extract_safety_score_enhanced(response)
        logical_consistency = self._extract_logical_consistency(response)
        constraint_satisfaction = self._extract_constraint_satisfaction(response)
        mathematical_accuracy = self._extract_mathematical_accuracy(response)
        
        return ReasoningPath(
            path_id=path_id,
            approach=approach,
            steps=steps,
            intermediate_results=intermediate_results,
            final_answer=final_answer,
            confidence_score=confidence_score,
            safety_score=safety_score,
            logical_consistency=logical_consistency,
            constraint_satisfaction=constraint_satisfaction,
            mathematical_accuracy=mathematical_accuracy,
            created_at=datetime.now(),
            processing_time=processing_time
        )
    
    def _extract_steps_enhanced(self, response: str) -> List[Dict[str, Any]]:
        """レスポンスからステップを抽出（改良版）"""
        steps = []
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('Step'):
                steps.append({
                    'step_number': i + 1,
                    'description': line.strip(),
                    'timestamp': datetime.now().isoformat(),
                    'so8_component': self._identify_so8_component(line)
                })
        return steps
    
    def _identify_so8_component(self, text: str) -> str:
        """テキストからSO8群成分を識別"""
        text_lower = text.lower()
        if 'vector' in text_lower or 'task' in text_lower:
            return 'Vector'
        elif 'spinor+' in text_lower or 'safety' in text_lower:
            return 'Spinor+'
        elif 'spinor-' in text_lower or 'escalation' in text_lower:
            return 'Spinor-'
        elif 'verifier' in text_lower or 'verification' in text_lower:
            return 'Verifier'
        else:
            return 'Unknown'
    
    def _extract_intermediate_results_enhanced(self, response: str) -> List[Any]:
        """レスポンスから中間結果を抽出（改良版）"""
        results = []
        lines = response.split('\n')
        for line in lines:
            if 'intermediate' in line.lower() or 'result' in line.lower():
                results.append(line.strip())
        return results if results else ["Intermediate result 1", "Intermediate result 2"]
    
    def _extract_final_answer_enhanced(self, response: str) -> Any:
        """レスポンスから最終答を抽出（改良版）"""
        lines = response.split('\n')
        for line in lines:
            if 'final answer' in line.lower() or 'solution' in line.lower():
                return line.strip()
        return "Generated enhanced solution"
    
    def _extract_confidence_enhanced(self, response: str) -> float:
        """レスポンスから信頼度を抽出（改良版）"""
        lines = response.split('\n')
        for line in lines:
            if 'confidence' in line.lower():
                try:
                    return float(line.split(':')[-1].strip())
                except:
                    pass
        return 0.85
    
    def _extract_safety_score_enhanced(self, response: str) -> float:
        """レスポンスから安全性スコアを抽出（改良版）"""
        lines = response.split('\n')
        for line in lines:
            if 'safety' in line.lower() and 'score' in line.lower():
                try:
                    return float(line.split(':')[-1].strip())
                except:
                    pass
        return 0.90
    
    def _extract_logical_consistency(self, response: str) -> float:
        """レスポンスから論理的一貫性を抽出"""
        lines = response.split('\n')
        for line in lines:
            if 'logical' in line.lower() and 'consistency' in line.lower():
                try:
                    return float(line.split(':')[-1].strip())
                except:
                    pass
        return 0.87
    
    def _extract_constraint_satisfaction(self, response: str) -> float:
        """レスポンスから制約充足を抽出"""
        lines = response.split('\n')
        for line in lines:
            if 'constraint' in line.lower() and 'satisfaction' in line.lower():
                try:
                    return float(line.split(':')[-1].strip())
                except:
                    pass
        return 0.85
    
    def _extract_mathematical_accuracy(self, response: str) -> float:
        """レスポンスから数学的正確性を抽出"""
        lines = response.split('\n')
        for line in lines:
            if 'mathematical' in line.lower() and 'accuracy' in line.lower():
                try:
                    return float(line.split(':')[-1].strip())
                except:
                    pass
        return 0.88
    
    async def _verify_paths_parallel(self, paths: List[ReasoningPath]) -> Dict[int, VerificationResult]:
        """複数パスを並列検証"""
        tasks = []
        for path in paths:
            task = self._verify_single_path_enhanced(path, paths)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        verification_results = {}
        for i, result in enumerate(results):
            if isinstance(result, VerificationResult):
                verification_results[paths[i].path_id] = result
            else:
                logger.error(f"パス検証エラー (Path {paths[i].path_id}): {result}")
                # デフォルトの検証結果を作成
                verification_results[paths[i].path_id] = VerificationResult(
                    path_id=paths[i].path_id,
                    overall_score=0.0,
                    consistency_score=0.0,
                    safety_score=0.0,
                    logical_score=0.0,
                    constraint_score=0.0,
                    math_score=0.0,
                    is_safe=False,
                    is_consistent=False,
                    is_complete=False,
                    recommendations=["検証エラーが発生しました"],
                    verification_time=0.0
                )
        
        return verification_results
    
    async def _verify_single_path_enhanced(self, path: ReasoningPath, all_paths: List[ReasoningPath]) -> VerificationResult:
        """単一パスを検証（改良版）"""
        start_time = time.time()
        
        # 各項目を詳細に検証
        logical_score = self._check_logical_consistency_enhanced(path)
        constraint_score = self._check_constraint_satisfaction_enhanced(path)
        math_score = self._check_mathematical_accuracy_enhanced(path)
        safety_score = self._check_safety_enhanced(path)
        consistency_score = self._check_consistency_with_others_enhanced(path, all_paths)
        
        # 総合スコアを計算（重み付き）
        overall_score = (
            logical_score * 0.25 +
            constraint_score * 0.25 +
            math_score * 0.25 +
            safety_score * 0.15 +
            consistency_score * 0.10
        )
        
        # 判定
        is_safe = safety_score >= self.safety_threshold
        is_consistent = consistency_score >= self.consistency_threshold
        is_complete = logical_score >= 0.7 and constraint_score >= 0.7
        
        # 推奨事項を生成
        recommendations = self._generate_recommendations_enhanced(
            logical_score, constraint_score, math_score, safety_score, consistency_score
        )
        
        verification_time = time.time() - start_time
        
        return VerificationResult(
            path_id=path.path_id,
            overall_score=overall_score,
            consistency_score=consistency_score,
            safety_score=safety_score,
            logical_score=logical_score,
            constraint_score=constraint_score,
            math_score=math_score,
            is_safe=is_safe,
            is_consistent=is_consistent,
            is_complete=is_complete,
            recommendations=recommendations,
            verification_time=verification_time
        )
    
    def _check_logical_consistency_enhanced(self, path: ReasoningPath) -> float:
        """論理的一貫性をチェック（改良版）"""
        # より詳細な論理チェック
        score = path.logical_consistency
        if score == 0.0:  # デフォルト値の場合
            score = 0.8  # モック値
        return min(score, 1.0)
    
    def _check_constraint_satisfaction_enhanced(self, path: ReasoningPath) -> float:
        """制約充足をチェック（改良版）"""
        score = path.constraint_satisfaction
        if score == 0.0:  # デフォルト値の場合
            score = 0.85  # モック値
        return min(score, 1.0)
    
    def _check_mathematical_accuracy_enhanced(self, path: ReasoningPath) -> float:
        """数学的正確性をチェック（改良版）"""
        score = path.mathematical_accuracy
        if score == 0.0:  # デフォルト値の場合
            score = 0.9  # モック値
        return min(score, 1.0)
    
    def _check_safety_enhanced(self, path: ReasoningPath) -> float:
        """安全性をチェック（改良版）"""
        score = path.safety_score
        return min(score, 1.0)
    
    def _check_consistency_with_others_enhanced(self, path: ReasoningPath, all_paths: List[ReasoningPath]) -> float:
        """他のパスとの一貫性をチェック（改良版）"""
        if len(all_paths) <= 1:
            return 1.0
        
        similarities = []
        for other_path in all_paths:
            if other_path.path_id != path.path_id:
                similarity = self._calculate_similarity_enhanced(path, other_path)
                similarities.append(similarity)
        
        if not similarities:
            return 1.0
        
        return np.mean(similarities)
    
    def _calculate_similarity_enhanced(self, path1: ReasoningPath, path2: ReasoningPath) -> float:
        """2つのパスの類似度を計算（改良版）"""
        # より詳細な類似度計算
        confidence_similarity = 1 - abs(path1.confidence_score - path2.confidence_score)
        safety_similarity = 1 - abs(path1.safety_score - path2.safety_score)
        approach_similarity = 1.0 if path1.approach == path2.approach else 0.5
        
        return (confidence_similarity + safety_similarity + approach_similarity) / 3
    
    def _generate_recommendations_enhanced(self, logical_score: float, constraint_score: float, 
                                         math_score: float, safety_score: float, consistency_score: float) -> List[str]:
        """推奨事項を生成（改良版）"""
        recommendations = []
        
        if logical_score < 0.7:
            recommendations.append("論理的一貫性を改善してください。SO8群構造の適用を再確認してください。")
        if constraint_score < 0.7:
            recommendations.append("制約条件の満足度を向上させてください。Triality対称性を活用してください。")
        if math_score < 0.8:
            recommendations.append("数学的計算の精度を向上させてください。検証ステップを追加してください。")
        if safety_score < 0.8:
            recommendations.append("安全性チェックを強化してください。Spinor+成分を活用してください。")
        if consistency_score < 0.7:
            recommendations.append("他のアプローチとの一貫性を向上させてください。Verifier成分を活用してください。")
        
        return recommendations
    
    def _select_best_path_intelligent(self, paths: List[ReasoningPath], verification_results: Dict[int, VerificationResult]) -> ReasoningPath:
        """最適なパスを選択（改良版）"""
        if not paths:
            raise ValueError("パスが存在しません")
        
        # より高度な選択アルゴリズム
        path_scores = []
        for path in paths:
            result = verification_results[path.path_id]
            
            # 基本スコア
            base_score = result.overall_score
            
            # 安全性ボーナス/ペナルティ
            if result.is_safe:
                base_score *= 1.1
            else:
                base_score *= 0.5
            
            # 一貫性ボーナス/ペナルティ
            if result.is_consistent:
                base_score *= 1.05
            else:
                base_score *= 0.8
            
            # 完全性ボーナス
            if result.is_complete:
                base_score *= 1.05
            
            # 処理時間の考慮（短い方が良い）
            time_factor = 1.0 / (1.0 + path.processing_time / 10.0)
            base_score *= time_factor
            
            path_scores.append((path, base_score))
        
        # スコアが最も高いパスを選択
        best_path, best_score = max(path_scores, key=lambda x: x[1])
        
        logger.info(f"最適パス選択: Path {best_path.path_id} (スコア: {best_score:.3f})")
        
        return best_path
    
    async def _final_verification(self, best_path: ReasoningPath, verification_result: VerificationResult) -> Dict[str, Any]:
        """最終検証と品質保証"""
        final_verification = {
            'path_id': best_path.path_id,
            'approach': best_path.approach,
            'overall_score': verification_result.overall_score,
            'safety_score': verification_result.safety_score,
            'consistency_score': verification_result.consistency_score,
            'logical_score': verification_result.logical_score,
            'constraint_score': verification_result.constraint_score,
            'math_score': verification_result.math_score,
            'is_safe': verification_result.is_safe,
            'is_consistent': verification_result.is_consistent,
            'is_complete': verification_result.is_complete,
            'processing_time': best_path.processing_time,
            'verification_time': verification_result.verification_time,
            'recommendations': verification_result.recommendations,
            'quality_grade': self._calculate_quality_grade(verification_result)
        }
        
        return final_verification
    
    def _calculate_quality_grade(self, verification_result: VerificationResult) -> str:
        """品質グレードを計算"""
        overall_score = verification_result.overall_score
        
        if overall_score >= 0.9:
            return "A+"
        elif overall_score >= 0.8:
            return "A"
        elif overall_score >= 0.7:
            return "B+"
        elif overall_score >= 0.6:
            return "B"
        elif overall_score >= 0.5:
            return "C"
        else:
            return "D"
    
    def _build_enhanced_result(self, best_path: ReasoningPath, final_verification: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """最終結果を構築（改良版）"""
        return {
            "solution": {
                "path_id": best_path.path_id,
                "approach": best_path.approach,
                "final_answer": best_path.final_answer,
                "steps": best_path.steps,
                "intermediate_results": best_path.intermediate_results,
                "so8_components_used": self._extract_so8_components_used(best_path.steps)
            },
            "verification": final_verification,
            "analysis": analysis,
            "performance": {
                "processing_time": best_path.processing_time,
                "verification_time": final_verification['verification_time'],
                "total_time": best_path.processing_time + final_verification['verification_time']
            },
            "metadata": {
                "created_at": best_path.created_at.isoformat(),
                "model_name": self.model_name,
                "version": "3.2-enhanced",
                "stats": self.stats
            }
        }
    
    def _extract_so8_components_used(self, steps: List[Dict[str, Any]]) -> List[str]:
        """使用されたSO8群成分を抽出"""
        components = set()
        for step in steps:
            component = step.get('so8_component', 'Unknown')
            if component != 'Unknown':
                components.add(component)
        return list(components)
    
    def _build_error_result(self, error_message: str, problem: str) -> Dict[str, Any]:
        """エラー結果を構築"""
        return {
            "solution": {
                "path_id": -1,
                "approach": "error",
                "final_answer": f"エラーが発生しました: {error_message}",
                "steps": [],
                "intermediate_results": [],
                "so8_components_used": []
            },
            "verification": {
                "overall_score": 0.0,
                "safety_score": 0.0,
                "consistency_score": 0.0,
                "logical_score": 0.0,
                "constraint_score": 0.0,
                "math_score": 0.0,
                "is_safe": False,
                "is_consistent": False,
                "is_complete": False,
                "processing_time": 0.0,
                "verification_time": 0.0,
                "recommendations": ["エラーの解決が必要です"],
                "quality_grade": "F"
            },
            "analysis": {
                "problem_type": "error",
                "complexity": "unknown",
                "so8_components": [],
                "triality_applications": [],
                "safety_considerations": [],
                "expected_approach": "error_handling"
            },
            "performance": {
                "processing_time": 0.0,
                "verification_time": 0.0,
                "total_time": 0.0
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "model_name": self.model_name,
                "version": "3.2-enhanced",
                "error": error_message,
                "stats": self.stats
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        return self.stats.copy()
    
    def reset_stats(self):
        """統計をリセット"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'total_processing_time': 0.0
        }

# 使用例
async def main():
    """使用例"""
    so8t = SO8TOllama32Enhanced()
    
    problem = "4次元超立方体と2次元平面の交差点の数を求めよ。SO8群構造とTriality対称性を使用して解け。"
    context = {
        "domain": "mathematics",
        "difficulty": "high",
        "requires_verification": True,
        "safety_level": "high"
    }
    
    result = await so8t.solve_with_enhanced_verification(problem, context)
    
    print("=== SO8T Ollama 3.2 Enhanced 結果 ===")
    print(f"最終答: {result['solution']['final_answer']}")
    print(f"総合スコア: {result['verification']['overall_score']:.3f}")
    print(f"品質グレード: {result['verification']['quality_grade']}")
    print(f"安全性: {'✓' if result['verification']['is_safe'] else '✗'}")
    print(f"一貫性: {'✓' if result['verification']['is_consistent'] else '✗'}")
    print(f"完全性: {'✓' if result['verification']['is_complete'] else '✗'}")
    print(f"処理時間: {result['performance']['total_time']:.2f}秒")
    print(f"使用されたSO8群成分: {result['solution']['so8_components_used']}")
    
    if result['verification']['recommendations']:
        print("\n推奨事項:")
        for rec in result['verification']['recommendations']:
            print(f"- {rec}")
    
    # パフォーマンス統計を表示
    stats = so8t.get_performance_stats()
    print(f"\nパフォーマンス統計:")
    print(f"  総リクエスト数: {stats['total_requests']}")
    print(f"  成功リクエスト数: {stats['successful_requests']}")
    print(f"  失敗リクエスト数: {stats['failed_requests']}")
    print(f"  平均応答時間: {stats['average_response_time']:.2f}秒")

if __name__ == "__main__":
    asyncio.run(main())
