#!/usr/bin/env python3
"""
SO8T Self-Retry Mechanism
自己リトライ機能の実装
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    """リトライ戦略"""
    LINEAR = "linear"           # 線形増加
    EXPONENTIAL = "exponential" # 指数増加
    FIBONACCI = "fibonacci"     # フィボナッチ数列
    ADAPTIVE = "adaptive"       # 適応的

class RetryReason(Enum):
    """リトライ理由"""
    TIMEOUT = "timeout"
    INCONSISTENCY = "inconsistency"
    LOW_CONFIDENCE = "low_confidence"
    SAFETY_VIOLATION = "safety_violation"
    MATHEMATICAL_ERROR = "mathematical_error"
    LOGICAL_ERROR = "logical_error"
    INCOMPLETE_SOLUTION = "incomplete_solution"

@dataclass
class RetryAttempt:
    """リトライ試行のデータ構造"""
    attempt_id: int
    timestamp: datetime
    reason: RetryReason
    strategy: RetryStrategy
    delay_seconds: float
    success: bool
    error_message: Optional[str]
    improvement_score: float
    context: Dict[str, Any]

@dataclass
class RetryConfig:
    """リトライ設定"""
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    confidence_threshold: float = 0.7
    safety_threshold: float = 0.8
    consistency_threshold: float = 0.75
    timeout_seconds: float = 30.0
    enable_adaptive: bool = True
    learning_rate: float = 0.1

class SelfRetryMechanism:
    """自己リトライ機能のメインクラス"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.attempt_history: List[RetryAttempt] = []
        self.learning_data: Dict[str, Any] = {}
        self.adaptive_delays: Dict[RetryReason, float] = {}
        
    async def solve_with_retry(self, problem: str, solver_func: Callable, 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        リトライ機能付きで問題を解決
        
        Args:
            problem: 解決すべき問題
            solver_func: 問題解決関数
            context: 追加のコンテキスト情報
            
        Returns:
            解決結果
        """
        logger.info(f"リトライ機能付きで問題を解決開始: {problem[:100]}...")
        
        best_result = None
        best_score = 0.0
        last_error = None
        
        for attempt_id in range(self.config.max_attempts):
            try:
                # リトライ前の待機
                if attempt_id > 0:
                    delay = self._calculate_delay(attempt_id, last_error)
                    logger.info(f"試行 {attempt_id + 1} 前の待機: {delay:.2f}秒")
                    await asyncio.sleep(delay)
                
                # 問題解決を試行
                start_time = time.time()
                result = await self._execute_with_timeout(
                    solver_func, problem, context, self.config.timeout_seconds
                )
                execution_time = time.time() - start_time
                
                # 結果を評価
                evaluation = await self._evaluate_result(result, problem, context)
                
                # リトライ試行を記録
                retry_attempt = RetryAttempt(
                    attempt_id=attempt_id,
                    timestamp=datetime.now(),
                    reason=RetryReason.INCOMPLETE_SOLUTION,  # デフォルト
                    strategy=self.config.strategy,
                    delay_seconds=delay if attempt_id > 0 else 0,
                    success=evaluation['is_success'],
                    error_message=None,
                    improvement_score=evaluation['improvement_score'],
                    context=context or {}
                )
                self.attempt_history.append(retry_attempt)
                
                # 結果が良好な場合
                if evaluation['is_success'] and evaluation['overall_score'] > best_score:
                    best_result = result
                    best_score = evaluation['overall_score']
                    logger.info(f"試行 {attempt_id + 1} で改善: スコア {evaluation['overall_score']:.3f}")
                
                # 十分な品質に達した場合は終了
                if evaluation['is_success'] and evaluation['overall_score'] >= self.config.confidence_threshold:
                    logger.info(f"試行 {attempt_id + 1} で十分な品質に到達、終了")
                    break
                
                # リトライが必要かどうか判定
                if not self._should_retry(evaluation, attempt_id):
                    logger.info(f"試行 {attempt_id + 1} でリトライを停止")
                    break
                
                # 次の試行のための準備
                last_error = evaluation.get('error_message')
                context = self._update_context_for_retry(context, evaluation, attempt_id)
                
            except asyncio.TimeoutError:
                logger.warning(f"試行 {attempt_id + 1} でタイムアウト")
                last_error = "timeout"
                self._record_failed_attempt(attempt_id, RetryReason.TIMEOUT, "timeout")
                
            except Exception as e:
                logger.error(f"試行 {attempt_id + 1} でエラー: {e}")
                last_error = str(e)
                self._record_failed_attempt(attempt_id, RetryReason.LOGICAL_ERROR, str(e))
        
        # 最終結果を構築
        final_result = self._build_final_result(best_result, best_score, attempt_id + 1)
        
        return final_result
    
    async def _execute_with_timeout(self, solver_func: Callable, problem: str, 
                                  context: Dict[str, Any], timeout: float) -> Any:
        """タイムアウト付きで関数を実行"""
        try:
            result = await asyncio.wait_for(
                solver_func(problem, context),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"実行が {timeout} 秒でタイムアウトしました")
    
    async def _evaluate_result(self, result: Any, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """結果を評価"""
        evaluation = {
            'is_success': False,
            'overall_score': 0.0,
            'confidence_score': 0.0,
            'safety_score': 0.0,
            'consistency_score': 0.0,
            'completeness_score': 0.0,
            'improvement_score': 0.0,
            'error_message': None,
            'recommendations': []
        }
        
        if result is None:
            evaluation['error_message'] = "結果がNoneです"
            return evaluation
        
        # 結果の形式をチェック
        if isinstance(result, dict):
            # 辞書形式の結果を評価
            evaluation['confidence_score'] = result.get('confidence_score', 0.0)
            evaluation['safety_score'] = result.get('safety_score', 0.0)
            evaluation['consistency_score'] = result.get('consistency_score', 0.0)
            evaluation['completeness_score'] = result.get('completeness_score', 0.0)
            
            # 総合スコアを計算
            evaluation['overall_score'] = (
                evaluation['confidence_score'] * 0.3 +
                evaluation['safety_score'] * 0.3 +
                evaluation['consistency_score'] * 0.2 +
                evaluation['completeness_score'] * 0.2
            )
            
            # 成功判定
            evaluation['is_success'] = (
                evaluation['confidence_score'] >= self.config.confidence_threshold and
                evaluation['safety_score'] >= self.config.safety_threshold and
                evaluation['consistency_score'] >= self.config.consistency_threshold
            )
            
            # 改善スコアを計算
            evaluation['improvement_score'] = self._calculate_improvement_score(result)
            
        else:
            # 単純な結果を評価
            evaluation['confidence_score'] = 0.5  # デフォルト値
            evaluation['safety_score'] = 0.8      # デフォルト値
            evaluation['consistency_score'] = 0.5 # デフォルト値
            evaluation['completeness_score'] = 0.5 # デフォルト値
            evaluation['overall_score'] = 0.5
            evaluation['is_success'] = True  # デフォルトで成功とする
        
        return evaluation
    
    def _calculate_improvement_score(self, result: Any) -> float:
        """改善スコアを計算"""
        if not self.attempt_history:
            return 1.0
        
        # 前回の試行と比較
        last_attempt = self.attempt_history[-1]
        if last_attempt.improvement_score > 0:
            return min(1.0, last_attempt.improvement_score * 1.1)  # 10%改善
        else:
            return 0.5  # デフォルト値
    
    def _should_retry(self, evaluation: Dict[str, Any], attempt_id: int) -> bool:
        """リトライが必要かどうか判定"""
        # 最大試行回数に達している
        if attempt_id >= self.config.max_attempts - 1:
            return False
        
        # 成功している場合はリトライ不要
        if evaluation['is_success'] and evaluation['overall_score'] >= self.config.confidence_threshold:
            return False
        
        # 安全性に問題がある場合はリトライ
        if evaluation['safety_score'] < self.config.safety_threshold:
            return True
        
        # 一貫性に問題がある場合はリトライ
        if evaluation['consistency_score'] < self.config.consistency_threshold:
            return True
        
        # 信頼度が低い場合はリトライ
        if evaluation['confidence_score'] < self.config.confidence_threshold:
            return True
        
        return False
    
    def _calculate_delay(self, attempt_id: int, last_error: str = None) -> float:
        """リトライ前の待機時間を計算"""
        if attempt_id == 0:
            return 0.0
        
        # 適応的遅延を使用する場合
        if self.config.enable_adaptive and last_error:
            if last_error in self.adaptive_delays:
                return self.adaptive_delays[last_error]
        
        # 戦略に応じて遅延を計算
        if self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt_id
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (2 ** (attempt_id - 1))
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.base_delay * self._fibonacci(attempt_id)
        elif self.config.strategy == RetryStrategy.ADAPTIVE:
            delay = self._calculate_adaptive_delay(attempt_id, last_error)
        else:
            delay = self.config.base_delay
        
        # 最大遅延時間を適用
        delay = min(delay, self.config.max_delay)
        
        # 適応的遅延を更新
        if self.config.enable_adaptive and last_error:
            self.adaptive_delays[last_error] = delay
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """フィボナッチ数列を計算"""
        if n <= 1:
            return n
        return self._fibonacci(n - 1) + self._fibonacci(n - 2)
    
    def _calculate_adaptive_delay(self, attempt_id: int, last_error: str = None) -> float:
        """適応的遅延を計算"""
        base_delay = self.config.base_delay
        
        # エラータイプに応じて遅延を調整
        if last_error == "timeout":
            return base_delay * 2  # タイムアウトの場合は長めに待機
        elif last_error == "safety_violation":
            return base_delay * 1.5  # 安全性違反の場合は中程度に待機
        elif last_error == "mathematical_error":
            return base_delay * 1.2  # 数学的エラーの場合は短めに待機
        else:
            return base_delay * (1 + attempt_id * 0.5)  # 試行回数に応じて増加
    
    def _update_context_for_retry(self, context: Dict[str, Any], evaluation: Dict[str, Any], 
                                attempt_id: int) -> Dict[str, Any]:
        """リトライ用にコンテキストを更新"""
        if context is None:
            context = {}
        
        # 前回の試行結果をコンテキストに追加
        context['previous_attempts'] = attempt_id
        context['last_evaluation'] = evaluation
        context['retry_count'] = attempt_id + 1
        
        # 学習データを追加
        if 'learning_data' not in context:
            context['learning_data'] = {}
        
        context['learning_data'][f'attempt_{attempt_id}'] = {
            'timestamp': datetime.now().isoformat(),
            'evaluation': evaluation,
            'improvement_score': evaluation.get('improvement_score', 0.0)
        }
        
        return context
    
    def _record_failed_attempt(self, attempt_id: int, reason: RetryReason, error_message: str):
        """失敗した試行を記録"""
        retry_attempt = RetryAttempt(
            attempt_id=attempt_id,
            timestamp=datetime.now(),
            reason=reason,
            strategy=self.config.strategy,
            delay_seconds=0,
            success=False,
            error_message=error_message,
            improvement_score=0.0,
            context={}
        )
        self.attempt_history.append(retry_attempt)
    
    def _build_final_result(self, best_result: Any, best_score: float, total_attempts: int) -> Dict[str, Any]:
        """最終結果を構築"""
        return {
            'solution': best_result,
            'score': best_score,
            'total_attempts': total_attempts,
            'success_rate': len([a for a in self.attempt_history if a.success]) / len(self.attempt_history) if self.attempt_history else 0,
            'attempt_history': [
                {
                    'attempt_id': a.attempt_id,
                    'timestamp': a.timestamp.isoformat(),
                    'reason': a.reason.value,
                    'success': a.success,
                    'improvement_score': a.improvement_score,
                    'error_message': a.error_message
                }
                for a in self.attempt_history
            ],
            'retry_statistics': self._calculate_retry_statistics(),
            'recommendations': self._generate_retry_recommendations()
        }
    
    def _calculate_retry_statistics(self) -> Dict[str, Any]:
        """リトライ統計を計算"""
        if not self.attempt_history:
            return {}
        
        total_attempts = len(self.attempt_history)
        successful_attempts = len([a for a in self.attempt_history if a.success])
        
        # 理由別の統計
        reason_stats = {}
        for attempt in self.attempt_history:
            reason = attempt.reason.value
            if reason not in reason_stats:
                reason_stats[reason] = {'count': 0, 'success': 0}
            reason_stats[reason]['count'] += 1
            if attempt.success:
                reason_stats[reason]['success'] += 1
        
        # 改善スコアの統計
        improvement_scores = [a.improvement_score for a in self.attempt_history if a.improvement_score > 0]
        avg_improvement = np.mean(improvement_scores) if improvement_scores else 0
        
        return {
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'success_rate': successful_attempts / total_attempts,
            'reason_statistics': reason_stats,
            'average_improvement_score': avg_improvement,
            'total_retry_time': sum(a.delay_seconds for a in self.attempt_history)
        }
    
    def _generate_retry_recommendations(self) -> List[str]:
        """リトライ推奨事項を生成"""
        recommendations = []
        
        if not self.attempt_history:
            return recommendations
        
        # 成功率が低い場合
        success_rate = len([a for a in self.attempt_history if a.success]) / len(self.attempt_history)
        if success_rate < 0.5:
            recommendations.append("成功率が低いため、問題の分解やアプローチの変更を検討してください")
        
        # タイムアウトが多い場合
        timeout_count = len([a for a in self.attempt_history if a.reason == RetryReason.TIMEOUT])
        if timeout_count > len(self.attempt_history) * 0.5:
            recommendations.append("タイムアウトが多いため、タイムアウト時間の延長や問題の簡素化を検討してください")
        
        # 安全性違反が多い場合
        safety_violations = len([a for a in self.attempt_history if a.reason == RetryReason.SAFETY_VIOLATION])
        if safety_violations > 0:
            recommendations.append("安全性違反が発生しているため、安全性チェックの強化を検討してください")
        
        # 改善スコアが低い場合
        improvement_scores = [a.improvement_score for a in self.attempt_history if a.improvement_score > 0]
        if improvement_scores and np.mean(improvement_scores) < 0.3:
            recommendations.append("改善スコアが低いため、学習率の調整や戦略の変更を検討してください")
        
        return recommendations
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """学習インサイトを取得"""
        if not self.attempt_history:
            return {}
        
        # 成功パターンの分析
        successful_attempts = [a for a in self.attempt_history if a.success]
        if successful_attempts:
            successful_reasons = [a.reason.value for a in successful_attempts]
            most_common_success_reason = max(set(successful_reasons), key=successful_reasons.count)
        else:
            most_common_success_reason = None
        
        # 失敗パターンの分析
        failed_attempts = [a for a in self.attempt_history if not a.success]
        if failed_attempts:
            failed_reasons = [a.reason.value for a in failed_attempts]
            most_common_failure_reason = max(set(failed_reasons), key=failed_reasons.count)
        else:
            most_common_failure_reason = None
        
        # 改善トレンドの分析
        improvement_trend = []
        for i, attempt in enumerate(self.attempt_history):
            if attempt.improvement_score > 0:
                improvement_trend.append(attempt.improvement_score)
        
        return {
            'most_common_success_reason': most_common_success_reason,
            'most_common_failure_reason': most_common_failure_reason,
            'improvement_trend': improvement_trend,
            'learning_curve': self._calculate_learning_curve(),
            'optimal_retry_strategy': self._suggest_optimal_strategy()
        }
    
    def _calculate_learning_curve(self) -> List[float]:
        """学習曲線を計算"""
        if len(self.attempt_history) < 2:
            return []
        
        scores = []
        for attempt in self.attempt_history:
            if attempt.improvement_score > 0:
                scores.append(attempt.improvement_score)
        
        if len(scores) < 2:
            return []
        
        # 簡易的な学習曲線（移動平均）
        window_size = min(3, len(scores))
        learning_curve = []
        for i in range(window_size - 1, len(scores)):
            window_scores = scores[i - window_size + 1:i + 1]
            learning_curve.append(np.mean(window_scores))
        
        return learning_curve
    
    def _suggest_optimal_strategy(self) -> str:
        """最適なリトライ戦略を提案"""
        if not self.attempt_history:
            return "exponential"
        
        # タイムアウトが多い場合は線形戦略を提案
        timeout_count = len([a for a in self.attempt_history if a.reason == RetryReason.TIMEOUT])
        if timeout_count > len(self.attempt_history) * 0.3:
            return "linear"
        
        # 安全性違反が多い場合は適応的戦略を提案
        safety_violations = len([a for a in self.attempt_history if a.reason == RetryReason.SAFETY_VIOLATION])
        if safety_violations > 0:
            return "adaptive"
        
        # デフォルトは指数戦略
        return "exponential"

# 使用例
async def mock_solver(problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """モックソルバー関数"""
    await asyncio.sleep(0.1)  # 非同期処理のシミュレーション
    
    # ランダムな結果を生成（テスト用）
    import random
    confidence = random.uniform(0.3, 0.9)
    safety = random.uniform(0.5, 1.0)
    consistency = random.uniform(0.4, 0.8)
    
    return {
        'solution': f"Generated solution for: {problem[:50]}...",
        'confidence_score': confidence,
        'safety_score': safety,
        'consistency_score': consistency,
        'completeness_score': random.uniform(0.5, 0.9)
    }

async def main():
    """使用例"""
    config = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        confidence_threshold=0.7,
        safety_threshold=0.8,
        strategy=RetryStrategy.EXPONENTIAL
    )
    
    retry_mechanism = SelfRetryMechanism(config)
    
    problem = "4次元超立方体と2次元平面の交差点の数を求めよ"
    context = {
        'domain': 'mathematics',
        'difficulty': 'high'
    }
    
    result = await retry_mechanism.solve_with_retry(problem, mock_solver, context)
    
    print("=== 自己リトライ結果 ===")
    print(f"最終スコア: {result['score']:.3f}")
    print(f"総試行回数: {result['total_attempts']}")
    print(f"成功率: {result['success_rate']:.3f}")
    
    print("\n試行履歴:")
    for attempt in result['attempt_history']:
        print(f"  試行 {attempt['attempt_id'] + 1}: {attempt['reason']} - {'成功' if attempt['success'] else '失敗'}")
    
    print("\n推奨事項:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
    
    # 学習インサイトを表示
    insights = retry_mechanism.get_learning_insights()
    print(f"\n学習インサイト:")
    print(f"  最適戦略: {insights.get('optimal_retry_strategy', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())
