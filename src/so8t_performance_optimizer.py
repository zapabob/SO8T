#!/usr/bin/env python3
"""
SO8T Performance Optimizer
パフォーマンス最適化機能の実装
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime
import psutil
import gc

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """最適化レベル"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    MAXIMUM = "maximum"

@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    cpu_usage: float
    memory_usage: float
    response_time: float
    throughput: float
    error_rate: float
    cache_hit_rate: float
    parallel_efficiency: float
    timestamp: datetime

@dataclass
class OptimizationConfig:
    """最適化設定"""
    max_concurrent_requests: int = 10
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1時間
    memory_limit_mb: int = 2048
    cpu_limit_percent: int = 80
    response_timeout: int = 60
    retry_attempts: int = 3
    batch_size: int = 5

class SO8TPerformanceOptimizer:
    """SO8T パフォーマンス最適化クラス"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.cache = {}
        self.request_queue = asyncio.Queue()
        self.active_requests = 0
        self.performance_history = []
        self.optimization_level = OptimizationLevel.BASIC
        
        # パフォーマンス監視
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
    async def optimize_performance(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """パフォーマンス最適化された問題解決"""
        start_time = time.time()
        
        # 1. リソース監視
        current_metrics = await self._monitor_resources()
        
        # 2. 最適化レベルを決定
        optimization_level = self._determine_optimization_level(current_metrics)
        
        # 3. キャッシュチェック
        cache_key = self._generate_cache_key(problem, context)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            logger.info("キャッシュから結果を取得")
            return cached_result
        
        # 4. 並列処理の最適化
        if optimization_level in [OptimizationLevel.ADVANCED, OptimizationLevel.MAXIMUM]:
            result = await self._optimized_parallel_processing(problem, context, optimization_level)
        else:
            result = await self._standard_processing(problem, context, optimization_level)
        
        # 5. 結果をキャッシュに保存
        self._save_to_cache(cache_key, result)
        
        # 6. パフォーマンスメトリクスを更新
        response_time = time.time() - start_time
        await self._update_performance_metrics(response_time, True)
        
        return result
    
    async def _monitor_resources(self) -> PerformanceMetrics:
        """リソースを監視"""
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # スループット計算
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        throughput = self.total_requests / elapsed_time if elapsed_time > 0 else 0
        
        # エラー率計算
        error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0
        
        # キャッシュヒット率計算
        cache_hits = len([r for r in self.performance_history if r.get('cache_hit', False)])
        cache_hit_rate = cache_hits / len(self.performance_history) if self.performance_history else 0
        
        # 並列効率計算
        parallel_efficiency = min(1.0, self.active_requests / self.config.max_concurrent_requests)
        
        metrics = PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            response_time=0.0,  # 後で更新
            throughput=throughput,
            error_rate=error_rate,
            cache_hit_rate=cache_hit_rate,
            parallel_efficiency=parallel_efficiency,
            timestamp=datetime.now()
        )
        
        self.performance_history.append(metrics)
        
        # 履歴を制限
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        return metrics
    
    def _determine_optimization_level(self, metrics: PerformanceMetrics) -> OptimizationLevel:
        """最適化レベルを決定"""
        if metrics.cpu_usage > 90 or metrics.memory_usage > 90:
            return OptimizationLevel.BASIC
        elif metrics.cpu_usage > 70 or metrics.memory_usage > 70:
            return OptimizationLevel.INTERMEDIATE
        elif metrics.cpu_usage > 50 or metrics.memory_usage > 50:
            return OptimizationLevel.ADVANCED
        else:
            return OptimizationLevel.MAXIMUM
    
    def _generate_cache_key(self, problem: str, context: Dict[str, Any] = None) -> str:
        """キャッシュキーを生成"""
        import hashlib
        
        key_data = f"{problem}:{context or 'None'}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """キャッシュから結果を取得"""
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            # TTLチェック
            if time.time() - cached_item['timestamp'] < self.config.cache_ttl:
                return cached_item['result']
            else:
                # 期限切れの場合は削除
                del self.cache[cache_key]
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """結果をキャッシュに保存"""
        # キャッシュサイズ制限
        if len(self.cache) >= self.config.cache_size:
            # 最も古いアイテムを削除
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    async def _optimized_parallel_processing(self, problem: str, context: Dict[str, Any], 
                                           optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """最適化された並列処理"""
        logger.info(f"最適化された並列処理を実行: {optimization_level.value}")
        
        # バッチサイズを最適化レベルに応じて調整
        batch_size = self.config.batch_size
        if optimization_level == OptimizationLevel.MAXIMUM:
            batch_size = min(batch_size * 2, 10)
        
        # 複数のアプローチを並列で実行
        approaches = ["systematic", "creative", "analytical", "heuristic", "backward"]
        tasks = []
        
        for i, approach in enumerate(approaches[:batch_size]):
            task = self._process_single_approach(problem, approach, i, context)
            tasks.append(task)
        
        # 並列実行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果を統合
        valid_results = [r for r in results if isinstance(r, dict) and 'error' not in r]
        
        if not valid_results:
            return self._build_error_result("すべてのアプローチでエラーが発生しました")
        
        # 最適な結果を選択
        best_result = self._select_best_result(valid_results)
        
        return best_result
    
    async def _standard_processing(self, problem: str, context: Dict[str, Any], 
                                 optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """標準処理"""
        logger.info(f"標準処理を実行: {optimization_level.value}")
        
        # 単一アプローチで処理
        result = await self._process_single_approach(problem, "systematic", 0, context)
        
        return result
    
    async def _process_single_approach(self, problem: str, approach: str, approach_id: int, 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """単一アプローチを処理"""
        try:
            # リソース制限チェック
            if not await self._check_resource_limits():
                return self._build_error_result("リソース制限に達しました")
            
            # リクエストキューに追加
            await self.request_queue.put((problem, approach, approach_id, context))
            self.active_requests += 1
            
            # 実際の処理をシミュレート（ここではモック）
            await asyncio.sleep(0.1)  # 処理時間のシミュレーション
            
            # モック結果を生成
            result = self._generate_mock_result(problem, approach, approach_id)
            
            self.active_requests -= 1
            return result
            
        except Exception as e:
            logger.error(f"アプローチ処理エラー: {e}")
            return self._build_error_result(str(e))
    
    async def _check_resource_limits(self) -> bool:
        """リソース制限をチェック"""
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        if cpu_usage > self.config.cpu_limit_percent:
            logger.warning(f"CPU使用率が制限を超過: {cpu_usage}%")
            return False
        
        if memory_usage > 90:  # メモリ使用率90%以上で制限
            logger.warning(f"メモリ使用率が制限を超過: {memory_usage}%")
            return False
        
        if self.active_requests >= self.config.max_concurrent_requests:
            logger.warning(f"同時リクエスト数が制限を超過: {self.active_requests}")
            return False
        
        return True
    
    def _generate_mock_result(self, problem: str, approach: str, approach_id: int) -> Dict[str, Any]:
        """モック結果を生成"""
        return {
            'solution': {
                'approach': approach,
                'approach_id': approach_id,
                'final_answer': f"Generated solution for: {problem[:50]}...",
                'confidence_score': np.random.uniform(0.7, 0.95),
                'safety_score': np.random.uniform(0.8, 0.95),
                'consistency_score': np.random.uniform(0.75, 0.90)
            },
            'verification': {
                'overall_score': np.random.uniform(0.7, 0.95),
                'is_safe': True,
                'is_consistent': True,
                'is_complete': True
            },
            'performance': {
                'processing_time': np.random.uniform(0.1, 2.0),
                'memory_usage': np.random.uniform(100, 500)
            },
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'optimization_level': 'enhanced'
            }
        }
    
    def _select_best_result(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """最適な結果を選択"""
        if not results:
            return self._build_error_result("結果がありません")
        
        # スコアに基づいて最適な結果を選択
        best_result = max(results, key=lambda r: r.get('verification', {}).get('overall_score', 0))
        
        return best_result
    
    def _build_error_result(self, error_message: str) -> Dict[str, Any]:
        """エラー結果を構築"""
        return {
            'solution': {
                'approach': 'error',
                'final_answer': f"エラー: {error_message}",
                'confidence_score': 0.0,
                'safety_score': 0.0,
                'consistency_score': 0.0
            },
            'verification': {
                'overall_score': 0.0,
                'is_safe': False,
                'is_consistent': False,
                'is_complete': False
            },
            'performance': {
                'processing_time': 0.0,
                'memory_usage': 0
            },
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'error': error_message
            }
        }
    
    async def _update_performance_metrics(self, response_time: float, success: bool):
        """パフォーマンスメトリクスを更新"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # 最新のメトリクスを更新
        if self.performance_history:
            self.performance_history[-1].response_time = response_time
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンスサマリーを取得"""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-10:]  # 最近の10件
        
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        avg_response_time = np.mean([m.response_time for m in recent_metrics])
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        avg_error_rate = np.mean([m.error_rate for m in recent_metrics])
        avg_cache_hit_rate = np.mean([m.cache_hit_rate for m in recent_metrics])
        avg_parallel_efficiency = np.mean([m.parallel_efficiency for m in recent_metrics])
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
            'average_cpu_usage': avg_cpu,
            'average_memory_usage': avg_memory,
            'average_response_time': avg_response_time,
            'average_throughput': avg_throughput,
            'average_error_rate': avg_error_rate,
            'average_cache_hit_rate': avg_cache_hit_rate,
            'average_parallel_efficiency': avg_parallel_efficiency,
            'cache_size': len(self.cache),
            'active_requests': self.active_requests
        }
    
    def optimize_cache(self):
        """キャッシュを最適化"""
        current_time = time.time()
        
        # 期限切れのアイテムを削除
        expired_keys = [
            key for key, item in self.cache.items()
            if current_time - item['timestamp'] > self.config.cache_ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        logger.info(f"キャッシュ最適化完了: {len(expired_keys)}件の期限切れアイテムを削除")
    
    def optimize_memory(self):
        """メモリを最適化"""
        # ガベージコレクションを実行
        collected = gc.collect()
        
        # パフォーマンス履歴を制限
        if len(self.performance_history) > 500:
            self.performance_history = self.performance_history[-500:]
        
        logger.info(f"メモリ最適化完了: {collected}件のオブジェクトを回収")
    
    def get_optimization_recommendations(self) -> List[str]:
        """最適化推奨事項を取得"""
        recommendations = []
        summary = self.get_performance_summary()
        
        if summary.get('average_cpu_usage', 0) > 80:
            recommendations.append("CPU使用率が高いため、並列処理数を削減することを推奨します")
        
        if summary.get('average_memory_usage', 0) > 80:
            recommendations.append("メモリ使用率が高いため、キャッシュサイズを削減することを推奨します")
        
        if summary.get('average_response_time', 0) > 5.0:
            recommendations.append("応答時間が長いため、処理の最適化を推奨します")
        
        if summary.get('average_cache_hit_rate', 0) < 0.3:
            recommendations.append("キャッシュヒット率が低いため、キャッシュ戦略の見直しを推奨します")
        
        if summary.get('average_parallel_efficiency', 0) < 0.5:
            recommendations.append("並列効率が低いため、同時リクエスト数の調整を推奨します")
        
        return recommendations

# 使用例
async def main():
    """使用例"""
    config = OptimizationConfig(
        max_concurrent_requests=5,
        cache_size=500,
        memory_limit_mb=1024,
        cpu_limit_percent=70
    )
    
    optimizer = SO8TPerformanceOptimizer(config)
    
    # 複数の問題を並列で処理
    problems = [
        "4次元超立方体と2次元平面の交差点の数を求めよ",
        "アインシュタインのパズルを解け",
        "気候変動問題の解決策を提案せよ"
    ]
    
    tasks = []
    for i, problem in enumerate(problems):
        task = optimizer.optimize_performance(problem, {'test_id': i})
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    print("=== パフォーマンス最適化結果 ===")
    for i, result in enumerate(results):
        print(f"問題 {i+1}: {result['solution']['final_answer'][:50]}...")
        print(f"  スコア: {result['verification']['overall_score']:.3f}")
        print(f"  処理時間: {result['performance']['processing_time']:.2f}秒")
    
    # パフォーマンスサマリーを表示
    summary = optimizer.get_performance_summary()
    print(f"\nパフォーマンスサマリー:")
    print(f"  総リクエスト数: {summary['total_requests']}")
    print(f"  成功率: {summary['success_rate']:.3f}")
    print(f"  平均CPU使用率: {summary['average_cpu_usage']:.1f}%")
    print(f"  平均メモリ使用率: {summary['average_memory_usage']:.1f}%")
    print(f"  平均応答時間: {summary['average_response_time']:.2f}秒")
    print(f"  キャッシュサイズ: {summary['cache_size']}")
    
    # 最適化推奨事項を表示
    recommendations = optimizer.get_optimization_recommendations()
    if recommendations:
        print(f"\n最適化推奨事項:")
        for rec in recommendations:
            print(f"  - {rec}")

if __name__ == "__main__":
    asyncio.run(main())
