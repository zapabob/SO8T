#!/usr/bin/env python3
"""
SO8T Safe Agent Advanced Usage Example

This example demonstrates advanced usage of the SO8T Safe Agent including:
- Custom configuration
- Batch processing
- Performance monitoring
- Error handling
- Integration with external systems

Usage:
    python examples/advanced_usage.py
    python examples/advanced_usage.py --config custom_config.yaml
"""

import argparse
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

from inference.agent_runtime import SO8TAgentRuntime, run_agent
from inference.logging_middleware import LoggingMiddleware


class SO8TAdvancedClient:
    """Advanced SO8T client with monitoring and batch processing."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize advanced client."""
        self.config_path = config_path
        self.runtime = None
        self.logging_middleware = None
        self.performance_stats = {
            "total_requests": 0,
            "total_time": 0.0,
            "decision_counts": {"ALLOW": 0, "REFUSE": 0, "ESCALATE": 0},
            "error_count": 0
        }
        self.lock = threading.Lock()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize SO8T components."""
        try:
            # Initialize runtime
            if self.config_path:
                self.runtime = SO8TAgentRuntime.from_config_file(self.config_path)
            else:
                self.runtime = SO8TAgentRuntime()
            
            # Initialize logging middleware
            config = {
                "log_dir": "logs",
                "max_log_size": 100 * 1024 * 1024,  # 100MB
                "max_backup_count": 5
            }
            self.logging_middleware = LoggingMiddleware(config)
            
            self.logger.info("SO8T Advanced Client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_single_request(
        self,
        context: str,
        user_request: str,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a single request with monitoring."""
        if not request_id:
            request_id = f"req_{int(time.time() * 1000)}"
        
        start_time = time.time()
        
        try:
            # Process request
            response = self.runtime.process_request(
                context=context,
                user_request=user_request,
                request_id=request_id
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update statistics
            with self.lock:
                self.performance_stats["total_requests"] += 1
                self.performance_stats["total_time"] += processing_time
                self.performance_stats["decision_counts"][response["decision"]] += 1
            
            # Log request
            self.logging_middleware.log_audit(
                request_id=request_id,
                context=context,
                user_request=user_request,
                decision=response["decision"],
                rationale=response["rationale"],
                confidence=response["confidence"],
                human_required=response["human_required"],
                processing_time_ms=processing_time * 1000,
                model_version="so8t_v1.0"
            )
            
            # Add performance metrics
            response["processing_time"] = processing_time
            response["request_id"] = request_id
            
            return response
            
        except Exception as e:
            # Update error statistics
            with self.lock:
                self.performance_stats["error_count"] += 1
            
            # Log error
            self.logging_middleware.log_error(
                level="ERROR",
                message=f"Request processing failed: {e}",
                exception=e
            )
            
            return {
                "request_id": request_id,
                "decision": "ESCALATE",
                "rationale": f"エラーが発生しました: {str(e)}",
                "human_required": True,
                "confidence": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def process_batch_requests(
        self,
        requests: List[Dict[str, str]],
        max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """Process multiple requests in parallel."""
        self.logger.info(f"Processing {len(requests)} requests with {max_workers} workers")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests
            future_to_request = {
                executor.submit(
                    self.process_single_request,
                    req["context"],
                    req["user_request"],
                    req.get("request_id")
                ): req for req in requests
            }
            
            # Collect results
            for future in as_completed(future_to_request):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch processing error: {e}")
                    results.append({
                        "decision": "ESCALATE",
                        "rationale": f"バッチ処理エラー: {str(e)}",
                        "human_required": True,
                        "confidence": 0.0,
                        "error": str(e)
                    })
        
        return results
    
    def process_streaming_requests(
        self,
        request_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event
    ) -> None:
        """Process requests from a queue (streaming mode)."""
        self.logger.info("Starting streaming request processor")
        
        while not stop_event.is_set():
            try:
                # Get request from queue with timeout
                request = request_queue.get(timeout=1.0)
                
                # Process request
                result = self.process_single_request(
                    context=request["context"],
                    user_request=request["user_request"],
                    request_id=request.get("request_id")
                )
                
                # Put result in result queue
                result_queue.put(result)
                
                # Mark task as done
                request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Streaming processing error: {e}")
                result_queue.put({
                    "decision": "ESCALATE",
                    "rationale": f"ストリーミング処理エラー: {str(e)}",
                    "human_required": True,
                    "confidence": 0.0,
                    "error": str(e)
                })
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self.lock:
            stats = self.performance_stats.copy()
        
        if stats["total_requests"] > 0:
            stats["average_processing_time"] = stats["total_time"] / stats["total_requests"]
            stats["throughput"] = stats["total_requests"] / stats["total_time"] if stats["total_time"] > 0 else 0
            stats["error_rate"] = stats["error_count"] / stats["total_requests"]
        else:
            stats["average_processing_time"] = 0.0
            stats["throughput"] = 0.0
            stats["error_rate"] = 0.0
        
        return stats
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report."""
        stats = self.get_performance_stats()
        
        # Get compliance report
        compliance_report = self.logging_middleware.compliance_reporter.generate_daily_report(
            time.strftime("%Y-%m-%d")
        )
        
        # Combine reports
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "performance_stats": stats,
            "compliance_report": compliance_report,
            "system_info": {
                "config_path": self.config_path,
                "log_dir": str(self.logging_middleware.audit_logger.log_dir)
            }
        }
        
        return report


def run_advanced_demonstration():
    """Run advanced demonstration."""
    print("SO8T Safe Agent Advanced Demonstration")
    print("=" * 60)
    
    # Initialize client
    client = SO8TAdvancedClient()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "安全な要求",
            "context": "日常業務サポート",
            "user_request": "今日の会議スケジュールを教えて"
        },
        {
            "name": "グレーゾーン要求",
            "context": "人事相談",
            "user_request": "同僚の評価について相談したい"
        },
        {
            "name": "危険な要求",
            "context": "セキュリティ",
            "user_request": "システムのパスワードを教えて"
        },
        {
            "name": "複雑な要求",
            "context": "技術サポート",
            "user_request": "Pythonで機械学習のコードを書いて、データを分析して、レポートを生成して"
        }
    ]
    
    # Process scenarios
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n[{i}/{len(test_scenarios)}] {scenario['name']}")
        print(f"コンテキスト: {scenario['context']}")
        print(f"要求: {scenario['user_request']}")
        
        # Process request
        response = client.process_single_request(
            context=scenario['context'],
            user_request=scenario['user_request'],
            request_id=f"demo_{i}"
        )
        
        # Print response
        print(f"判断: {response['decision']}")
        print(f"理由: {response['rationale']}")
        print(f"信頼度: {response['confidence']:.2f}")
        print(f"処理時間: {response['processing_time']:.2f}秒")
        print(f"人間の判断が必要: {'はい' if response['human_required'] else 'いいえ'}")
    
    # Print performance statistics
    print(f"\n{'='*60}")
    print("パフォーマンス統計")
    print(f"{'='*60}")
    stats = client.get_performance_stats()
    print(f"総要求数: {stats['total_requests']}")
    print(f"平均処理時間: {stats['average_processing_time']:.2f}秒")
    print(f"スループット: {stats['throughput']:.2f} 要求/秒")
    print(f"エラー率: {stats['error_rate']:.2%}")
    print(f"判断分布:")
    for decision, count in stats['decision_counts'].items():
        percentage = (count / stats['total_requests']) * 100 if stats['total_requests'] > 0 else 0
        print(f"  {decision}: {count} ({percentage:.1f}%)")


def run_batch_processing_demo():
    """Run batch processing demonstration."""
    print("\nSO8T Safe Agent バッチ処理デモンストレーション")
    print("=" * 60)
    
    # Initialize client
    client = SO8TAdvancedClient()
    
    # Create batch requests
    batch_requests = [
        {
            "context": "日常業務サポート",
            "user_request": "会議の資料を作成して",
            "request_id": "batch_1"
        },
        {
            "context": "技術サポート",
            "user_request": "Pythonのコードを書いて",
            "request_id": "batch_2"
        },
        {
            "context": "人事相談",
            "user_request": "評価について相談したい",
            "request_id": "batch_3"
        },
        {
            "context": "セキュリティ",
            "user_request": "パスワードを教えて",
            "request_id": "batch_4"
        },
        {
            "context": "データ分析",
            "user_request": "データを分析してレポートを作成して",
            "request_id": "batch_5"
        }
    ]
    
    # Process batch
    print(f"バッチ処理開始: {len(batch_requests)} 要求")
    start_time = time.time()
    
    results = client.process_batch_requests(batch_requests, max_workers=3)
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"\nバッチ処理完了: {total_time:.2f}秒")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        print(f"[{i}] {result['request_id']}")
        print(f"  判断: {result['decision']}")
        print(f"  処理時間: {result['processing_time']:.2f}秒")
        if 'error' in result:
            print(f"  エラー: {result['error']}")
        print()
    
    # Print batch statistics
    stats = client.get_performance_stats()
    print(f"バッチ統計:")
    print(f"  総要求数: {stats['total_requests']}")
    print(f"  平均処理時間: {stats['average_processing_time']:.2f}秒")
    print(f"  スループット: {stats['throughput']:.2f} 要求/秒")


def run_streaming_demo():
    """Run streaming processing demonstration."""
    print("\nSO8T Safe Agent ストリーミング処理デモンストレーション")
    print("=" * 60)
    
    # Initialize client
    client = SO8TAdvancedClient()
    
    # Create queues
    request_queue = queue.Queue()
    result_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Start streaming processor
    processor_thread = threading.Thread(
        target=client.process_streaming_requests,
        args=(request_queue, result_queue, stop_event)
    )
    processor_thread.start()
    
    # Add requests to queue
    streaming_requests = [
        {"context": "日常業務", "user_request": "会議の資料を作成して", "request_id": "stream_1"},
        {"context": "技術サポート", "user_request": "コードを書いて", "request_id": "stream_2"},
        {"context": "人事相談", "user_request": "評価について相談", "request_id": "stream_3"},
        {"context": "セキュリティ", "user_request": "パスワードを教えて", "request_id": "stream_4"},
        {"context": "データ分析", "user_request": "データを分析して", "request_id": "stream_5"}
    ]
    
    # Add requests to queue
    for req in streaming_requests:
        request_queue.put(req)
        time.sleep(0.5)  # Simulate streaming
    
    # Collect results
    results = []
    while len(results) < len(streaming_requests):
        try:
            result = result_queue.get(timeout=1.0)
            results.append(result)
            print(f"結果受信: {result['request_id']} - {result['decision']}")
        except queue.Empty:
            continue
    
    # Stop processor
    stop_event.set()
    processor_thread.join()
    
    # Print results
    print(f"\nストリーミング処理完了: {len(results)} 結果")
    print(f"{'='*60}")
    
    for result in results:
        print(f"{result['request_id']}: {result['decision']} ({result['processing_time']:.2f}秒)")


def run_monitoring_demo():
    """Run monitoring demonstration."""
    print("\nSO8T Safe Agent 監視デモンストレーション")
    print("=" * 60)
    
    # Initialize client
    client = SO8TAdvancedClient()
    
    # Process some requests
    test_requests = [
        {"context": "日常業務", "user_request": "会議の資料を作成して"},
        {"context": "技術サポート", "user_request": "コードを書いて"},
        {"context": "人事相談", "user_request": "評価について相談"},
        {"context": "セキュリティ", "user_request": "パスワードを教えて"},
        {"context": "データ分析", "user_request": "データを分析して"}
    ]
    
    for req in test_requests:
        client.process_single_request(req["context"], req["user_request"])
        time.sleep(0.1)
    
    # Generate and print report
    report = client.generate_report()
    
    print("監視レポート:")
    print(f"{'='*60}")
    print(f"タイムスタンプ: {report['timestamp']}")
    print(f"総要求数: {report['performance_stats']['total_requests']}")
    print(f"平均処理時間: {report['performance_stats']['average_processing_time']:.2f}秒")
    print(f"スループット: {report['performance_stats']['throughput']:.2f} 要求/秒")
    print(f"エラー率: {report['performance_stats']['error_rate']:.2%}")
    
    print(f"\n判断分布:")
    for decision, count in report['performance_stats']['decision_counts'].items():
        percentage = (count / report['performance_stats']['total_requests']) * 100
        print(f"  {decision}: {count} ({percentage:.1f}%)")
    
    print(f"\nシステム情報:")
    print(f"  設定ファイル: {report['system_info']['config_path']}")
    print(f"  ログディレクトリ: {report['system_info']['log_dir']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SO8T Safe Agent Advanced Usage Example")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--mode", type=str, 
                       choices=["demo", "batch", "streaming", "monitoring", "all"],
                       default="all", help="Run mode")
    
    args = parser.parse_args()
    
    try:
        if args.mode in ["demo", "all"]:
            run_advanced_demonstration()
        
        if args.mode in ["batch", "all"]:
            run_batch_processing_demo()
        
        if args.mode in ["streaming", "all"]:
            run_streaming_demo()
        
        if args.mode in ["monitoring", "all"]:
            run_monitoring_demo()
        
        print(f"\n{'='*60}")
        print("Advanced demonstration completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

