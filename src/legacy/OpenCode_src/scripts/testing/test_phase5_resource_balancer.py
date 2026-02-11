#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 9: Phase 5動作確認テスト

Phase 5（リソースバランス管理）の動作確認を行います。

Usage:
    python scripts/testing/test_phase5_resource_balancer.py
"""

import os
import sys
import json
import logging
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase5ResourceBalancerTester:
    """Phase 5リソースバランサーテスター"""
    
    def __init__(self):
        """初期化"""
        self.test_results = {}
        self.test_dir = PROJECT_ROOT / "test_results" / "phase5_resource_balancer"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("Phase 5 Resource Balancer Tester Initialized")
        logger.info("="*80)
        logger.info(f"Test directory: {self.test_dir}")
    
    def test_nvidia_smi_available(self) -> bool:
        """nvidia-smiの利用可能性確認"""
        logger.info("="*80)
        logger.info("TEST 1: nvidia-smi Availability")
        logger.info("="*80)
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                logger.info("[OK] nvidia-smi is available")
                logger.info(f"Version info: {result.stdout.split(chr(10))[0]}")
                self.test_results['nvidia_smi'] = {
                    'status': 'available',
                    'version': result.stdout.split(chr(10))[0] if result.stdout else 'unknown'
                }
                return True
            else:
                logger.warning("[WARNING] nvidia-smi returned non-zero exit code")
                self.test_results['nvidia_smi'] = {
                    'status': 'error',
                    'returncode': result.returncode
                }
                return False
                
        except FileNotFoundError:
            logger.warning("[WARNING] nvidia-smi not found")
            logger.info("GPU monitoring will not be available")
            self.test_results['nvidia_smi'] = {
                'status': 'not_found',
                'note': 'GPU monitoring will not be available'
            }
            return False
        except Exception as e:
            logger.error(f"[ERROR] Failed to check nvidia-smi: {e}")
            self.test_results['nvidia_smi'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def test_psutil_available(self) -> bool:
        """psutilの利用可能性確認"""
        logger.info("="*80)
        logger.info("TEST 2: psutil Availability")
        logger.info("="*80)
        
        try:
            import psutil
            
            # バージョン確認
            version = psutil.__version__
            logger.info(f"[OK] psutil is available (version: {version})")
            
            # CPU/メモリ情報取得テスト
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            logger.info(f"CPU usage: {cpu_percent}%")
            logger.info(f"Memory usage: {memory.percent}%")
            
            self.test_results['psutil'] = {
                'status': 'available',
                'version': version,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent
            }
            return True
            
        except ImportError:
            logger.warning("[WARNING] psutil not installed")
            logger.info("Install with: pip install psutil")
            self.test_results['psutil'] = {
                'status': 'not_installed',
                'note': 'Install with: pip install psutil'
            }
            return False
        except Exception as e:
            logger.error(f"[ERROR] Failed to check psutil: {e}")
            self.test_results['psutil'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def test_resource_balancer_script_structure(self) -> bool:
        """リソースバランサースクリプトの構造確認"""
        logger.info("="*80)
        logger.info("TEST 3: Resource Balancer Script Structure")
        logger.info("="*80)
        
        try:
            balancer_script = PROJECT_ROOT / "scripts" / "utils" / "resource_balancer.py"
            assert balancer_script.exists(), f"Resource balancer script not found: {balancer_script}"
            
            # モジュールインポートテスト
            import importlib.util
            spec = importlib.util.spec_from_file_location("resource_balancer", balancer_script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # クラスの存在確認
            assert hasattr(module, 'ResourceBalancer'), "ResourceBalancer class not found"
            assert hasattr(module, 'ResourceMetrics'), "ResourceMetrics class not found"
            
            # メソッドの存在確認
            balancer_class = module.ResourceBalancer
            required_methods = [
                'get_gpu_metrics',
                'get_cpu_metrics',
                'get_memory_metrics',
                'get_current_metrics',
                'check_thresholds',
                'adjust_resources',
                'start_monitoring',
                'stop_monitoring',
                'save_metrics_history',
                'get_metrics_summary'
            ]
            
            for method_name in required_methods:
                assert hasattr(balancer_class, method_name), f"Method {method_name} not found"
            
            logger.info("[OK] Resource balancer script structure is valid")
            self.test_results['resource_balancer_script'] = {
                'status': 'valid',
                'script_path': str(balancer_script),
                'methods_found': required_methods
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Resource balancer script structure check failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['resource_balancer_script'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_resource_balancer_initialization(self) -> bool:
        """リソースバランサーの初期化テスト"""
        logger.info("="*80)
        logger.info("TEST 4: Resource Balancer Initialization")
        logger.info("="*80)
        
        try:
            from scripts.utils.resource_balancer import ResourceBalancer
            
            # 設定ファイル読み込み
            config_path = PROJECT_ROOT / "configs" / "complete_automated_ab_pipeline.yaml"
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # リソースバランサー初期化
            balancer = ResourceBalancer(config)
            
            # 設定確認
            assert balancer.gpu_threshold > 0, "GPU threshold should be positive"
            assert balancer.memory_threshold > 0, "Memory threshold should be positive"
            assert balancer.cpu_threshold > 0, "CPU threshold should be positive"
            assert balancer.monitor_interval > 0, "Monitor interval should be positive"
            
            logger.info("[OK] Resource balancer initialization successful")
            logger.info(f"GPU threshold: {balancer.gpu_threshold}")
            logger.info(f"Memory threshold: {balancer.memory_threshold}")
            logger.info(f"CPU threshold: {balancer.cpu_threshold}")
            logger.info(f"Monitor interval: {balancer.monitor_interval}s")
            
            self.test_results['resource_balancer_initialization'] = {
                'status': 'success',
                'gpu_threshold': balancer.gpu_threshold,
                'memory_threshold': balancer.memory_threshold,
                'cpu_threshold': balancer.cpu_threshold,
                'monitor_interval': balancer.monitor_interval,
                'auto_adjust': balancer.auto_adjust
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Resource balancer initialization test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['resource_balancer_initialization'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_metrics_collection(self) -> bool:
        """メトリクス収集テスト"""
        logger.info("="*80)
        logger.info("TEST 5: Metrics Collection")
        logger.info("="*80)
        
        try:
            from scripts.utils.resource_balancer import ResourceBalancer
            
            # 設定ファイル読み込み
            config_path = PROJECT_ROOT / "configs" / "complete_automated_ab_pipeline.yaml"
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # リソースバランサー初期化
            balancer = ResourceBalancer(config)
            
            # メトリクス取得
            metrics = balancer.get_current_metrics()
            
            # メトリクス構造確認
            assert hasattr(metrics, 'gpu_usage'), "Metrics should have gpu_usage"
            assert hasattr(metrics, 'gpu_memory_usage'), "Metrics should have gpu_memory_usage"
            assert hasattr(metrics, 'cpu_usage'), "Metrics should have cpu_usage"
            assert hasattr(metrics, 'memory_usage'), "Metrics should have memory_usage"
            assert hasattr(metrics, 'timestamp'), "Metrics should have timestamp"
            
            # 値の範囲確認
            assert 0.0 <= metrics.gpu_usage <= 1.0, f"GPU usage should be 0-1, got {metrics.gpu_usage}"
            assert 0.0 <= metrics.gpu_memory_usage <= 1.0, f"GPU memory usage should be 0-1, got {metrics.gpu_memory_usage}"
            assert 0.0 <= metrics.cpu_usage <= 1.0, f"CPU usage should be 0-1, got {metrics.cpu_usage}"
            assert 0.0 <= metrics.memory_usage <= 1.0, f"Memory usage should be 0-1, got {metrics.memory_usage}"
            
            logger.info("[OK] Metrics collection successful")
            logger.info(f"GPU usage: {metrics.gpu_usage:.2%}")
            logger.info(f"GPU memory usage: {metrics.gpu_memory_usage:.2%}")
            logger.info(f"CPU usage: {metrics.cpu_usage:.2%}")
            logger.info(f"Memory usage: {metrics.memory_usage:.2%}")
            
            # 履歴確認
            assert len(balancer.metrics_history) > 0, "Metrics history should have at least one entry"
            
            self.test_results['metrics_collection'] = {
                'status': 'success',
                'gpu_usage': metrics.gpu_usage,
                'gpu_memory_usage': metrics.gpu_memory_usage,
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'history_count': len(balancer.metrics_history)
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Metrics collection test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['metrics_collection'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_threshold_checking(self) -> bool:
        """閾値チェックテスト"""
        logger.info("="*80)
        logger.info("TEST 6: Threshold Checking")
        logger.info("="*80)
        
        try:
            from scripts.utils.resource_balancer import ResourceBalancer, ResourceMetrics
            
            # 設定ファイル読み込み
            config_path = PROJECT_ROOT / "configs" / "complete_automated_ab_pipeline.yaml"
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # リソースバランサー初期化
            balancer = ResourceBalancer(config)
            
            # 閾値以下のメトリクス
            low_metrics = ResourceMetrics(
                gpu_usage=0.5,
                gpu_memory_usage=0.5,
                cpu_usage=0.5,
                memory_usage=0.5
            )
            violations_low = balancer.check_thresholds(low_metrics)
            assert not any(violations_low.values()), "No violations should be detected for low metrics"
            
            # 閾値超過のメトリクス
            high_metrics = ResourceMetrics(
                gpu_usage=0.95,
                gpu_memory_usage=0.95,
                cpu_usage=0.85,
                memory_usage=0.95
            )
            violations_high = balancer.check_thresholds(high_metrics)
            assert any(violations_high.values()), "Violations should be detected for high metrics"
            
            logger.info("[OK] Threshold checking successful")
            logger.info(f"Low metrics violations: {violations_low}")
            logger.info(f"High metrics violations: {violations_high}")
            
            self.test_results['threshold_checking'] = {
                'status': 'success',
                'low_metrics_violations': violations_low,
                'high_metrics_violations': violations_high
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Threshold checking test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['threshold_checking'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_monitoring_control(self) -> bool:
        """監視制御テスト"""
        logger.info("="*80)
        logger.info("TEST 7: Monitoring Control")
        logger.info("="*80)
        
        try:
            from scripts.utils.resource_balancer import ResourceBalancer
            
            # 設定ファイル読み込み
            config_path = PROJECT_ROOT / "configs" / "complete_automated_ab_pipeline.yaml"
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # リソースバランサー初期化
            balancer = ResourceBalancer(config)
            
            # 監視開始
            assert not balancer.monitoring, "Monitoring should not be active initially"
            balancer.start_monitoring()
            assert balancer.monitoring, "Monitoring should be active after start"
            assert balancer.monitor_thread is not None, "Monitor thread should be created"
            assert balancer.monitor_thread.is_alive(), "Monitor thread should be alive"
            
            # 少し待機
            time.sleep(2)
            
            # 監視停止
            balancer.stop_monitoring()
            assert not balancer.monitoring, "Monitoring should not be active after stop"
            
            logger.info("[OK] Monitoring control successful")
            self.test_results['monitoring_control'] = {
                'status': 'success',
                'start_successful': True,
                'stop_successful': True
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Monitoring control test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['monitoring_control'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_metrics_history_saving(self) -> bool:
        """メトリクス履歴保存テスト"""
        logger.info("="*80)
        logger.info("TEST 8: Metrics History Saving")
        logger.info("="*80)
        
        try:
            from scripts.utils.resource_balancer import ResourceBalancer
            
            # 設定ファイル読み込み
            config_path = PROJECT_ROOT / "configs" / "complete_automated_ab_pipeline.yaml"
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # リソースバランサー初期化
            balancer = ResourceBalancer(config)
            
            # いくつかのメトリクスを収集
            for _ in range(5):
                balancer.get_current_metrics()
                time.sleep(0.1)
            
            # 履歴保存
            history_file = self.test_dir / "test_metrics_history.json"
            balancer.save_metrics_history(history_file)
            
            # ファイル確認
            assert history_file.exists(), "History file should be created"
            
            # 内容確認
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            assert isinstance(history_data, list), "History data should be a list"
            assert len(history_data) > 0, "History data should have entries"
            
            # エントリ構造確認
            for entry in history_data:
                assert 'gpu_usage' in entry, "Entry should have gpu_usage"
                assert 'cpu_usage' in entry, "Entry should have cpu_usage"
                assert 'memory_usage' in entry, "Entry should have memory_usage"
                assert 'timestamp' in entry, "Entry should have timestamp"
            
            logger.info("[OK] Metrics history saving successful")
            logger.info(f"History file: {history_file}")
            logger.info(f"Entries saved: {len(history_data)}")
            
            self.test_results['metrics_history_saving'] = {
                'status': 'success',
                'history_file': str(history_file),
                'entries_count': len(history_data)
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Metrics history saving test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['metrics_history_saving'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """全テスト実行"""
        logger.info("="*80)
        logger.info("Running All Phase 5 Resource Balancer Tests")
        logger.info("="*80)
        
        tests = [
            ("nvidia-smi Availability", self.test_nvidia_smi_available),
            ("psutil Availability", self.test_psutil_available),
            ("Resource Balancer Script Structure", self.test_resource_balancer_script_structure),
            ("Resource Balancer Initialization", self.test_resource_balancer_initialization),
            ("Metrics Collection", self.test_metrics_collection),
            ("Threshold Checking", self.test_threshold_checking),
            ("Monitoring Control", self.test_monitoring_control),
            ("Metrics History Saving", self.test_metrics_history_saving)
        ]
        
        results = {}
        passed = 0
        failed = 0
        skipped = 0
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*80}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*80}\n")
            
            try:
                success = test_func()
                if success:
                    passed += 1
                    results[test_name] = "PASSED"
                else:
                    # skippedかどうか確認
                    test_key = test_name.lower().replace(' ', '_').replace('-', '_')
                    if self.test_results.get(test_key, {}).get('status') in ['not_found', 'not_installed', 'skipped']:
                        skipped += 1
                        results[test_name] = "SKIPPED"
                    else:
                        failed += 1
                        results[test_name] = "FAILED"
            except Exception as e:
                failed += 1
                results[test_name] = f"FAILED: {e}"
                logger.error(f"Test {test_name} raised exception: {e}")
        
        # 結果サマリー
        logger.info("="*80)
        logger.info("Test Summary")
        logger.info("="*80)
        logger.info(f"Total tests: {len(tests)}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped: {skipped}")
        logger.info("="*80)
        
        # 結果をJSONファイルに保存
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(tests),
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'results': results,
            'detailed_results': self.test_results
        }
        
        result_file = self.test_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test results saved to: {result_file}")
        
        return summary


def main():
    """メイン関数"""
    tester = Phase5ResourceBalancerTester()
    summary = tester.run_all_tests()
    
    # 音声通知
    audio_file = PROJECT_ROOT / ".cursor" / "marisa_owattaze.wav"
    if audio_file.exists():
        try:
            import subprocess
            ps_cmd = f"""
            if (Test-Path '{audio_file}') {{
                Add-Type -AssemblyName System.Windows.Forms
                $player = New-Object System.Media.SoundPlayer '{audio_file}'
                $player.PlaySync()
                Write-Host '[OK] Audio notification played' -ForegroundColor Green
            }}
            """
            subprocess.run(
                ["powershell", "-Command", ps_cmd],
                cwd=str(PROJECT_ROOT),
                check=False
            )
        except Exception as e:
            logger.warning(f"Failed to play audio: {e}")
    
    return 0 if summary['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())



















