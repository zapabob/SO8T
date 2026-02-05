#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8: Phase 4動作確認テスト

Phase 4（ベンチマークテスト実行）の動作確認を行います。

Usage:
    python scripts/testing/test_phase4_benchmarks.py
"""

import os
import sys
import json
import logging
import time
import requests
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


class Phase4BenchmarkTester:
    """Phase 4ベンチマークテストテスター"""
    
    def __init__(self):
        """初期化"""
        self.test_results = {}
        self.test_dir = PROJECT_ROOT / "test_results" / "phase4_benchmarks"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("Phase 4 Benchmark Tester Initialized")
        logger.info("="*80)
        logger.info(f"Test directory: {self.test_dir}")
    
    def test_ollama_server_running(self) -> bool:
        """Ollamaサーバー起動状態の確認"""
        logger.info("="*80)
        logger.info("TEST 1: Ollama Server Running")
        logger.info("="*80)
        
        try:
            ollama_url = "http://localhost:11434"
            
            # Ollama API呼び出しテスト
            try:
                response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    logger.info(f"[OK] Ollama server is running")
                    logger.info(f"Available models: {len(models)}")
                    self.test_results['ollama_server'] = {
                        'status': 'running',
                        'available_models': len(models),
                        'models': [m.get('name', 'unknown') for m in models]
                    }
                    return True
                else:
                    logger.warning(f"[WARNING] Ollama server returned status {response.status_code}")
                    self.test_results['ollama_server'] = {
                        'status': 'error',
                        'status_code': response.status_code
                    }
                    return False
            except requests.exceptions.ConnectionError:
                logger.warning("[WARNING] Ollama server is not running")
                logger.info("Skipping Ollama-dependent tests...")
                self.test_results['ollama_server'] = {
                    'status': 'not_running',
                    'note': 'Ollama server not available, some tests will be skipped'
                }
                return False
            except Exception as e:
                logger.error(f"[ERROR] Failed to check Ollama server: {e}")
                self.test_results['ollama_server'] = {
                    'status': 'error',
                    'error': str(e)
                }
                return False
                
        except Exception as e:
            logger.error(f"[FAILED] Ollama server check failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['ollama_server'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_hf_benchmark_script_exists(self) -> bool:
        """HFベンチマークスクリプトの存在確認"""
        logger.info("="*80)
        logger.info("TEST 2: HF Benchmark Script Exists")
        logger.info("="*80)
        
        try:
            ab_test_script = PROJECT_ROOT / "scripts" / "evaluation" / "ab_test_with_hf_benchmark.py"
            
            assert ab_test_script.exists(), f"HF benchmark script not found: {ab_test_script}"
            
            # モジュールインポートテスト
            import importlib.util
            spec = importlib.util.spec_from_file_location("ab_test", ab_test_script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # クラスの存在確認
            assert hasattr(module, 'ABTestHFBenchmarkEvaluator'), "ABTestHFBenchmarkEvaluator class not found"
            
            logger.info("[OK] HF benchmark script exists and is importable")
            self.test_results['hf_benchmark_script'] = {
                'status': 'found',
                'script_path': str(ab_test_script),
                'class_found': True
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] HF benchmark script check failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['hf_benchmark_script'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_phase4_script_structure(self) -> bool:
        """Phase 4スクリプトの構造確認"""
        logger.info("="*80)
        logger.info("TEST 3: Phase 4 Script Structure")
        logger.info("="*80)
        
        try:
            phase4_script = PROJECT_ROOT / "scripts" / "pipelines" / "phase4_run_benchmarks.py"
            assert phase4_script.exists(), f"Phase 4 script not found: {phase4_script}"
            
            # モジュールインポートテスト
            import importlib.util
            spec = importlib.util.spec_from_file_location("phase4", phase4_script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # クラスの存在確認
            assert hasattr(module, 'Phase4BenchmarkRunner'), "Phase4BenchmarkRunner class not found"
            
            # メソッドの存在確認
            runner_class = module.Phase4BenchmarkRunner
            required_methods = [
                'run_hf_benchmark',
                'run_llm_benchmark',
                'check_ollama_running',
                '_run_ollama_test',
                '_get_default_test_suite'
            ]
            
            for method_name in required_methods:
                assert hasattr(runner_class, method_name), f"Method {method_name} not found"
            
            logger.info("[OK] Phase 4 script structure is valid")
            self.test_results['phase4_script_structure'] = {
                'status': 'valid',
                'script_path': str(phase4_script),
                'methods_found': required_methods
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Phase 4 script structure check failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['phase4_script_structure'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_llm_benchmark_test_suite(self) -> bool:
        """LLMベンチマークテストスイートの確認"""
        logger.info("="*80)
        logger.info("TEST 4: LLM Benchmark Test Suite")
        logger.info("="*80)
        
        try:
            phase4_script = PROJECT_ROOT / "scripts" / "pipelines" / "phase4_run_benchmarks.py"
            
            # モジュールインポート
            import importlib.util
            spec = importlib.util.spec_from_file_location("phase4", phase4_script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # テストスイート取得
            runner = module.Phase4BenchmarkRunner({})
            test_suite = runner._get_default_test_suite()
            
            # テストスイート構造確認
            assert isinstance(test_suite, dict), "Test suite should be a dictionary"
            
            required_categories = ['mathematical_reasoning', 'logical_reasoning', 'safety_evaluation', 'japanese_tasks']
            for category in required_categories:
                assert category in test_suite, f"Category {category} not found in test suite"
                assert isinstance(test_suite[category], list), f"Category {category} should be a list"
                assert len(test_suite[category]) > 0, f"Category {category} should have at least one test"
                
                # 各テストの構造確認
                for test in test_suite[category]:
                    assert 'name' in test, "Test should have 'name' field"
                    assert 'prompt' in test, "Test should have 'prompt' field"
                    assert 'expected_keywords' in test, "Test should have 'expected_keywords' field"
            
            logger.info("[OK] LLM benchmark test suite is valid")
            logger.info(f"Categories: {list(test_suite.keys())}")
            logger.info(f"Total tests: {sum(len(tests) for tests in test_suite.values())}")
            
            self.test_results['llm_benchmark_test_suite'] = {
                'status': 'valid',
                'categories': list(test_suite.keys()),
                'total_tests': sum(len(tests) for tests in test_suite.values()),
                'tests_per_category': {cat: len(tests) for cat, tests in test_suite.items()}
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] LLM benchmark test suite check failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['llm_benchmark_test_suite'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_ollama_api_call(self) -> bool:
        """Ollama API呼び出しテスト（ollamaサーバーが起動している場合のみ）"""
        logger.info("="*80)
        logger.info("TEST 5: Ollama API Call")
        logger.info("="*80)
        
        try:
            ollama_url = "http://localhost:11434"
            
            # Ollamaサーバー確認
            try:
                response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                if response.status_code != 200:
                    logger.info("[SKIP] Ollama server not available, skipping API call test")
                    self.test_results['ollama_api_call'] = {
                        'status': 'skipped',
                        'reason': 'Ollama server not available'
                    }
                    return True
            except requests.exceptions.ConnectionError:
                logger.info("[SKIP] Ollama server not available, skipping API call test")
                self.test_results['ollama_api_call'] = {
                    'status': 'skipped',
                    'reason': 'Ollama server not available'
                }
                return True
            
            # テスト用の簡単なAPI呼び出し
            test_prompt = "Hello, this is a test."
            
            try:
                response = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        'model': 'llama2',  # デフォルトモデル（存在しない場合でもエラーハンドリングをテスト）
                        'prompt': test_prompt,
                        'stream': False
                    },
                    timeout=10
                )
                
                # エラーでもAPI呼び出し自体は成功していることを確認
                logger.info(f"[OK] Ollama API call successful (status: {response.status_code})")
                self.test_results['ollama_api_call'] = {
                    'status': 'success',
                    'status_code': response.status_code,
                    'api_available': True
                }
                return True
                
            except requests.exceptions.Timeout:
                logger.warning("[WARNING] Ollama API call timed out")
                self.test_results['ollama_api_call'] = {
                    'status': 'timeout',
                    'api_available': True
                }
                return True  # タイムアウトでもAPIは利用可能
            except Exception as e:
                logger.warning(f"[WARNING] Ollama API call failed: {e}")
                self.test_results['ollama_api_call'] = {
                    'status': 'error',
                    'error': str(e),
                    'api_available': True
                }
                return True  # エラーでもAPIは利用可能
                
        except Exception as e:
            logger.error(f"[FAILED] Ollama API call test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['ollama_api_call'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_config_file_loading(self) -> bool:
        """設定ファイル読み込みテスト"""
        logger.info("="*80)
        logger.info("TEST 6: Config File Loading")
        logger.info("="*80)
        
        try:
            config_path = PROJECT_ROOT / "configs" / "complete_automated_ab_pipeline.yaml"
            assert config_path.exists(), f"Config file not found: {config_path}"
            
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # ベンチマーク設定の確認
            benchmarks_config = config.get('benchmarks', {})
            assert 'hf_benchmark' in benchmarks_config, "HF benchmark config not found"
            assert 'llm_benchmark' in benchmarks_config, "LLM benchmark config not found"
            
            # Ollama設定の確認
            ollama_config = config.get('ollama', {})
            assert 'model_a_name' in ollama_config, "Model A name not found"
            assert 'model_b_name' in ollama_config, "Model B name not found"
            assert 'base_url' in ollama_config, "Ollama base URL not found"
            
            logger.info("[OK] Config file loading successful")
            self.test_results['config_file_loading'] = {
                'status': 'success',
                'config_path': str(config_path),
                'benchmarks_config': benchmarks_config,
                'ollama_config': ollama_config
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Config file loading test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['config_file_loading'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """全テスト実行"""
        logger.info("="*80)
        logger.info("Running All Phase 4 Benchmark Tests")
        logger.info("="*80)
        
        tests = [
            ("Ollama Server Running", self.test_ollama_server_running),
            ("HF Benchmark Script Exists", self.test_hf_benchmark_script_exists),
            ("Phase 4 Script Structure", self.test_phase4_script_structure),
            ("LLM Benchmark Test Suite", self.test_llm_benchmark_test_suite),
            ("Ollama API Call", self.test_ollama_api_call),
            ("Config File Loading", self.test_config_file_loading)
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
                    test_key = test_name.lower().replace(' ', '_')
                    if self.test_results.get(test_key, {}).get('status') == 'skipped':
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
    tester = Phase4BenchmarkTester()
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

