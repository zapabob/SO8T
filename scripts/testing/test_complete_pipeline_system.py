#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T完全パイプラインシステム統合テストスクリプト

1. 進捗管理システムの動作確認
2. チェックリスト自動更新の動作確認
3. 統合パイプラインの動作確認（モック実行）
4. 自動起動機能の動作確認
5. エラーハンドリングの検証
6. パフォーマンス最適化設定の確認

Usage:
    python scripts/testing/test_complete_pipeline_system.py
"""

import os
import sys
import json
import logging
import time
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# テスト対象のインポート
from scripts.utils.progress_manager import ProgressManager
from scripts.utils.checklist_updater import ChecklistUpdater

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompletePipelineSystemTester:
    """完全パイプラインシステム統合テスター"""
    
    def __init__(self):
        """初期化"""
        self.test_results = {}
        self.test_dir = PROJECT_ROOT / "test_results" / "complete_pipeline_system"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("Complete Pipeline System Tester Initialized")
        logger.info("="*80)
        logger.info(f"Test directory: {self.test_dir}")
    
    def test_progress_manager(self) -> bool:
        """進捗管理システムの動作確認"""
        logger.info("="*80)
        logger.info("TEST 1: Progress Manager")
        logger.info("="*80)
        
        try:
            session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # ProgressManager初期化（テスト用に短い間隔）
            manager = ProgressManager(session_id=session_id, log_interval=10)  # 10秒間隔
            
            # フェーズ状態更新テスト
            manager.update_phase_status("phase1", "running", progress=0.3)
            time.sleep(1)
            manager.update_phase_status("phase1", "completed", progress=1.0, metrics={"accuracy": 0.95})
            
            manager.update_phase_status("phase2", "running", progress=0.5)
            time.sleep(1)
            
            # ログ生成テスト
            manager.log_progress()
            
            # サマリー取得テスト
            summary = manager.get_progress_summary()
            
            # 結果確認
            assert summary['total_phases'] == 2, f"Expected 2 phases, got {summary['total_phases']}"
            assert summary['completed'] == 1, f"Expected 1 completed, got {summary['completed']}"
            assert summary['running'] == 1, f"Expected 1 running, got {summary['running']}"
            
            # ログファイル確認
            logs_dir = PROJECT_ROOT / "_docs" / "progress_logs"
            log_files = list(logs_dir.glob(f"{session_id}_*.md"))
            assert len(log_files) > 0, "No log files generated"
            
            logger.info("[OK] Progress Manager test passed")
            self.test_results['progress_manager'] = {
                'status': 'passed',
                'session_id': session_id,
                'log_files': [str(f) for f in log_files],
                'summary': summary
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Progress Manager test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['progress_manager'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_checklist_updater(self) -> bool:
        """チェックリスト自動更新の動作確認"""
        logger.info("="*80)
        logger.info("TEST 2: Checklist Updater")
        logger.info("="*80)
        
        try:
            # テスト用チェックリストファイル
            test_checklist_path = self.test_dir / "test_checklist.md"
            
            # ChecklistUpdater初期化
            updater = ChecklistUpdater(checklist_path=test_checklist_path)
            
            # フェーズ完了更新テスト
            updater.update_phase_completion(
                "phase1",
                status="completed",
                metrics={"accuracy": 0.95, "f1_score": 0.92}
            )
            
            time.sleep(1)
            
            updater.update_phase_completion(
                "phase2",
                status="running"
            )
            
            # チェックリスト内容確認
            checklist_content = updater.generate_checklist()
            
            assert "[x]" in checklist_content, "Checkmark not found in checklist"
            assert "Phase 1" in checklist_content or "phase 1" in checklist_content.lower(), "Phase 1 not found in checklist"
            assert "Phase 2" in checklist_content or "phase 2" in checklist_content.lower(), "Phase 2 not found in checklist"
            assert "完了" in checklist_content or "completed" in checklist_content.lower(), "Completed status not found in checklist"
            
            logger.info("[OK] Checklist Updater test passed")
            self.test_results['checklist_updater'] = {
                'status': 'passed',
                'checklist_path': str(test_checklist_path),
                'checklist_length': len(checklist_content)
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Checklist Updater test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['checklist_updater'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_integrated_pipeline_structure(self) -> bool:
        """統合パイプラインの構造確認（モック実行）"""
        logger.info("="*80)
        logger.info("TEST 3: Integrated Pipeline Structure")
        logger.info("="*80)
        
        try:
            # 統合パイプラインスクリプトの存在確認
            pipeline_script = PROJECT_ROOT / "scripts" / "pipelines" / "run_complete_so8t_ab_pipeline.py"
            assert pipeline_script.exists(), f"Pipeline script not found: {pipeline_script}"
            
            # 設定ファイルの存在確認
            config_file = PROJECT_ROOT / "configs" / "ab_test_so8t_complete.yaml"
            assert config_file.exists(), f"Config file not found: {config_file}"
            
            # モジュールインポートテスト
            import importlib.util
            spec = importlib.util.spec_from_file_location("complete_pipeline", pipeline_script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # クラスの存在確認
            assert hasattr(module, 'CompleteSO8TABPipeline'), "CompleteSO8TABPipeline class not found"
            
            # メソッドの存在確認
            pipeline_class = module.CompleteSO8TABPipeline
            required_methods = [
                'run_phase1_data_pipeline',
                'run_phase2_training',
                'run_phase3_gguf_conversion',
                'run_phase4_ab_test',
                'run_phase5_visualization',
                'run_complete_pipeline'
            ]
            
            for method_name in required_methods:
                assert hasattr(pipeline_class, method_name), f"Method {method_name} not found"
            
            logger.info("[OK] Integrated Pipeline Structure test passed")
            self.test_results['integrated_pipeline_structure'] = {
                'status': 'passed',
                'script_path': str(pipeline_script),
                'config_path': str(config_file),
                'methods_found': required_methods
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Integrated Pipeline Structure test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['integrated_pipeline_structure'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_auto_start_functionality(self) -> bool:
        """自動起動機能の動作確認"""
        logger.info("="*80)
        logger.info("TEST 4: Auto Start Functionality")
        logger.info("="*80)
        
        try:
            # 自動起動スクリプトの存在確認
            auto_start_script = PROJECT_ROOT / "scripts" / "pipelines" / "auto_start_complete_pipeline.py"
            assert auto_start_script.exists(), f"Auto start script not found: {auto_start_script}"
            
            # モジュールインポートテスト
            import importlib.util
            spec = importlib.util.spec_from_file_location("auto_start", auto_start_script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # クラスの存在確認
            assert hasattr(module, 'AutoStartCompletePipeline'), "AutoStartCompletePipeline class not found"
            
            # メソッドの存在確認
            auto_start_class = module.AutoStartCompletePipeline
            required_methods = [
                'setup_auto_start',
                'check_and_resume',
                'run_pipeline_with_progress'
            ]
            
            for method_name in required_methods:
                assert hasattr(auto_start_class, method_name), f"Method {method_name} not found"
            
            # チェックポイント検出機能のテスト（モック）
            auto_start = auto_start_class()
            checkpoint_data = auto_start.check_and_resume()
            
            logger.info("[OK] Auto Start Functionality test passed")
            self.test_results['auto_start_functionality'] = {
                'status': 'passed',
                'script_path': str(auto_start_script),
                'methods_found': required_methods,
                'checkpoint_detection': checkpoint_data is not None
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Auto Start Functionality test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['auto_start_functionality'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_error_handling(self) -> bool:
        """エラーハンドリングの検証"""
        logger.info("="*80)
        logger.info("TEST 5: Error Handling")
        logger.info("="*80)
        
        try:
            # ProgressManagerのエラーハンドリングテスト
            session_id = f"test_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            manager = ProgressManager(session_id=session_id, log_interval=10)
            
            # 無効な進捗値のテスト
            manager.update_phase_status("phase1", "running", progress=-0.1)  # 負の値
            assert manager.phases["phase1"].progress == 0.0, "Negative progress should be clamped to 0.0"
            
            manager.update_phase_status("phase1", "running", progress=1.5)  # 1より大きい値
            assert manager.phases["phase1"].progress == 1.0, "Progress > 1.0 should be clamped to 1.0"
            
            # エラー状態のテスト
            manager.update_phase_status("phase1", "failed", progress=0.5, error_message="Test error")
            assert manager.phases["phase1"].status == "failed", "Status should be failed"
            assert manager.phases["phase1"].error_message == "Test error", "Error message should be set"
            
            # ChecklistUpdaterのエラーハンドリングテスト
            test_checklist_path = self.test_dir / "test_checklist_error.md"
            updater = ChecklistUpdater(checklist_path=test_checklist_path)
            
            # 存在しないフェーズの更新テスト（エラーにならないことを確認）
            try:
                updater.update_phase_completion("nonexistent_phase", status="completed")
                # エラーが発生しないことを確認
            except Exception as e:
                logger.warning(f"Unexpected error: {e}")
            
            logger.info("[OK] Error Handling test passed")
            self.test_results['error_handling'] = {
                'status': 'passed',
                'progress_clamping': True,
                'error_state': True
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Error Handling test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['error_handling'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_performance_optimization(self) -> bool:
        """パフォーマンス最適化設定の確認"""
        logger.info("="*80)
        logger.info("TEST 6: Performance Optimization")
        logger.info("="*80)
        
        try:
            # 設定ファイルの読み込み
            config_file = PROJECT_ROOT / "configs" / "ab_test_so8t_complete.yaml"
            assert config_file.exists(), f"Config file not found: {config_file}"
            
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 最適化設定の確認
            optimization = config.get('optimization', {})
            
            required_settings = [
                'batch_size',
                'gradient_accumulation_steps',
                'cpu_offload',
                'mixed_precision',
                'gradient_checkpointing'
            ]
            
            for setting in required_settings:
                assert setting in optimization, f"Optimization setting {setting} not found"
            
            # RTX3060/32GB最適化設定の確認
            assert optimization['batch_size'] == 4, f"Expected batch_size=4, got {optimization['batch_size']}"
            assert optimization['gradient_accumulation_steps'] == 4, f"Expected gradient_accumulation_steps=4, got {optimization['gradient_accumulation_steps']}"
            assert optimization['cpu_offload'] == True, f"Expected cpu_offload=True, got {optimization['cpu_offload']}"
            assert optimization['mixed_precision'] == True, f"Expected mixed_precision=True, got {optimization['mixed_precision']}"
            assert optimization['gradient_checkpointing'] == True, f"Expected gradient_checkpointing=True, got {optimization['gradient_checkpointing']}"
            
            logger.info("[OK] Performance Optimization test passed")
            self.test_results['performance_optimization'] = {
                'status': 'passed',
                'config_path': str(config_file),
                'optimization_settings': optimization
            }
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Performance Optimization test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['performance_optimization'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """全テスト実行"""
        logger.info("="*80)
        logger.info("Running All Tests")
        logger.info("="*80)
        
        tests = [
            ("Progress Manager", self.test_progress_manager),
            ("Checklist Updater", self.test_checklist_updater),
            ("Integrated Pipeline Structure", self.test_integrated_pipeline_structure),
            ("Auto Start Functionality", self.test_auto_start_functionality),
            ("Error Handling", self.test_error_handling),
            ("Performance Optimization", self.test_performance_optimization)
        ]
        
        results = {}
        passed = 0
        failed = 0
        
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
        logger.info("="*80)
        
        # 結果をJSONファイルに保存
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(tests),
            'passed': passed,
            'failed': failed,
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
    tester = CompletePipelineSystemTester()
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

