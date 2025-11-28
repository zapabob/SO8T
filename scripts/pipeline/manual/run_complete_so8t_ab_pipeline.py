#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T完全パイプラインA/Bテスト統合実行スクリプト

Phase 1-5を統合実行:
- Phase 1: データ収集・前処理パイプライン
- Phase 2: SO(8) Transformer再学習 + ベイズ最適化
- Phase 3: GGUF変換（A/Bモデル）
- Phase 4: A/Bテスト評価 + HFベンチマーク
- Phase 5: 可視化・レポート生成

Usage:
    python scripts/pipelines/run_complete_so8t_ab_pipeline.py --config configs/ab_test_so8t_complete.yaml
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

# 進捗管理システムのインポート
from scripts.utils.progress_manager import ProgressManager
from scripts.utils.checklist_updater import ChecklistUpdater

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/complete_so8t_ab_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CompleteSO8TABPipeline:
    """SO8T完全パイプラインA/Bテスト統合クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 出力ディレクトリ
        self.output_dir = Path(config.get('output_dir', 'eval_results/complete_so8t_ab'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # チェックポイント管理
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/complete_pipeline'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 進捗管理システム
        log_interval = config.get('progress', {}).get('log_interval', 1800)  # 30分
        self.progress_manager = ProgressManager(session_id=self.session_id, log_interval=log_interval)
        self.checklist_updater = ChecklistUpdater()
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        # フェーズ状態
        self.current_phase = None
        self.phase_results = {}
        
        logger.info("="*80)
        logger.info("Complete SO8T A/B Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, saving checkpoint...")
            self._save_checkpoint()
            sys.exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _save_checkpoint(self):
        """チェックポイント保存"""
        checkpoint_data = {
            'session_id': self.session_id,
            'current_phase': self.current_phase,
            'phase_results': self.phase_results,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / f"{self.session_id}_checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self) -> Optional[Dict]:
        """チェックポイント読み込み"""
        checkpoint_path = self.checkpoint_dir / f"{self.session_id}_checkpoint.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def run_phase1_data_pipeline(self) -> Dict[str, Any]:
        """
        Phase 1: データ収集・前処理パイプライン実行
        
        Returns:
            phase_result: フェーズ実行結果
        """
        logger.info("="*80)
        logger.info("PHASE 1: Data Collection and Preprocessing Pipeline")
        logger.info("="*80)
        
        self.current_phase = "phase1"
        self.progress_manager.update_phase_status("phase1", "running", progress=0.0)
        self.checklist_updater.update_phase_completion("phase1", status="running")
        
        start_time = time.time()
        
        try:
            # 設定ファイルパス
            data_config_path = self.config.get('phase1', {}).get('config', 'configs/data_pipeline_config.yaml')
            
            # データパイプライン実行
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "pipelines" / "complete_data_pipeline.py"),
                "--config", data_config_path
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            
            if result.returncode != 0:
                raise RuntimeError(f"Phase 1 failed with return code {result.returncode}")
            
            # 結果を取得
            output_dir = Path(self.config.get('phase1', {}).get('output_dir', 'data/processed'))
            metrics = {
                'output_dir': str(output_dir),
                'status': 'completed'
            }
            
            duration = time.time() - start_time
            
            self.progress_manager.update_phase_status("phase1", "completed", progress=1.0, metrics=metrics)
            self.checklist_updater.update_phase_completion("phase1", status="completed", metrics=metrics)
            
            logger.info(f"[OK] Phase 1 completed in {duration:.2f} seconds")
            
            return {
                'status': 'completed',
                'duration': duration,
                'metrics': metrics,
                'output_dir': output_dir
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            self.progress_manager.update_phase_status("phase1", "failed", progress=0.0, error_message=error_msg)
            self.checklist_updater.update_phase_completion("phase1", status="failed", error_message=error_msg)
            
            logger.error(f"[FAILED] Phase 1 failed: {e}")
            raise
    
    def run_phase2_training(self) -> Dict[str, Any]:
        """
        Phase 2: SO(8) Transformer再学習 + ベイズ最適化実行
        
        Returns:
            phase_result: フェーズ実行結果
        """
        logger.info("="*80)
        logger.info("PHASE 2: SO(8) Transformer Training + Bayesian Optimization")
        logger.info("="*80)
        
        self.current_phase = "phase2"
        self.progress_manager.update_phase_status("phase2", "running", progress=0.0)
        self.checklist_updater.update_phase_completion("phase2", status="running")
        
        start_time = time.time()
        
        try:
            # 設定ファイルパス
            training_config_path = self.config.get('phase2', {}).get('config', 'configs/so8t_bayesian_config.yaml')
            
            # 学習スクリプト実行
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "training" / "train_so8t_with_bayesian.py"),
                "--config", training_config_path
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            
            if result.returncode != 0:
                raise RuntimeError(f"Phase 2 failed with return code {result.returncode}")
            
            # 結果を取得
            checkpoint_dir = Path(self.config.get('phase2', {}).get('checkpoint_dir', 'D:/webdataset/checkpoints/training'))
            metrics = {
                'checkpoint_dir': str(checkpoint_dir),
                'status': 'completed'
            }
            
            duration = time.time() - start_time
            
            self.progress_manager.update_phase_status("phase2", "completed", progress=1.0, metrics=metrics)
            self.checklist_updater.update_phase_completion("phase2", status="completed", metrics=metrics)
            
            logger.info(f"[OK] Phase 2 completed in {duration:.2f} seconds")
            
            return {
                'status': 'completed',
                'duration': duration,
                'metrics': metrics,
                'checkpoint_dir': checkpoint_dir
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            self.progress_manager.update_phase_status("phase2", "failed", progress=0.0, error_message=error_msg)
            self.checklist_updater.update_phase_completion("phase2", status="failed", error_message=error_msg)
            
            logger.error(f"[FAILED] Phase 2 failed: {e}")
            raise
    
    def run_phase3_gguf_conversion(self) -> Dict[str, Any]:
        """
        Phase 3: A/BモデルGGUF変換実行
        
        Returns:
            phase_result: フェーズ実行結果
        """
        logger.info("="*80)
        logger.info("PHASE 3: A/B Model GGUF Conversion")
        logger.info("="*80)
        
        self.current_phase = "phase3"
        self.progress_manager.update_phase_status("phase3", "running", progress=0.0)
        self.checklist_updater.update_phase_completion("phase3", status="running")
        
        start_time = time.time()
        
        try:
            # モデルA/Bのパス取得
            model_a_base = self.config.get('phase3', {}).get('model_a_base')
            model_b_path = self.config.get('phase3', {}).get('model_b_path')
            output_dir = Path(self.config.get('phase3', {}).get('output_dir', 'D:/webdataset/gguf_models'))
            
            if not model_a_base or not model_b_path:
                raise ValueError("model_a_base and model_b_path must be specified in config")
            
            # GGUF変換スクリプト実行
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "conversion" / "convert_ab_models_to_gguf.py"),
                "--model-a-base", model_a_base,
                "--model-b-path", model_b_path,
                "--output-dir", str(output_dir)
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            
            if result.returncode != 0:
                raise RuntimeError(f"Phase 3 failed with return code {result.returncode}")
            
            # 結果を取得
            metrics = {
                'model_a_base': model_a_base,
                'model_b_path': model_b_path,
                'output_dir': str(output_dir),
                'status': 'completed'
            }
            
            duration = time.time() - start_time
            
            self.progress_manager.update_phase_status("phase3", "completed", progress=1.0, metrics=metrics)
            self.checklist_updater.update_phase_completion("phase3", status="completed", metrics=metrics)
            
            logger.info(f"[OK] Phase 3 completed in {duration:.2f} seconds")
            
            return {
                'status': 'completed',
                'duration': duration,
                'metrics': metrics,
                'output_dir': output_dir
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            self.progress_manager.update_phase_status("phase3", "failed", progress=0.0, error_message=error_msg)
            self.checklist_updater.update_phase_completion("phase3", status="failed", error_message=error_msg)
            
            logger.error(f"[FAILED] Phase 3 failed: {e}")
            raise
    
    def run_phase4_ab_test(self) -> Dict[str, Any]:
        """
        Phase 4: A/Bテスト評価 + HFベンチマーク実行
        
        Returns:
            phase_result: フェーズ実行結果
        """
        logger.info("="*80)
        logger.info("PHASE 4: A/B Test Evaluation + HF Benchmark")
        logger.info("="*80)
        
        self.current_phase = "phase4"
        self.progress_manager.update_phase_status("phase4", "running", progress=0.0)
        self.checklist_updater.update_phase_completion("phase4", status="running")
        
        start_time = time.time()
        
        try:
            # モデルA/Bのパス取得
            model_a_path = Path(self.config.get('phase4', {}).get('model_a_path'))
            model_b_path = Path(self.config.get('phase4', {}).get('model_b_path'))
            test_data_path = Path(self.config.get('phase4', {}).get('test_data_path'))
            
            if not model_a_path or not model_b_path or not test_data_path:
                raise ValueError("model_a_path, model_b_path, and test_data_path must be specified in config")
            
            # A/Bテスト評価スクリプト実行
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluation" / "ab_test_with_hf_benchmark.py"),
                "--model-a", str(model_a_path),
                "--model-b", str(model_b_path),
                "--test-data", str(test_data_path),
                "--output-dir", str(self.output_dir / "ab_test_hf_benchmark")
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            
            if result.returncode != 0:
                raise RuntimeError(f"Phase 4 failed with return code {result.returncode}")
            
            # 結果を読み込み
            results_path = self.output_dir / "ab_test_hf_benchmark" / "ab_test_results.json"
            metrics = {}
            if results_path.exists():
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    metrics = {
                        'model_a_accuracy': results.get('model_a', {}).get('accuracy', 0.0),
                        'model_b_accuracy': results.get('model_b', {}).get('accuracy', 0.0),
                        'improvement': results.get('model_b', {}).get('accuracy', 0.0) - results.get('model_a', {}).get('accuracy', 0.0),
                        'status': 'completed'
                    }
            
            duration = time.time() - start_time
            
            self.progress_manager.update_phase_status("phase4", "completed", progress=1.0, metrics=metrics)
            self.checklist_updater.update_phase_completion("phase4", status="completed", metrics=metrics)
            
            logger.info(f"[OK] Phase 4 completed in {duration:.2f} seconds")
            
            return {
                'status': 'completed',
                'duration': duration,
                'metrics': metrics,
                'results_path': results_path
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            self.progress_manager.update_phase_status("phase4", "failed", progress=0.0, error_message=error_msg)
            self.checklist_updater.update_phase_completion("phase4", status="failed", error_message=error_msg)
            
            logger.error(f"[FAILED] Phase 4 failed: {e}")
            raise
    
    def run_phase5_visualization(self) -> Dict[str, Any]:
        """
        Phase 5: 可視化・レポート生成実行
        
        Returns:
            phase_result: フェーズ実行結果
        """
        logger.info("="*80)
        logger.info("PHASE 5: Visualization and Report Generation")
        logger.info("="*80)
        
        self.current_phase = "phase5"
        self.progress_manager.update_phase_status("phase5", "running", progress=0.0)
        self.checklist_updater.update_phase_completion("phase5", status="running")
        
        start_time = time.time()
        
        try:
            # 結果ファイルパス取得
            ab_results_path = self.output_dir / "ab_test_hf_benchmark" / "ab_test_results.json"
            hf_results_path = self.output_dir / "ab_test_hf_benchmark" / "hf_benchmark_results.json"
            
            if not ab_results_path.exists():
                raise FileNotFoundError(f"AB test results not found: {ab_results_path}")
            
            # 可視化スクリプト実行
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluation" / "visualization" / "visualize_ab_hf_benchmark.py"),
                "--ab-results", str(ab_results_path),
                "--output-dir", str(self.output_dir / "visualizations")
            ]
            
            if hf_results_path.exists():
                cmd.extend(["--hf-results", str(hf_results_path)])
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            
            if result.returncode != 0:
                logger.warning(f"Visualization failed with return code {result.returncode}, continuing...")
            else:
                logger.info("[OK] Visualization completed")
            
            # 結果を取得
            metrics = {
                'plots_generated': len(list((self.output_dir / "visualizations").glob("*.png"))) if (self.output_dir / "visualizations").exists() else 0,
                'status': 'completed'
            }
            
            duration = time.time() - start_time
            
            self.progress_manager.update_phase_status("phase5", "completed", progress=1.0, metrics=metrics)
            self.checklist_updater.update_phase_completion("phase5", status="completed", metrics=metrics)
            
            logger.info(f"[OK] Phase 5 completed in {duration:.2f} seconds")
            
            return {
                'status': 'completed',
                'duration': duration,
                'metrics': metrics,
                'visualization_dir': self.output_dir / "visualizations"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            self.progress_manager.update_phase_status("phase5", "failed", progress=0.0, error_message=error_msg)
            self.checklist_updater.update_phase_completion("phase5", status="failed", error_message=error_msg)
            
            logger.error(f"[FAILED] Phase 5 failed: {e}")
            raise
    
    def run_complete_pipeline(self, resume: bool = True, skip_phases: Optional[List[str]] = None):
        """
        全フェーズ統合実行
        
        Args:
            resume: チェックポイントから再開するか
            skip_phases: スキップするフェーズのリスト
        """
        logger.info("="*80)
        logger.info("Starting Complete SO8T A/B Pipeline")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Resume: {resume}")
        logger.info(f"Skip phases: {skip_phases or []}")
        
        # 進捗管理開始
        self.progress_manager.start_logging()
        
        # チェックポイント読み込み
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                logger.info(f"Resuming from checkpoint: {checkpoint.get('current_phase')}")
                self.current_phase = checkpoint.get('current_phase')
                self.phase_results = checkpoint.get('phase_results', {})
        
        pipeline_start_time = time.time()
        
        try:
            # Phase 1: データ収集・前処理
            if skip_phases is None or "phase1" not in skip_phases:
                self.phase_results['phase1'] = self.run_phase1_data_pipeline()
                self._save_checkpoint()
            else:
                logger.info("[SKIP] Phase 1")
            
            # Phase 2: SO(8) Transformer再学習
            if skip_phases is None or "phase2" not in skip_phases:
                self.phase_results['phase2'] = self.run_phase2_training()
                self._save_checkpoint()
            else:
                logger.info("[SKIP] Phase 2")
            
            # Phase 3: GGUF変換
            if skip_phases is None or "phase3" not in skip_phases:
                self.phase_results['phase3'] = self.run_phase3_gguf_conversion()
                self._save_checkpoint()
            else:
                logger.info("[SKIP] Phase 3")
            
            # Phase 4: A/Bテスト評価
            if skip_phases is None or "phase4" not in skip_phases:
                self.phase_results['phase4'] = self.run_phase4_ab_test()
                self._save_checkpoint()
            else:
                logger.info("[SKIP] Phase 4")
            
            # Phase 5: 可視化
            if skip_phases is None or "phase5" not in skip_phases:
                self.phase_results['phase5'] = self.run_phase5_visualization()
                self._save_checkpoint()
            else:
                logger.info("[SKIP] Phase 5")
            
            # 最終ログ生成
            self.progress_manager.log_progress()
            self.progress_manager.stop_logging()
            
            total_duration = time.time() - pipeline_start_time
            
            logger.info("="*80)
            logger.info("[SUCCESS] Complete SO8T A/B Pipeline Finished!")
            logger.info("="*80)
            logger.info(f"Total duration: {total_duration/3600:.2f} hours ({total_duration:.2f} seconds)")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("="*80)
            
            # 音声通知
            self._play_audio_notification()
            
            return self.phase_results
            
        except Exception as e:
            self.progress_manager.stop_logging()
            logger.error("="*80)
            logger.error(f"[ERROR] Pipeline failed: {e}")
            logger.error("="*80)
            logger.exception(e)
            raise
    
    def _play_audio_notification(self):
        """音声通知を再生"""
        audio_file = PROJECT_ROOT / ".cursor" / "marisa_owattaze.wav"
        if audio_file.exists():
            try:
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


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Complete SO8T A/B Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ab_test_so8t_complete.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from checkpoint"
    )
    parser.add_argument(
        "--skip-phases",
        type=str,
        nargs='+',
        help="Phases to skip (e.g., --skip-phases phase1 phase2)"
    )
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        logger.info("Using default configuration...")
        config = {
            'output_dir': 'eval_results/complete_so8t_ab',
            'checkpoint_dir': 'checkpoints/complete_pipeline',
            'progress': {'log_interval': 1800}
        }
    else:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # パイプライン初期化
    pipeline = CompleteSO8TABPipeline(config)
    
    # パイプライン実行
    try:
        results = pipeline.run_complete_pipeline(
            resume=not args.no_resume,
            skip_phases=args.skip_phases
        )
        
        logger.info("Pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("[WARNING] Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"[FAILED] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())




















