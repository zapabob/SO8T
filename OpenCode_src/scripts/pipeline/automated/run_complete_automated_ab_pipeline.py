#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6: 完全自動化統合パイプライン

Phase 1-4を統合実行し、チェックポイント管理・電源断リカバリー・リソースバランス監視を行います。

Usage:
    python scripts/pipelines/run_complete_automated_ab_pipeline.py --config configs/complete_automated_ab_pipeline.yaml
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
import signal
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/run_complete_automated_ab_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 進捗管理システムのインポート
try:
    from scripts.utils.progress_manager import ProgressManager
    from scripts.utils.checklist_updater import ChecklistUpdater
    PROGRESS_MANAGER_AVAILABLE = True
except ImportError:
    PROGRESS_MANAGER_AVAILABLE = False
    logger.warning("Progress manager not available")

# リソースバランサーのインポート
try:
    from scripts.utils.resource_balancer import ResourceBalancer
    RESOURCE_BALANCER_AVAILABLE = True
except ImportError:
    RESOURCE_BALANCER_AVAILABLE = False
    logger.warning("Resource balancer not available")

# 自動再開マネージャーのインポート
try:
    from scripts.utils.auto_resume import AutoResumeManager
    AUTO_RESUME_AVAILABLE = True
except ImportError:
    AUTO_RESUME_AVAILABLE = False
    logger.warning("Auto resume manager not available")


class CompleteAutomatedABPipeline:
    """完全自動化A/Bテストパイプラインクラス"""
    
    def __init__(self, config: Dict[str, Any], resume_from_checkpoint: Optional[str] = None):
        """
        Args:
            config: 設定辞書
            resume_from_checkpoint: チェックポイントパス（復旧時）
        """
        self.config = config
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # チェックポイント設定
        checkpoint_config = config.get('checkpoint', {})
        self.checkpoint_dir = Path(checkpoint_config.get('save_dir', 'D:/webdataset/checkpoints/complete_ab_pipeline'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_config.get('interval_seconds', 180)  # デフォルト3分
        self.max_checkpoints = checkpoint_config.get('max_checkpoints', 5)  # デフォルト5回分
        self.last_checkpoint_time = time.time()
        
        # 進捗管理システム
        if PROGRESS_MANAGER_AVAILABLE:
            log_interval = config.get('progress', {}).get('log_interval', 1800)
            self.progress_manager = ProgressManager(session_id=self.session_id, log_interval=log_interval)
            self.checklist_updater = ChecklistUpdater()
        else:
            self.progress_manager = None
            self.checklist_updater = None
        
        # リソースバランサー
        if RESOURCE_BALANCER_AVAILABLE:
            self.resource_balancer = ResourceBalancer(config)
        else:
            self.resource_balancer = None
        
        # 自動再開マネージャー
        if AUTO_RESUME_AVAILABLE:
            self.auto_resume_manager = AutoResumeManager()
        else:
            self.auto_resume_manager = None
        
        # フェーズ状態
        self.current_phase = None
        self.phase_results = {}
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        logger.info("="*80)
        logger.info("Complete Automated A/B Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Resume from checkpoint: {resume_from_checkpoint}")
    
    def _setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, saving checkpoint...")
            self._save_checkpoint()
            if self.resource_balancer:
                self.resource_balancer.stop_monitoring()
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
        if self.resume_from_checkpoint:
            checkpoint_path = Path(self.resume_from_checkpoint)
        else:
            # 最新のチェックポイントを検索
            checkpoint_files = list(self.checkpoint_dir.glob("*_checkpoint.json"))
            if not checkpoint_files:
                return None
            checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def run_phase1(self) -> Dict[str, Any]:
        """Phase 1: モデルA準備"""
        logger.info("="*80)
        logger.info("PHASE 1: Model A Preparation")
        logger.info("="*80)
        
        self.current_phase = "phase1"
        if self.progress_manager:
            self.progress_manager.update_phase_status("phase1", "running", progress=0.0)
        if self.checklist_updater:
            self.checklist_updater.update_phase_completion("phase1", status="running")
        
        start_time = time.time()
        
        try:
            # Phase 1スクリプト実行
            phase1_script = PROJECT_ROOT / "scripts" / "pipelines" / "phase1_prepare_model_a.py"
            cmd = [
                sys.executable,
                str(phase1_script),
                "--config", str(PROJECT_ROOT / "configs" / "complete_automated_ab_pipeline.yaml")
            ]
            
            logger.info(f"[PHASE 1] Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=True
            )
            
            # 結果をパース（簡易版）
            phase1_result = {
                'status': 'completed',
                'duration': time.time() - start_time
            }
            
            self.phase_results['phase1'] = phase1_result
            
            if self.progress_manager:
                self.progress_manager.update_phase_status("phase1", "completed", progress=1.0)
            if self.checklist_updater:
                self.checklist_updater.update_phase_completion("phase1", status="completed")
            
            self._save_checkpoint()
            
            logger.info("[OK] Phase 1 completed")
            return phase1_result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Phase 1 failed: {e}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            if self.progress_manager:
                self.progress_manager.update_phase_status("phase1", "error", progress=0.0)
            if self.checklist_updater:
                self.checklist_updater.update_phase_completion("phase1", status="error")
            raise
    
    def run_phase2(self) -> Dict[str, Any]:
        """Phase 2: モデルB準備"""
        logger.info("="*80)
        logger.info("PHASE 2: Model B Preparation")
        logger.info("="*80)
        
        self.current_phase = "phase2"
        if self.progress_manager:
            self.progress_manager.update_phase_status("phase2", "running", progress=0.0)
        if self.checklist_updater:
            self.checklist_updater.update_phase_completion("phase2", status="running")
        
        start_time = time.time()
        
        try:
            # Phase 2スクリプト実行
            phase2_script = PROJECT_ROOT / "scripts" / "pipelines" / "phase2_prepare_model_b.py"
            cmd = [
                sys.executable,
                str(phase2_script),
                "--config", str(PROJECT_ROOT / "configs" / "complete_automated_ab_pipeline.yaml")
            ]
            
            logger.info(f"[PHASE 2] Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=True
            )
            
            phase2_result = {
                'status': 'completed',
                'duration': time.time() - start_time
            }
            
            self.phase_results['phase2'] = phase2_result
            
            if self.progress_manager:
                self.progress_manager.update_phase_status("phase2", "completed", progress=1.0)
            if self.checklist_updater:
                self.checklist_updater.update_phase_completion("phase2", status="completed")
            
            self._save_checkpoint()
            
            logger.info("[OK] Phase 2 completed")
            return phase2_result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Phase 2 failed: {e}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            if self.progress_manager:
                self.progress_manager.update_phase_status("phase2", "error", progress=0.0)
            if self.checklist_updater:
                self.checklist_updater.update_phase_completion("phase2", status="error")
            raise
    
    def run_phase3(self) -> Dict[str, Any]:
        """Phase 3: Ollamaモデル登録"""
        logger.info("="*80)
        logger.info("PHASE 3: Ollama Model Registration")
        logger.info("="*80)
        
        self.current_phase = "phase3"
        if self.progress_manager:
            self.progress_manager.update_phase_status("phase3", "running", progress=0.0)
        if self.checklist_updater:
            self.checklist_updater.update_phase_completion("phase3", status="running")
        
        start_time = time.time()
        
        try:
            # Phase 3スクリプト実行
            phase3_script = PROJECT_ROOT / "scripts" / "pipelines" / "phase3_register_ollama_models.py"
            cmd = [
                sys.executable,
                str(phase3_script),
                "--config", str(PROJECT_ROOT / "configs" / "complete_automated_ab_pipeline.yaml")
            ]
            
            # Phase 1/2の結果を渡す
            if 'phase1' in self.phase_results:
                phase1_result_path = self.checkpoint_dir / f"{self.session_id}_phase1_result.json"
                with open(phase1_result_path, 'w', encoding='utf-8') as f:
                    json.dump(self.phase_results['phase1'], f, indent=2, ensure_ascii=False)
                cmd.extend(["--phase1-result", str(phase1_result_path)])
            
            if 'phase2' in self.phase_results:
                phase2_result_path = self.checkpoint_dir / f"{self.session_id}_phase2_result.json"
                with open(phase2_result_path, 'w', encoding='utf-8') as f:
                    json.dump(self.phase_results['phase2'], f, indent=2, ensure_ascii=False)
                cmd.extend(["--phase2-result", str(phase2_result_path)])
            
            logger.info(f"[PHASE 3] Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=True
            )
            
            phase3_result = {
                'status': 'completed',
                'duration': time.time() - start_time
            }
            
            self.phase_results['phase3'] = phase3_result
            
            if self.progress_manager:
                self.progress_manager.update_phase_status("phase3", "completed", progress=1.0)
            if self.checklist_updater:
                self.checklist_updater.update_phase_completion("phase3", status="completed")
            
            self._save_checkpoint()
            
            logger.info("[OK] Phase 3 completed")
            return phase3_result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Phase 3 failed: {e}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            if self.progress_manager:
                self.progress_manager.update_phase_status("phase3", "error", progress=0.0)
            if self.checklist_updater:
                self.checklist_updater.update_phase_completion("phase3", status="error")
            raise
    
    def run_phase4(self) -> Dict[str, Any]:
        """Phase 4: ベンチマークテスト実行"""
        logger.info("="*80)
        logger.info("PHASE 4: Benchmark Tests")
        logger.info("="*80)
        
        self.current_phase = "phase4"
        if self.progress_manager:
            self.progress_manager.update_phase_status("phase4", "running", progress=0.0)
        if self.checklist_updater:
            self.checklist_updater.update_phase_completion("phase4", status="running")
        
        start_time = time.time()
        
        try:
            # Phase 4スクリプト実行
            phase4_script = PROJECT_ROOT / "scripts" / "pipelines" / "phase4_run_benchmarks.py"
            cmd = [
                sys.executable,
                str(phase4_script),
                "--config", str(PROJECT_ROOT / "configs" / "complete_automated_ab_pipeline.yaml")
            ]
            
            # Phase 3の結果を渡す
            if 'phase3' in self.phase_results:
                phase3_result_path = self.checkpoint_dir / f"{self.session_id}_phase3_result.json"
                with open(phase3_result_path, 'w', encoding='utf-8') as f:
                    json.dump(self.phase_results['phase3'], f, indent=2, ensure_ascii=False)
                cmd.extend(["--phase3-result", str(phase3_result_path)])
            
            logger.info(f"[PHASE 4] Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=True
            )
            
            phase4_result = {
                'status': 'completed',
                'duration': time.time() - start_time
            }
            
            self.phase_results['phase4'] = phase4_result
            
            if self.progress_manager:
                self.progress_manager.update_phase_status("phase4", "completed", progress=1.0)
            if self.checklist_updater:
                self.checklist_updater.update_phase_completion("phase4", status="completed")
            
            self._save_checkpoint()
            
            logger.info("[OK] Phase 4 completed")
            return phase4_result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Phase 4 failed: {e}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            if self.progress_manager:
                self.progress_manager.update_phase_status("phase4", "error", progress=0.0)
            if self.checklist_updater:
                self.checklist_updater.update_phase_completion("phase4", status="error")
            raise
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """完全パイプライン実行"""
        logger.info("="*80)
        logger.info("Starting Complete Automated A/B Pipeline")
        logger.info("="*80)
        
        start_time = time.time()
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint.get('current_phase')}")
            self.session_id = checkpoint.get('session_id', self.session_id)
            self.phase_results = checkpoint.get('phase_results', {})
        
        # リソースバランス監視開始
        if self.resource_balancer:
            self.resource_balancer.start_monitoring()
        
        try:
            # Phase 1実行（未完了の場合）
            if 'phase1' not in self.phase_results or self.phase_results['phase1'].get('status') != 'completed':
                self.run_phase1()
            
            # Phase 2実行（未完了の場合）
            if 'phase2' not in self.phase_results or self.phase_results['phase2'].get('status') != 'completed':
                self.run_phase2()
            
            # Phase 3実行（未完了の場合）
            if 'phase3' not in self.phase_results or self.phase_results['phase3'].get('status') != 'completed':
                self.run_phase3()
            
            # Phase 4実行（未完了の場合）
            if 'phase4' not in self.phase_results or self.phase_results['phase4'].get('status') != 'completed':
                self.run_phase4()
            
            # リソースバランス監視停止
            if self.resource_balancer:
                self.resource_balancer.stop_monitoring()
                self.resource_balancer.save_metrics_history()
            
            duration = time.time() - start_time
            
            result = {
                'status': 'completed',
                'duration': duration,
                'session_id': self.session_id,
                'phase_results': self.phase_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # 最終チェックポイント保存
            self._save_checkpoint()
            
            logger.info("="*80)
            logger.info("[SUCCESS] Complete Automated A/B Pipeline Completed!")
            logger.info("="*80)
            logger.info(f"Total duration: {duration/60:.2f} minutes")
            logger.info(f"Session ID: {self.session_id}")
            
            # 音声通知
            self._play_audio_notification()
            
            return result
            
        except Exception as e:
            logger.error("="*80)
            logger.error(f"[ERROR] Complete Automated A/B Pipeline Failed: {e}")
            logger.error("="*80)
            logger.exception(e)
            
            if self.resource_balancer:
                self.resource_balancer.stop_monitoring()
            
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
        description="Complete Automated A/B Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/complete_automated_ab_pipeline.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint file path"
    )
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # パイプライン実行
    pipeline = CompleteAutomatedABPipeline(config, resume_from_checkpoint=args.resume)
    
    try:
        result = pipeline.run_complete_pipeline()
        logger.info("Complete Automated A/B Pipeline completed successfully!")
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

