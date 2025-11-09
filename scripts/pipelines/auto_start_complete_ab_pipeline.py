#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 7: 自動起動設定

Windowsタスクスケジューラ登録（電源投入時自動実行）、前回セッションからの自動復旧、
進捗管理システム統合を行います。

Usage:
    python scripts/pipelines/auto_start_complete_ab_pipeline.py --setup
    python scripts/pipelines/auto_start_complete_ab_pipeline.py --run
"""

import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_start_complete_ab_pipeline.log', encoding='utf-8'),
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

# 自動再開マネージャーのインポート
try:
    from scripts.utils.auto_resume import AutoResumeManager
    AUTO_RESUME_AVAILABLE = True
except ImportError:
    AUTO_RESUME_AVAILABLE = False
    logger.warning("Auto resume manager not available")


class AutoStartCompleteABPipeline:
    """自動起動完全A/Bテストパイプラインクラス"""
    
    def __init__(self):
        """初期化"""
        self.project_root = PROJECT_ROOT
        self.config_path = self.project_root / "configs" / "complete_automated_ab_pipeline.yaml"
        checkpoint_base = self.config_path.parent.parent
        self.checkpoint_dir = checkpoint_base / "checkpoints" / "complete_ab_pipeline"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 進捗管理システム
        if PROGRESS_MANAGER_AVAILABLE:
            self.progress_manager = None  # 実行時に初期化
            self.checklist_updater = ChecklistUpdater()
        else:
            self.progress_manager = None
            self.checklist_updater = None
        
        # 自動再開マネージャー
        if AUTO_RESUME_AVAILABLE:
            self.auto_resume_manager = AutoResumeManager()
        else:
            self.auto_resume_manager = None
        
        logger.info("="*80)
        logger.info("Auto Start Complete A/B Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Config path: {self.config_path}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def setup_auto_start(self):
        """Windowsタスクスケジューラに自動実行タスクを登録"""
        logger.info("="*80)
        logger.info("Setting up auto-start task")
        logger.info("="*80)
        
        # タスク名
        task_name = "SO8T-CompleteABPipeline-AutoStart"
        
        # スクリプトパス
        script_path = self.project_root / "scripts" / "pipelines" / "auto_start_complete_ab_pipeline.py"
        python_exe = sys.executable
        
        # タスクコマンド
        task_command = f'"{python_exe}" "{script_path}" --run'
        
        # 既存のタスクを削除（存在する場合）
        try:
            result = subprocess.run(
                ["schtasks", "/query", "/tn", task_name],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"Removing existing task: {task_name}")
                subprocess.run(
                    ["schtasks", "/delete", "/tn", task_name, "/f"],
                    check=False
                )
        except Exception as e:
            logger.warning(f"Failed to check/remove existing task: {e}")
        
        # 新しいタスクを作成
        logger.info(f"Creating new task: {task_name}")
        
        # タスクスケジューラコマンド（電源投入時）
        create_cmd = [
            "schtasks", "/create",
            "/tn", task_name,
            "/tr", task_command,
            "/sc", "onstart",  # システム起動時
            "/ru", "SYSTEM",  # SYSTEMアカウントで実行
            "/rl", "highest",  # 最高権限
            "/f"  # 強制作成
        ]
        
        try:
            result = subprocess.run(create_cmd, capture_output=True, text=True, check=True)
            logger.info("[OK] Task created successfully")
            logger.info(f"Task name: {task_name}")
            logger.info(f"Trigger: On system start")
            logger.info(f"Command: {task_command}")
            
            # タスクの詳細を表示
            subprocess.run(["schtasks", "/query", "/tn", task_name, "/fo", "list", "/v"], check=False)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Failed to create task: {e}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            logger.info("Note: Task creation may require administrator privileges")
            return False
    
    def check_and_resume(self) -> Optional[Dict]:
        """前回セッションからの自動復旧"""
        logger.info("="*80)
        logger.info("Checking for previous session...")
        logger.info("="*80)
        
        # チェックポイントファイルを検索
        checkpoint_files = list(self.checkpoint_dir.glob("*_checkpoint.json"))
        
        if not checkpoint_files:
            logger.info("No checkpoint found. Starting new session.")
            return None
        
        # 最新のチェックポイントを取得
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        
        logger.info(f"Found checkpoint: {latest_checkpoint}")
        
        try:
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            session_id = checkpoint_data.get('session_id')
            current_phase = checkpoint_data.get('current_phase')
            phase_results = checkpoint_data.get('phase_results', {})
            timestamp = checkpoint_data.get('timestamp')
            
            logger.info(f"Session ID: {session_id}")
            logger.info(f"Current phase: {current_phase}")
            logger.info(f"Timestamp: {timestamp}")
            logger.info(f"Completed phases: {list(phase_results.keys())}")
            
            return {
                'checkpoint_path': str(latest_checkpoint),
                'session_id': session_id,
                'current_phase': current_phase,
                'phase_results': phase_results
            }
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def run_pipeline_with_progress(self, resume_checkpoint: Optional[str] = None):
        """進捗管理付きパイプライン実行"""
        logger.info("="*80)
        logger.info("Running Complete A/B Pipeline with Progress Management")
        logger.info("="*80)
        
        # 設定ファイル読み込み
        if not self.config_path.exists():
            logger.error(f"Config file not found: {self.config_path}")
            return False
        
        import yaml
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 進捗管理システム初期化
        if PROGRESS_MANAGER_AVAILABLE:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_interval = config.get('progress', {}).get('log_interval', 1800)
            self.progress_manager = ProgressManager(session_id=session_id, log_interval=log_interval)
        
        # 統合パイプライン実行
        from scripts.pipelines.run_complete_automated_ab_pipeline import CompleteAutomatedABPipeline
        
        pipeline = CompleteAutomatedABPipeline(config, resume_from_checkpoint=resume_checkpoint)
        
        try:
            result = pipeline.run_complete_pipeline()
            
            logger.info("="*80)
            logger.info("[SUCCESS] Complete A/B Pipeline Completed!")
            logger.info("="*80)
            logger.info(f"Session ID: {result.get('session_id')}")
            logger.info(f"Duration: {result.get('duration', 0)/60:.2f} minutes")
            
            # 音声通知
            self._play_audio_notification()
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Pipeline execution failed: {e}")
            logger.exception(e)
            return False
    
    def run(self):
        """自動実行メイン処理"""
        logger.info("="*80)
        logger.info("Auto Start Complete A/B Pipeline - Run Mode")
        logger.info("="*80)
        
        # 前回セッションからの復旧チェック
        checkpoint_info = self.check_and_resume()
        
        resume_checkpoint = None
        if checkpoint_info:
            logger.info("Resuming from previous session...")
            resume_checkpoint = checkpoint_info['checkpoint_path']
        else:
            logger.info("Starting new session...")
        
        # パイプライン実行
        success = self.run_pipeline_with_progress(resume_checkpoint=resume_checkpoint)
        
        if success:
            logger.info("[OK] Auto start pipeline completed successfully")
            return 0
        else:
            logger.error("[ERROR] Auto start pipeline failed")
            return 1
    
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
        description="Auto Start Complete A/B Pipeline"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Setup Windows Task Scheduler auto-start task"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the pipeline (called by task scheduler)"
    )
    
    args = parser.parse_args()
    
    auto_start = AutoStartCompleteABPipeline()
    
    if args.setup:
        # タスクスケジューラ登録
        success = auto_start.setup_auto_start()
        if success:
            logger.info("[OK] Auto-start task setup completed")
            return 0
        else:
            logger.error("[ERROR] Auto-start task setup failed")
            return 1
    
    elif args.run:
        # パイプライン実行
        return auto_start.run()
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

