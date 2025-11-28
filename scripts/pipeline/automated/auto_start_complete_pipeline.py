#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T完全パイプライン全自動化スクリプト

Windows起動時自動実行、前回セッションからの自動復旧機能

Usage:
    python scripts/pipelines/auto_start_complete_pipeline.py --setup
    python scripts/pipelines/auto_start_complete_pipeline.py --run
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_start_complete_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutoStartCompletePipeline:
    """全自動化スクリプトクラス"""
    
    def __init__(self):
        """初期化"""
        self.project_root = PROJECT_ROOT
        self.config_path = self.project_root / "configs" / "ab_test_so8t_complete.yaml"
        self.checkpoint_dir = self.project_root / "checkpoints" / "complete_pipeline"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("Auto Start Complete Pipeline Initialized")
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
        task_name = "SO8T-CompletePipeline-AutoStart"
        
        # スクリプトパス
        script_path = self.project_root / "scripts" / "pipelines" / "auto_start_complete_pipeline.py"
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
        
        # タスクスケジューラコマンド
        create_cmd = [
            "schtasks", "/create",
            "/tn", task_name,
            "/tr", task_command,
            "/sc", "onlogon",  # ログオン時
            "/rl", "highest",  # 最高権限
            "/f"  # 強制作成
        ]
        
        try:
            result = subprocess.run(create_cmd, capture_output=True, text=True, check=True)
            logger.info("[OK] Task created successfully")
            logger.info(f"Task name: {task_name}")
            logger.info(f"Trigger: On user logon")
            logger.info(f"Command: {task_command}")
            
            # タスクの詳細を表示
            subprocess.run(["schtasks", "/query", "/tn", task_name, "/fo", "list", "/v"], check=False)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Failed to create task: {e}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
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
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def run_pipeline_with_progress(self, resume: bool = True):
        """進捗管理付きパイプライン実行"""
        logger.info("="*80)
        logger.info("Running pipeline with progress tracking")
        logger.info("="*80)
        
        # チェックポイント確認
        checkpoint_data = None
        if resume:
            checkpoint_data = self.check_and_resume()
        
        # パイプライン実行
        cmd = [
            sys.executable,
            str(self.project_root / "scripts" / "pipelines" / "run_complete_so8t_ab_pipeline.py"),
            "--config", str(self.config_path)
        ]
        
        if checkpoint_data:
            logger.info("Resuming from checkpoint...")
            # 完了済みフェーズをスキップ
            completed_phases = list(checkpoint_data.get('phase_results', {}).keys())
            if completed_phases:
                skip_phases = completed_phases
                cmd.extend(["--skip-phases"] + skip_phases)
                logger.info(f"Skipping completed phases: {skip_phases}")
        else:
            logger.info("Starting new pipeline session...")
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, cwd=str(self.project_root))
            
            if result.returncode == 0:
                logger.info("[SUCCESS] Pipeline completed successfully!")
                self._play_audio_notification()
                return True
            else:
                logger.error(f"[FAILED] Pipeline failed with return code {result.returncode}")
                return False
                
        except KeyboardInterrupt:
            logger.warning("[WARNING] Pipeline interrupted by user")
            return False
        except Exception as e:
            logger.error(f"[FAILED] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _play_audio_notification(self):
        """音声通知を再生"""
        audio_file = self.project_root / ".cursor" / "marisa_owattaze.wav"
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
                    cwd=str(self.project_root),
                    check=False
                )
            except Exception as e:
                logger.warning(f"Failed to play audio: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Auto Start Complete Pipeline"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Setup auto-start task in Windows Task Scheduler"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run pipeline with auto-resume"
    )
    
    args = parser.parse_args()
    
    auto_start = AutoStartCompletePipeline()
    
    if args.setup:
        success = auto_start.setup_auto_start()
        return 0 if success else 1
    elif args.run:
        success = auto_start.run_pipeline_with_progress(resume=True)
        return 0 if success else 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())




















