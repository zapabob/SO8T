#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
電源断リカバリー自動再開スクリプト

電源オン時に学習とクロールを自動的に再開する機能
- セッションファイルの自動検出
- 学習とクロールの状態確認
- 未完了セッションの自動再開
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import argparse

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_resume.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# プロジェクトルートパス
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# セッションファイルパス
TRAINING_SESSION_PATHS = [
    PROJECT_ROOT / "checkpoints" / "training" / "training_session.json",
    PROJECT_ROOT / "so8t-mmllm" / "scripts" / "training" / "checkpoints" / "training_session.json",
]

CRAWL_SESSION_PATHS = [
    PROJECT_ROOT / "data" / "checkpoints" / "session.json",
    PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "data" / "checkpoints" / "session.json",
]

# セッション有効期限（7日間）
SESSION_EXPIRY_DAYS = 7
SESSION_EXPIRY_SECONDS = SESSION_EXPIRY_DAYS * 24 * 60 * 60


class AutoResumeManager:
    """自動再開マネージャー"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or PROJECT_ROOT
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Auto Resume Manager initialized")
        logger.info(f"Project root: {self.project_root}")
    
    def find_training_session(self) -> Optional[Dict[str, Any]]:
        """学習セッションファイルを検索"""
        for session_path in TRAINING_SESSION_PATHS:
            # プロジェクトルートからの相対パスを解決
            full_path = self.project_root / session_path if not session_path.is_absolute() else session_path
            
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    # セッション有効期限チェック
                    if self._is_session_expired(session_data):
                        logger.warning(f"Training session expired: {full_path}")
                        continue
                    
                    # 完了フラグチェック
                    if session_data.get('completed', False):
                        logger.info(f"Training session already completed: {full_path}")
                        continue
                    
                    logger.info(f"Found training session: {full_path}")
                    return {
                        'path': str(full_path),
                        'data': session_data,
                        'type': 'training'
                    }
                except Exception as e:
                    logger.error(f"Failed to load training session {full_path}: {e}")
                    continue
        
        return None
    
    def find_crawl_session(self) -> Optional[Dict[str, Any]]:
        """クロールセッションファイルを検索"""
        for session_path in CRAWL_SESSION_PATHS:
            # プロジェクトルートからの相対パスを解決
            full_path = self.project_root / session_path if not session_path.is_absolute() else session_path
            
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    # セッション有効期限チェック
                    if self._is_session_expired(session_data):
                        logger.warning(f"Crawl session expired: {full_path}")
                        continue
                    
                    # 完了フラグチェック
                    if session_data.get('completed', False):
                        logger.info(f"Crawl session already completed: {full_path}")
                        continue
                    
                    # 進捗チェック（目標達成済みか）
                    samples_collected = session_data.get('samples_collected', 0)
                    target_samples = session_data.get('target_samples', 0)
                    if target_samples > 0 and samples_collected >= target_samples:
                        logger.info(f"Crawl session target reached: {samples_collected}/{target_samples}")
                        continue
                    
                    logger.info(f"Found crawl session: {full_path}")
                    return {
                        'path': str(full_path),
                        'data': session_data,
                        'type': 'crawl'
                    }
                except Exception as e:
                    logger.error(f"Failed to load crawl session {full_path}: {e}")
                    continue
        
        return None
    
    def _is_session_expired(self, session_data: Dict[str, Any]) -> bool:
        """セッションが有効期限切れかチェック"""
        start_time = session_data.get('start_time', 0)
        if start_time == 0:
            return False
        
        elapsed = time.time() - start_time
        return elapsed > SESSION_EXPIRY_SECONDS
    
    def resume_training(self, session_info: Dict[str, Any]) -> bool:
        """学習を再開"""
        try:
            session_data = session_info['data']
            session_id = session_data.get('session_id', 'unknown')
            current_step = session_data.get('current_step', 0)
            total_steps = session_data.get('total_steps', 0)
            
            logger.info(f"Resuming training session: {session_id}")
            logger.info(f"Progress: {current_step}/{total_steps}")
            
            # チェックポイントパスを取得
            checkpoints = session_data.get('checkpoints', [])
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                checkpoint_path = Path(latest_checkpoint)
                if checkpoint_path.exists():
                    resume_path = str(checkpoint_path.parent)
                else:
                    # チェックポイントディレクトリを検索
                    checkpoint_dir = self._find_checkpoint_dir(session_id, 'training')
                    resume_path = str(checkpoint_dir) if checkpoint_dir else None
            else:
                checkpoint_dir = self._find_checkpoint_dir(session_id, 'training')
                resume_path = str(checkpoint_dir) if checkpoint_dir else None
            
            # 学習スクリプトを実行
            training_script = self.project_root / "scripts" / "train_so8t_recovery.py"
            if not training_script.exists():
                logger.error(f"Training script not found: {training_script}")
                return False
            
            cmd = [sys.executable, str(training_script)]
            if resume_path:
                cmd.extend(["--resume", resume_path])
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # バックグラウンドで実行
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            logger.info(f"Training resumed with PID: {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume training: {e}")
            return False
    
    def resume_crawl(self, session_info: Dict[str, Any]) -> bool:
        """クロールを再開"""
        try:
            session_data = session_info['data']
            session_id = session_data.get('session_id', 'unknown')
            samples_collected = session_data.get('samples_collected', 0)
            target_samples = session_data.get('target_samples', 0)
            
            logger.info(f"Resuming crawl session: {session_id}")
            logger.info(f"Progress: {samples_collected}/{target_samples}")
            
            # クロールスクリプトを実行
            crawl_script = self.project_root / "so8t-mmllm" / "scripts" / "data" / "collect_japanese_data.py"
            if not crawl_script.exists():
                logger.error(f"Crawl script not found: {crawl_script}")
                return False
            
            # セッションファイルが存在する場合、自動的に復旧される
            cmd = [
                sys.executable,
                str(crawl_script),
                "--target", str(target_samples),
                "--auto-resume"
            ]
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # バックグラウンドで実行
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            logger.info(f"Crawl resumed with PID: {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume crawl: {e}")
            return False
    
    def _find_checkpoint_dir(self, session_id: str, session_type: str) -> Optional[Path]:
        """チェックポイントディレクトリを検索"""
        if session_type == 'training':
            # 学習チェックポイントディレクトリを検索
            checkpoint_base = self.project_root / "checkpoints"
            if checkpoint_base.exists():
                # セッションIDを含むディレクトリを検索
                for checkpoint_dir in checkpoint_base.iterdir():
                    if checkpoint_dir.is_dir() and session_id in checkpoint_dir.name:
                        return checkpoint_dir
        elif session_type == 'crawl':
            # クロールチェックポイントディレクトリを検索
            checkpoint_base = self.project_root / "data" / "checkpoints"
            if checkpoint_base.exists():
                return checkpoint_base
        
        return None
    
    def run(self, resume_training: bool = True, resume_crawl: bool = True) -> Tuple[bool, bool]:
        """自動再開を実行"""
        logger.info("="*60)
        logger.info("Auto Resume Manager - Starting")
        logger.info("="*60)
        
        training_resumed = False
        crawl_resumed = False
        
        # 学習セッションを検索して再開
        if resume_training:
            training_session = self.find_training_session()
            if training_session:
                logger.info("Found incomplete training session, resuming...")
                training_resumed = self.resume_training(training_session)
                if training_resumed:
                    logger.info("[OK] Training resumed successfully")
                else:
                    logger.error("[NG] Failed to resume training")
            else:
                logger.info("No incomplete training session found")
        
        # クロールセッションを検索して再開
        if resume_crawl:
            crawl_session = self.find_crawl_session()
            if crawl_session:
                logger.info("Found incomplete crawl session, resuming...")
                crawl_resumed = self.resume_crawl(crawl_session)
                if crawl_resumed:
                    logger.info("[OK] Crawl resumed successfully")
                else:
                    logger.error("[NG] Failed to resume crawl")
            else:
                logger.info("No incomplete crawl session found")
        
        logger.info("="*60)
        logger.info("Auto Resume Manager - Completed")
        logger.info(f"Training: {'Resumed' if training_resumed else 'No action'}")
        logger.info(f"Crawl: {'Resumed' if crawl_resumed else 'No action'}")
        logger.info("="*60)
        
        return training_resumed, crawl_resumed


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Auto Resume Manager for SO8T")
    parser.add_argument("--no-training", action="store_true", help="Skip training resume")
    parser.add_argument("--no-crawl", action="store_true", help="Skip crawl resume")
    parser.add_argument("--project-root", type=str, default=None, help="Project root directory")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else None
    
    manager = AutoResumeManager(project_root)
    
    resume_training = not args.no_training
    resume_crawl = not args.no_crawl
    
    training_resumed, crawl_resumed = manager.run(
        resume_training=resume_training,
        resume_crawl=resume_crawl
    )
    
    # 終了コード
    if training_resumed or crawl_resumed:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

