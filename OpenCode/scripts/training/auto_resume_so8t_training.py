#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T再学習自動復旧スクリプト

Windows起動時にチェックポイントから自動復旧する

機能:
- チェックポイントの自動検出
- 最新セッションの特定
- チェックポイントからの自動復旧
- 新規セッション開始（チェックポイントがない場合）
- 音声通知（PlaySync()を使用して確実に再生）

Usage:
    python scripts/training/auto_resume_so8t_training.py
"""

import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_resume_so8t_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def play_audio_notification():
    """音声通知を再生（PlaySync()を使用して確実に再生）"""
    audio_path = PROJECT_ROOT / ".cursor" / "marisa_owattaze.wav"
    
    if not audio_path.exists():
        logger.warning(f"Audio file not found: {audio_path}")
        # フォールバック: システムビープ
        try:
            import sys
            if sys.platform == 'win32':
                import winsound
                winsound.Beep(1000, 500)
        except Exception:
            pass
        return False
    
    try:
        # PowerShellを使用してPlaySync()で確実に再生
        ps_command = f'''
        $audioPath = "{audio_path}"
        if (Test-Path $audioPath) {{
            try {{
                Add-Type -AssemblyName System.Windows.Forms
                $player = [System.Media.SoundPlayer]::new($audioPath)
                $player.PlaySync()
                Write-Host "[OK] Audio notification played successfully"
            }} catch {{
                Write-Host "[WARNING] Failed to play audio: $($_.Exception.Message)"
                [System.Console]::Beep(1000, 500)
            }}
        }} else {{
            Write-Host "[WARNING] Audio file not found: $audioPath"
            [System.Console]::Beep(800, 1000)
        }}
        '''
        
        result = subprocess.run(
            ['powershell', '-Command', ps_command],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            logger.info("Audio notification played successfully")
            return True
        else:
            logger.warning(f"Failed to play audio: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to play audio notification: {e}")
        return False


def find_latest_session(checkpoint_base: Path) -> Optional[Dict[str, Any]]:
    """最新セッションの特定"""
    if not checkpoint_base.exists():
        logger.info(f"Checkpoint base directory not found: {checkpoint_base}")
        return None
    
    # 全セッションディレクトリを検索
    session_dirs = [d for d in checkpoint_base.iterdir() if d.is_dir()]
    
    if not session_dirs:
        logger.info("No session directories found")
        return None
    
    # セッション情報ファイルがあるディレクトリを検索
    latest_session = None
    latest_time = None
    
    for session_dir in session_dirs:
        session_info_path = session_dir / "session_info.json"
        
        if session_info_path.exists():
            try:
                with open(session_info_path, 'r', encoding='utf-8') as f:
                    session_info = json.load(f)
                
                status = session_info.get('status', 'unknown')
                
                # runningまたはinterruptedのセッションを優先
                if status in ['running', 'interrupted']:
                    last_checkpoint_time = session_info.get('last_checkpoint_time', '')
                    if last_checkpoint_time:
                        try:
                            checkpoint_time = datetime.fromisoformat(last_checkpoint_time)
                            if latest_time is None or checkpoint_time > latest_time:
                                latest_time = checkpoint_time
                                latest_session = {
                                    'session_dir': session_dir,
                                    'session_info': session_info
                                }
                        except Exception as e:
                            logger.warning(f"Failed to parse checkpoint time: {e}")
            except Exception as e:
                logger.warning(f"Failed to load session info from {session_info_path}: {e}")
    
    if latest_session:
        logger.info(f"Found latest session: {latest_session['session_dir'].name}")
        return latest_session
    
    # セッション情報がない場合、最新のディレクトリを使用
    latest_dir = max(session_dirs, key=lambda d: d.stat().st_mtime)
    logger.info(f"Using latest directory (no session info): {latest_dir.name}")
    
    return {
        'session_dir': latest_dir,
        'session_info': None
    }


def find_latest_checkpoint(session_dir: Path) -> Optional[Path]:
    """最新チェックポイントの検索"""
    # ローリングチェックポイントを検索
    rolling_checkpoints = list(session_dir.glob("checkpoint_rolling_*.pt"))
    if rolling_checkpoints:
        latest = max(rolling_checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found latest rolling checkpoint: {latest.name}")
        return latest
    
    # 最終チェックポイントを検索
    final_checkpoint = session_dir / "checkpoint_final.pt"
    if final_checkpoint.exists():
        logger.info(f"Found final checkpoint: {final_checkpoint.name}")
        return final_checkpoint
    
    # 緊急チェックポイントを検索
    emergency_checkpoints = list(session_dir.glob("checkpoint_emergency_*.pt"))
    if emergency_checkpoints:
        latest = max(emergency_checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found emergency checkpoint: {latest.name}")
        return latest
    
    return None


def should_resume(session_info: Optional[Dict[str, Any]]) -> bool:
    """復旧すべきか判定"""
    if session_info is None:
        return True  # セッション情報がない場合も復旧を試みる
    
    status = session_info.get('status', 'unknown')
    
    # runningまたはinterruptedの場合は復旧
    if status in ['running', 'interrupted']:
        return True
    
    # completedの場合は新規セッション開始
    if status == 'completed':
        logger.info("Session already completed. Starting new session.")
        return False
    
    # その他の場合は復旧を試みる
    return True


def main():
    """メイン関数"""
    logger.info("="*80)
    logger.info("SO8T Training Auto Resume")
    logger.info("="*80)
    
    # チェックポイントベースディレクトリ
    checkpoint_base = Path("D:/webdataset/checkpoints/training")
    
    # 最新セッションの検索
    latest_session = find_latest_session(checkpoint_base)
    
    if latest_session is None:
        logger.info("No session found. Starting new training session.")
        # 新規セッション開始
        training_script = PROJECT_ROOT / "scripts" / "training" / "train_so8t_borea_phi35_bayesian_recovery.py"
        config_file = PROJECT_ROOT / "configs" / "so8t_borea_phi35_bayesian_recovery_config.yaml"
        
        cmd = [
            sys.executable,
            str(training_script),
            "--config", str(config_file)
        ]
        
        logger.info(f"Starting new training session: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        
        if result.returncode == 0:
            logger.info("[OK] Training completed successfully")
            play_audio_notification()
        else:
            logger.error(f"[ERROR] Training failed with exit code {result.returncode}")
        
        return
    
    session_dir = latest_session['session_dir']
    session_info = latest_session['session_info']
    
    # 復旧判定
    if not should_resume(session_info):
        # 新規セッション開始
        training_script = PROJECT_ROOT / "scripts" / "training" / "train_so8t_borea_phi35_bayesian_recovery.py"
        config_file = PROJECT_ROOT / "configs" / "so8t_borea_phi35_bayesian_recovery_config.yaml"
        
        cmd = [
            sys.executable,
            str(training_script),
            "--config", str(config_file)
        ]
        
        logger.info(f"Starting new training session: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        
        if result.returncode == 0:
            logger.info("[OK] Training completed successfully")
            play_audio_notification()
        else:
            logger.error(f"[ERROR] Training failed with exit code {result.returncode}")
        
        return
    
    # 最新チェックポイントの検索
    latest_checkpoint = find_latest_checkpoint(session_dir)
    
    if latest_checkpoint is None:
        logger.warning("No checkpoint found. Starting new training session.")
        # 新規セッション開始
        training_script = PROJECT_ROOT / "scripts" / "training" / "train_so8t_borea_phi35_bayesian_recovery.py"
        config_file = PROJECT_ROOT / "configs" / "so8t_borea_phi35_bayesian_recovery_config.yaml"
        
        cmd = [
            sys.executable,
            str(training_script),
            "--config", str(config_file)
        ]
        
        logger.info(f"Starting new training session: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        
        if result.returncode == 0:
            logger.info("[OK] Training completed successfully")
            play_audio_notification()
        else:
            logger.error(f"[ERROR] Training failed with exit code {result.returncode}")
        
        return
    
    # チェックポイントから復旧
    logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
    
    training_script = PROJECT_ROOT / "scripts" / "training" / "train_so8t_borea_phi35_bayesian_recovery.py"
    config_file = PROJECT_ROOT / "configs" / "so8t_borea_phi35_bayesian_recovery_config.yaml"
    
    cmd = [
        sys.executable,
        str(training_script),
        "--config", str(config_file),
        "--resume", str(latest_checkpoint)
    ]
    
    logger.info(f"Resuming training: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode == 0:
        logger.info("[OK] Training completed successfully")
        play_audio_notification()
    else:
        logger.error(f"[ERROR] Training failed with exit code {result.returncode}")
        play_audio_notification()  # エラー時も音声通知


if __name__ == '__main__':
    main()

