#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T再学習進捗ダッシュボードユーティリティ

進捗ログ、セッション情報、チェックポイント情報の読み込み機能
"""

import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd


def load_progress_logs(progress_logs_dir: Path) -> List[Dict[str, Any]]:
    """
    進捗ログファイルを読み込み
    
    Args:
        progress_logs_dir: 進捗ログディレクトリ
    
    Returns:
        進捗ログのリスト（時系列順）
    """
    if not progress_logs_dir.exists():
        return []
    
    # 進捗ログファイルを検索
    log_files = sorted(progress_logs_dir.glob("progress_*.json"))
    
    logs = []
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
                logs.append(log_data)
        except Exception as e:
            print(f"Failed to load progress log {log_file}: {e}")
            continue
    
    # タイムスタンプでソート
    logs.sort(key=lambda x: x.get('timestamp', ''))
    
    return logs


def load_session_info(session_dir: Path) -> Optional[Dict[str, Any]]:
    """
    セッション情報を読み込み
    
    Args:
        session_dir: セッションディレクトリ
    
    Returns:
        セッション情報辞書、またはNone
    """
    session_info_path = session_dir / "session_info.json"
    
    if not session_info_path.exists():
        return None
    
    try:
        with open(session_info_path, 'r', encoding='utf-8') as f:
            session_info = json.load(f)
        return session_info
    except Exception as e:
        print(f"Failed to load session info {session_info_path}: {e}")
        return None


def load_checkpoint_info(session_dir: Path) -> Dict[str, Any]:
    """
    チェックポイント情報を読み込み
    
    Args:
        session_dir: セッションディレクトリ
    
    Returns:
        チェックポイント情報辞書
    """
    checkpoint_info = {
        'rolling_checkpoints': [],
        'final_checkpoint': None,
        'emergency_checkpoints': [],
        'total_count': 0
    }
    
    if not session_dir.exists():
        return checkpoint_info
    
    # ローリングチェックポイント
    rolling_checkpoints = sorted(session_dir.glob("checkpoint_rolling_*.pt"), 
                                  key=lambda p: p.stat().st_mtime, reverse=True)
    checkpoint_info['rolling_checkpoints'] = [str(p) for p in rolling_checkpoints]
    
    # 最終チェックポイント
    final_checkpoint = session_dir / "checkpoint_final.pt"
    if final_checkpoint.exists():
        checkpoint_info['final_checkpoint'] = str(final_checkpoint)
    
    # 緊急チェックポイント
    emergency_checkpoints = sorted(session_dir.glob("checkpoint_emergency_*.pt"),
                                    key=lambda p: p.stat().st_mtime, reverse=True)
    checkpoint_info['emergency_checkpoints'] = [str(p) for p in emergency_checkpoints]
    
    # 総数
    checkpoint_info['total_count'] = (
        len(checkpoint_info['rolling_checkpoints']) +
        (1 if checkpoint_info['final_checkpoint'] else 0) +
        len(checkpoint_info['emergency_checkpoints'])
    )
    
    return checkpoint_info


def calculate_progress(current_step: int, total_steps: int) -> float:
    """
    進捗率を計算
    
    Args:
        current_step: 現在のステップ
        total_steps: 総ステップ数
    
    Returns:
        進捗率（0.0-1.0）
    """
    if total_steps == 0:
        return 0.0
    return min(current_step / total_steps, 1.0)


def estimate_remaining_time(logs: List[Dict[str, Any]], current_step: int, total_steps: int) -> Optional[str]:
    """
    残り時間を推定
    
    Args:
        logs: 進捗ログのリスト
        current_step: 現在のステップ
        total_steps: 総ステップ数
    
    Returns:
        残り時間の文字列（例: "2h 30m"）、またはNone
    """
    if len(logs) < 2 or total_steps == 0:
        return None
    
    # 最新の2つのログから時間差を計算
    try:
        latest_log = logs[-1]
        previous_log = logs[-2]
        
        latest_time = datetime.fromisoformat(latest_log['timestamp'])
        previous_time = datetime.fromisoformat(previous_log['timestamp'])
        
        time_diff = (latest_time - previous_time).total_seconds()
        step_diff = latest_log['step'] - previous_log['step']
        
        if step_diff == 0:
            return None
        
        # 1ステップあたりの時間を計算
        time_per_step = time_diff / step_diff
        
        # 残りステップ数
        remaining_steps = total_steps - current_step
        
        # 残り時間（秒）
        remaining_seconds = time_per_step * remaining_steps
        
        # 時間と分に変換
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    except Exception as e:
        print(f"Failed to estimate remaining time: {e}")
        return None


def get_latest_session(checkpoint_base: Path) -> Optional[Path]:
    """
    最新セッションディレクトリを取得
    
    Args:
        checkpoint_base: チェックポイントベースディレクトリ
    
    Returns:
        最新セッションディレクトリ、またはNone
    """
    if not checkpoint_base.exists():
        return None
    
    # 全セッションディレクトリを検索
    session_dirs = [d for d in checkpoint_base.iterdir() if d.is_dir()]
    
    if not session_dirs:
        return None
    
    # セッション情報ファイルがあるディレクトリを優先
    latest_session = None
    latest_time = None
    
    for session_dir in session_dirs:
        session_info_path = session_dir / "session_info.json"
        
        if session_info_path.exists():
            try:
                mtime = session_info_path.stat().st_mtime
                if latest_time is None or mtime > latest_time:
                    latest_time = mtime
                    latest_session = session_dir
            except Exception:
                continue
    
    # セッション情報がない場合、最新のディレクトリを使用
    if latest_session is None:
        latest_session = max(session_dirs, key=lambda d: d.stat().st_mtime)
    
    return latest_session


def format_duration(seconds: float) -> str:
    """
    経過時間をフォーマット
    
    Args:
        seconds: 経過時間（秒）
    
    Returns:
        フォーマットされた時間文字列（例: "1h 23m"）
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def get_elapsed_time(session_info: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    経過時間を取得
    
    Args:
        session_info: セッション情報
    
    Returns:
        経過時間の文字列、またはNone
    """
    if session_info is None:
        return None
    
    start_time_str = session_info.get('start_time')
    if not start_time_str:
        return None
    
    try:
        start_time = datetime.fromisoformat(start_time_str)
        current_time = datetime.now()
        elapsed_seconds = (current_time - start_time).total_seconds()
        return format_duration(elapsed_seconds)
    except Exception as e:
        print(f"Failed to calculate elapsed time: {e}")
        return None

