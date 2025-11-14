#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T再学習進捗ダッシュボードユーティリティ

進捗ログ、セッション情報、チェックポイント情報の読み込み機能
"""

import json
import glob
import re
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
    セッション情報を読み込み（session_info.jsonとtraining_session.jsonの両方に対応）
    
    Args:
        session_dir: セッションディレクトリ
    
    Returns:
        セッション情報辞書、またはNone
    """
    # まずtraining_session.jsonを探す
    training_session_path = session_dir / "training_session.json"
    if training_session_path.exists():
        try:
            with open(training_session_path, 'r', encoding='utf-8') as f:
                session_info = json.load(f)
            return session_info
        except Exception as e:
            print(f"Failed to load training session {training_session_path}: {e}")
    
    # チェックポイントディレクトリ内も探す
    checkpoint_dir = session_dir / "checkpoints"
    if checkpoint_dir.exists():
        training_session_path = checkpoint_dir / "training_session.json"
        if training_session_path.exists():
            try:
                with open(training_session_path, 'r', encoding='utf-8') as f:
                    session_info = json.load(f)
                return session_info
            except Exception as e:
                print(f"Failed to load training session {training_session_path}: {e}")
    
    # 次にsession_info.jsonを探す（後方互換性のため）
    session_info_path = session_dir / "session_info.json"
    if session_info_path.exists():
        try:
            with open(session_info_path, 'r', encoding='utf-8') as f:
                session_info = json.load(f)
            return session_info
        except Exception as e:
            print(f"Failed to load session info {session_info_path}: {e}")
    
    return None


def load_checkpoint_info(session_dir: Path) -> Dict[str, Any]:
    """
    チェックポイント情報を読み込み（HuggingFace形式、時間ベースチェックポイント対応）
    
    Args:
        session_dir: セッションディレクトリ
    
    Returns:
        チェックポイント情報辞書
    """
    checkpoint_info = {
        'rolling_checkpoints': [],
        'final_checkpoint': None,
        'emergency_checkpoints': [],
        'hf_checkpoints': [],  # HuggingFace形式のチェックポイント
        'time_based_checkpoints': [],  # 時間ベースチェックポイント
        'total_count': 0
    }
    
    if not session_dir.exists():
        return checkpoint_info
    
    # チェックポイントディレクトリを探す
    checkpoint_dirs = []
    if (session_dir / "checkpoints").exists():
        checkpoint_dirs.append(session_dir / "checkpoints")
    checkpoint_dirs.append(session_dir)  # セッションディレクトリ自体も検索
    
    for checkpoint_base in checkpoint_dirs:
        # HuggingFace形式のチェックポイント（checkpoint-*ディレクトリ）
        hf_checkpoints = sorted(
            checkpoint_base.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
            reverse=True
        )
        checkpoint_info['hf_checkpoints'].extend([str(p) for p in hf_checkpoints])
        
        # 時間ベースチェックポイント（TimeBasedCheckpointCallbackで保存されたもの）
        time_checkpoints = sorted(
            checkpoint_base.glob("checkpoint_time_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        checkpoint_info['time_based_checkpoints'].extend([str(p) for p in time_checkpoints])
    
    # ローリングチェックポイント
    rolling_checkpoints = sorted(session_dir.glob("checkpoint_rolling_*.pt"), 
                                  key=lambda p: p.stat().st_mtime, reverse=True)
    checkpoint_info['rolling_checkpoints'] = [str(p) for p in rolling_checkpoints]
    
    # 最終チェックポイント
    final_checkpoint = session_dir / "checkpoint_final.pt"
    if final_checkpoint.exists():
        checkpoint_info['final_checkpoint'] = str(final_checkpoint)
    else:
        # チェックポイントディレクトリ内も探す
        for checkpoint_base in checkpoint_dirs:
            final_checkpoint = checkpoint_base / "checkpoint_final.pt"
            if final_checkpoint.exists():
                checkpoint_info['final_checkpoint'] = str(final_checkpoint)
                break
    
    # 緊急チェックポイント
    emergency_checkpoints = sorted(session_dir.glob("checkpoint_emergency_*.pt"),
                                    key=lambda p: p.stat().st_mtime, reverse=True)
    checkpoint_info['emergency_checkpoints'] = [str(p) for p in emergency_checkpoints]
    
    # 総数
    checkpoint_info['total_count'] = (
        len(checkpoint_info['rolling_checkpoints']) +
        (1 if checkpoint_info['final_checkpoint'] else 0) +
        len(checkpoint_info['emergency_checkpoints']) +
        len(checkpoint_info['hf_checkpoints']) +
        len(checkpoint_info['time_based_checkpoints'])
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


def parse_training_log(log_file: Path, max_lines: int = 1000) -> Dict[str, Any]:
    """
    ログファイルを解析して進捗情報を抽出
    
    Args:
        log_file: ログファイルパス
        max_lines: 読み込む最大行数（最新N行）
    
    Returns:
        進捗情報辞書
    """
    info = {
        'dataset_loading': {
            'status': 'not_started',  # not_started, loading, completed
            'progress': 0,
            'total_lines': 0,
            'current_line': 0,
            'loaded_samples': 0,
            'message': ''
        },
        'training': {
            'current_epoch': 0,
            'total_epochs': 0,
            'current_step': 0,
            'total_steps': 0,
            'loss': 0.0,
            'learning_rate': 0.0,
            'status': 'not_started'  # not_started, running, completed
        },
        'latest_logs': [],
        'errors': [],
        'warnings': []
    }
    
    if not log_file.exists():
        return info
    
    try:
        # 最新N行のみを読み込む（効率化のため）
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > max_lines:
                lines = lines[-max_lines:]
        
        # 最新のログエントリを保存
        info['latest_logs'] = lines[-50:] if len(lines) >= 50 else lines
        
        # ログを逆順に解析（最新の情報を優先）
        dataset_loaded_found = False
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            
            # エラーメッセージの検出
            if re.search(r'ERROR|Exception|Traceback|Failed|TypeError|ValueError', line, re.IGNORECASE):
                if line not in info['errors']:
                    info['errors'].insert(0, line)
                    if len(info['errors']) > 20:  # 最大20件
                        info['errors'] = info['errors'][:20]
            
            # 警告メッセージの検出
            if re.search(r'WARNING', line, re.IGNORECASE) and 'ERROR' not in line:
                if line not in info['warnings']:
                    info['warnings'].insert(0, line)
                    if len(info['warnings']) > 20:  # 最大20件
                        info['warnings'] = info['warnings'][:20]
            
            # データセット読み込み完了を先にチェック（優先度が高い）
            match = re.search(r'\[OK\]\s*Loaded\s*(\d+[,.]?\d*)\s*training samples', line)
            if match and not dataset_loaded_found:
                samples_str = match.group(1).replace(',', '')
                info['dataset_loading']['loaded_samples'] = int(float(samples_str))
                info['dataset_loading']['status'] = 'completed'
                info['dataset_loading']['progress'] = 1.0
                # 読み込み完了時は総行数も設定（読み込み済みサンプル数と同じと仮定）
                if info['dataset_loading']['total_lines'] == 0:
                    info['dataset_loading']['total_lines'] = info['dataset_loading']['loaded_samples']
                    info['dataset_loading']['current_line'] = info['dataset_loading']['loaded_samples']
                dataset_loaded_found = True
                continue
            
            # データセット読み込み進捗（完了していない場合のみ）
            if not dataset_loaded_found and ('Loading /think format dataset' in line or 'Loading /think format dataset' in line):
                info['dataset_loading']['status'] = 'loading'
                info['dataset_loading']['message'] = line
            
            # データセット読み込み進捗（詳細）- カンマ区切りの数値にも対応
            match = re.search(r'Loading progress:\s*(\d+[,.]?\d*)/(\d+[,.]?\d*)\s*lines\s*\((\d+)%\)', line)
            if match:
                current_line_str = match.group(1).replace(',', '')
                total_lines_str = match.group(2).replace(',', '')
                progress_pct = int(match.group(3))
                info['dataset_loading']['current_line'] = int(float(current_line_str))
                info['dataset_loading']['total_lines'] = int(float(total_lines_str))
                info['dataset_loading']['progress'] = progress_pct / 100.0
                info['dataset_loading']['status'] = 'loading'
            
            # データセット読み込み完了 - より柔軟なパターンマッチング
            match = re.search(r'\[OK\]\s*Loaded\s*(\d+[,.]?\d*)\s*training samples', line)
            if match:
                samples_str = match.group(1).replace(',', '')
                info['dataset_loading']['loaded_samples'] = int(float(samples_str))
                info['dataset_loading']['status'] = 'completed'
                info['dataset_loading']['progress'] = 1.0
                # 読み込み完了時は総行数も設定（読み込み済みサンプル数と同じと仮定）
                if info['dataset_loading']['total_lines'] == 0:
                    info['dataset_loading']['total_lines'] = info['dataset_loading']['loaded_samples']
                    info['dataset_loading']['current_line'] = info['dataset_loading']['loaded_samples']
            
            # 総行数の検出
            match = re.search(r'Total lines in dataset:\s*(\d+[,.]?\d*)', line)
            if match:
                total_lines_str = match.group(1).replace(',', '')
                info['dataset_loading']['total_lines'] = int(float(total_lines_str))
            
            # 学習開始
            if 'Training Started' in line or 'Epoch' in line and 'Started' in line:
                info['training']['status'] = 'running'
            
            # エポック情報
            match = re.search(r'Epoch\s+(\d+)/(\d+)\s+Started', line)
            if match:
                info['training']['current_epoch'] = int(match.group(1))
                info['training']['total_epochs'] = int(match.group(2))
                info['training']['status'] = 'running'
            
            # 総ステップ数
            match = re.search(r'Total steps:\s*(\d+[,.]?\d*)', line)
            if match:
                total_steps_str = match.group(1).replace(',', '')
                info['training']['total_steps'] = int(float(total_steps_str))
            
            # ステップ進捗（Trainerのログから）
            match = re.search(r'{\'loss\':\s*([\d.]+)', line)
            if match:
                try:
                    info['training']['loss'] = float(match.group(1))
                except:
                    pass
            
            # 学習率
            match = re.search(r'{\'learning_rate\':\s*([\d.e+-]+)', line)
            if match:
                try:
                    info['training']['learning_rate'] = float(match.group(1))
                except:
                    pass
            
            # ステップ番号
            match = re.search(r'{\'epoch\':\s*[\d.]+\s*,\s*\'step\':\s*(\d+)', line)
            if match:
                try:
                    info['training']['current_step'] = int(match.group(1))
                except:
                    pass
            
            # 学習完了
            if 'Training Completed' in line or 'Training finished' in line:
                info['training']['status'] = 'completed'
        
    except Exception as e:
        print(f"Failed to parse training log {log_file}: {e}")
    
    return info


def load_training_session(output_dir: Path) -> Optional[Dict[str, Any]]:
    """
    training_session.jsonを読み込む
    
    Args:
        output_dir: 出力ディレクトリ（training_session.jsonがあるディレクトリ）
    
    Returns:
        セッション情報辞書、またはNone
    """
    # training_session.jsonを探す
    session_file = output_dir / "training_session.json"
    
    if not session_file.exists():
        # チェックポイントディレクトリ内も探す
        checkpoint_dir = output_dir / "checkpoints"
        if checkpoint_dir.exists():
            session_file = checkpoint_dir / "training_session.json"
    
    if not session_file.exists():
        return None
    
    try:
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        return session_data
    except Exception as e:
        print(f"Failed to load training session {session_file}: {e}")
        return None


def get_system_metrics() -> Dict[str, Any]:
    """
    システムメトリクスを取得（CPU、メモリ、GPU）
    
    Returns:
        システムメトリクス辞書
    """
    metrics = {
        'cpu_usage': 0.0,
        'memory_usage': 0.0,
        'gpu_usage': 0.0,
        'gpu_memory_usage': 0.0,
        'gpu_temperature': 0.0,
        'gpu_available': False
    }
    
    try:
        import psutil
        # CPU使用率
        metrics['cpu_usage'] = psutil.cpu_percent(interval=0.1)
        
        # メモリ使用率
        memory = psutil.virtual_memory()
        metrics['memory_usage'] = memory.percent
    except ImportError:
        pass
    except Exception as e:
        print(f"Failed to get CPU/Memory metrics: {e}")
    
    # GPU情報の取得
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # GPU使用率
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics['gpu_usage'] = float(util.gpu)
        
        # GPUメモリ使用率
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        metrics['gpu_memory_usage'] = float(mem_info.used / mem_info.total * 100)
        
        # GPU温度
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            metrics['gpu_temperature'] = float(temp)
        except:
            pass
        
        metrics['gpu_available'] = True
        pynvml.nvmlShutdown()
    except ImportError:
        # pynvmlが利用できない場合、nvidia-smiコマンドを試す
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 4:
                    metrics['gpu_usage'] = float(parts[0])
                    metrics['gpu_memory_usage'] = float(parts[1]) / float(parts[2]) * 100
                    metrics['gpu_temperature'] = float(parts[3])
                    metrics['gpu_available'] = True
        except Exception as e:
            print(f"Failed to get GPU metrics via nvidia-smi: {e}")
    except Exception as e:
        print(f"Failed to get GPU metrics: {e}")
    
    return metrics

