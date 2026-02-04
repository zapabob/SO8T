#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合パイプラインダッシュボードユーティリティ

チェックポイント、ブラウザ状態、データ処理状態、A/Bテスト状態の読み込み機能
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from PIL import Image

logger = logging.getLogger(__name__)


def load_unified_pipeline_checkpoint(checkpoint_dir: Path) -> Optional[Dict]:
    """
    統合マスターパイプラインのチェックポイントを読み込み
    
    Args:
        checkpoint_dir: チェックポイントディレクトリ
    
    Returns:
        チェックポイントデータ、またはNone
    """
    if not checkpoint_dir.exists():
        return None
    
    # 最新のチェックポイントを検索
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not checkpoints:
        return None
    
    try:
        with open(checkpoints[0], 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


def load_parallel_scraping_status(output_dir: Path) -> Dict:
    """
    並列スクレイピングの状態を読み込み
    
    Args:
        output_dir: 出力ディレクトリ
    
    Returns:
        スクレイピング状態辞書
    """
    status = {
        'browser_status': {},
        'total_samples': 0,
        'screenshots': []
    }
    
    # ダッシュボード状態ファイルを探す
    dashboard_state_files = list(output_dir.glob("dashboard_state_*.json"))
    if dashboard_state_files:
        latest_state_file = max(dashboard_state_files, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest_state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
                status['browser_status'] = state_data.get('browser_status', {})
                status['total_samples'] = state_data.get('total_samples', 0)
        except Exception as e:
            logger.warning(f"Failed to load dashboard state: {e}")
    
    # スクリーンショットディレクトリを探す
    screenshots_dir = output_dir / "screenshots"
    if screenshots_dir.exists():
        status['screenshots'] = load_browser_screenshots(screenshots_dir)
    
    return status


def load_browser_screenshots(screenshots_dir: Path, max_count: int = 10) -> List[Dict]:
    """
    ブラウザスクリーンショットを読み込み
    
    Args:
        screenshots_dir: スクリーンショットディレクトリ
        max_count: 最大読み込み数
    
    Returns:
        スクリーンショット情報のリスト
    """
    if not screenshots_dir.exists():
        return []
    
    screenshot_files = sorted(
        screenshots_dir.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    screenshots = []
    for screenshot_file in screenshot_files[:max_count]:
        try:
            img = Image.open(screenshot_file)
            browser_num = None
            
            # ファイル名からブラウザ番号を抽出
            filename = screenshot_file.name
            if "browser_" in filename:
                try:
                    browser_num = int(filename.split("browser_")[1].split("_")[0])
                except:
                    pass
            
            screenshots.append({
                'file': screenshot_file.name,
                'path': screenshot_file,
                'image': img,
                'browser_num': browser_num,
                'timestamp': datetime.fromtimestamp(screenshot_file.stat().st_mtime)
            })
        except Exception as e:
            logger.warning(f"Failed to load screenshot {screenshot_file}: {e}")
            continue
    
    return screenshots


def load_data_processing_status(checkpoint_dir: Path) -> Dict:
    """
    データ処理パイプラインの状態を読み込み
    
    Args:
        checkpoint_dir: チェックポイントディレクトリ
    
    Returns:
        データ処理状態辞書
    """
    status = {
        'phase1_data_cleaning': {'status': 'pending', 'samples': 0},
        'phase2_incremental_labeling': {'status': 'pending', 'samples': 0, 'quality': 0.0},
        'phase3_quadruple_classification': {'status': 'pending', 'samples': 0}
    }
    
    # チェックポイントを読み込み
    checkpoint = load_unified_pipeline_checkpoint(checkpoint_dir)
    if checkpoint:
        phase_progress = checkpoint.get('phase_progress', {})
        
        # Phase 1: データクレンジング
        if 'data_cleaning' in phase_progress:
            phase1_data = phase_progress['data_cleaning']
            status['phase1_data_cleaning'] = {
                'status': phase1_data.get('status', 'pending'),
                'samples': phase1_data.get('samples', 0)
            }
        
        # Phase 2: 漸次ラベル付け
        if 'incremental_labeling' in phase_progress:
            phase2_data = phase_progress['incremental_labeling']
            status['phase2_incremental_labeling'] = {
                'status': phase2_data.get('status', 'pending'),
                'samples': phase2_data.get('samples', 0),
                'quality': phase2_data.get('quality', 0.0)
            }
        
        # Phase 3: 四値分類
        if 'quadruple_classification' in phase_progress:
            phase3_data = phase_progress['quadruple_classification']
            status['phase3_quadruple_classification'] = {
                'status': phase3_data.get('status', 'pending'),
                'samples': phase3_data.get('samples', 0)
            }
    
    return status


def load_ab_test_status(checkpoint_dir: Path, results_dir: Path) -> Dict:
    """
    A/Bテストパイプラインの状態を読み込み
    
    Args:
        checkpoint_dir: チェックポイントディレクトリ
        results_dir: 結果ディレクトリ
    
    Returns:
        A/Bテスト状態辞書
    """
    status = {
        'phase1_model_a_gguf': {'status': 'pending'},
        'phase2_train_model_b': {'status': 'pending'},
        'phase3_model_b_gguf': {'status': 'pending'},
        'phase4_ollama_import': {'status': 'pending', 'model_a': False, 'model_b': False},
        'phase5_ab_test': {'status': 'pending'},
        'phase6_visualization': {'status': 'pending'},
        'results': None
    }
    
    # チェックポイントを読み込み
    checkpoint = load_unified_pipeline_checkpoint(checkpoint_dir)
    if checkpoint:
        phase_progress = checkpoint.get('phase_progress', {})
        
        # 各フェーズの状態を取得
        for phase_key in status.keys():
            if phase_key in phase_progress:
                phase_data = phase_progress[phase_key]
                status[phase_key] = {
                    'status': phase_data.get('status', 'pending'),
                    **{k: v for k, v in phase_data.items() if k != 'status'}
                }
    
    # 結果ファイルを読み込み
    results_file = results_dir / "ab_test_results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                status['results'] = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load A/B test results: {e}")
    
    return status


def load_pipeline_logs(log_dir: Path, max_lines: int = 200) -> List[str]:
    """
    パイプラインログを読み込み
    
    Args:
        log_dir: ログディレクトリ
        max_lines: 最大読み込み行数
    
    Returns:
        ログ行のリスト
    """
    log_file = log_dir / "unified_master_pipeline.log"
    
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-max_lines:]  # 最新のN行
    except Exception as e:
        logger.warning(f"Failed to load log file: {e}")
        return []


def calculate_phase_progress(phase_progress: Dict) -> float:
    """
    フェーズの進捗率を計算
    
    Args:
        phase_progress: フェーズ進捗データ
    
    Returns:
        進捗率（0.0-1.0）
    """
    status = phase_progress.get('status', 'pending')
    
    if status == 'completed':
        return 1.0
    elif status == 'running':
        # 実行中の場合は、サンプル数などから推定
        samples = phase_progress.get('samples', 0)
        if samples > 0:
            # 簡易的な進捗計算（実際の実装では、より詳細な情報が必要）
            return min(samples / 1000.0, 0.9)  # 1000サンプルで90%と仮定
        return 0.5
    elif status == 'failed':
        return 0.0
    else:
        return 0.0


def get_phase_status_color(status: str) -> str:
    """
    フェーズステータスに応じた色を返す
    
    Args:
        status: ステータス文字列
    
    Returns:
        色コード
    """
    status_colors = {
        'completed': '#00ff41',
        'running': '#00ffff',
        'failed': '#ff0040',
        'pending': '#888888',
        'skipped': '#ffaa00'
    }
    return status_colors.get(status, '#888888')


def format_duration(seconds: float) -> str:
    """
    経過時間をフォーマット
    
    Args:
        seconds: 経過時間（秒）
    
    Returns:
        フォーマットされた時間文字列
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds_remainder}s"
    elif minutes > 0:
        return f"{minutes}m {seconds_remainder}s"
    else:
        return f"{seconds_remainder}s"

