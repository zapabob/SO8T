#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本番環境起動スクリプト

依存関係自動インストール、環境チェック、パイプライン実行、音声通知を統合
"""

import sys
import os
import subprocess
import logging
import argparse
import signal
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
        logging.FileHandler('logs/production_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 依存関係自動インストール
try:
    from scripts.utils.auto_install_dependencies import check_and_install_all
    AUTO_INSTALL_AVAILABLE = True
except ImportError:
    AUTO_INSTALL_AVAILABLE = False
    logger.warning("Auto install dependencies not available")

# パイプライン
try:
    from scripts.pipelines.complete_data_pipeline import CompleteDataPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    logger.error("Pipeline not available")


def play_audio_notification():
    """音声通知を再生"""
    audio_path = Path(r"C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav")
    
    if not audio_path.exists():
        logger.warning(f"Audio file not found: {audio_path}")
        return False
    
    try:
        import winsound
        winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
        logger.info("[AUDIO] Notification played successfully")
        return True
    except Exception as e:
        logger.warning(f"[AUDIO] Failed to play audio: {e}")
        return False


def check_environment() -> Dict[str, bool]:
    """環境チェック（GPU、メモリ、ディスク容量）"""
    logger.info("="*80)
    logger.info("Environment Check")
    logger.info("="*80)
    
    checks = {
        'gpu_available': False,
        'sufficient_memory': False,
        'sufficient_disk': False,
        'python_version': False
    }
    
    # GPUチェック
    try:
        import torch
        checks['gpu_available'] = torch.cuda.is_available()
        if checks['gpu_available']:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"[OK] GPU available: {gpu_name} ({gpu_memory:.2f} GB)")
        else:
            logger.warning("[WARNING] GPU not available")
    except ImportError:
        logger.warning("[WARNING] PyTorch not installed")
    
    # メモリチェック
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        checks['sufficient_memory'] = available_gb >= 8.0  # 最低8GB必要
        logger.info(f"[MEMORY] Total: {total_gb:.2f} GB, Available: {available_gb:.2f} GB")
        if checks['sufficient_memory']:
            logger.info("[OK] Sufficient memory available")
        else:
            logger.warning("[WARNING] Insufficient memory (need at least 8GB)")
    except ImportError:
        logger.warning("[WARNING] psutil not installed")
    
    # ディスク容量チェック
    try:
        import shutil
        disk = shutil.disk_usage(PROJECT_ROOT)
        free_gb = disk.free / (1024**3)
        checks['sufficient_disk'] = free_gb >= 100.0  # 最低100GB必要
        logger.info(f"[DISK] Free space: {free_gb:.2f} GB")
        if checks['sufficient_disk']:
            logger.info("[OK] Sufficient disk space")
        else:
            logger.warning("[WARNING] Insufficient disk space (need at least 100GB)")
    except Exception as e:
        logger.warning(f"[WARNING] Failed to check disk: {e}")
    
    # Pythonバージョンチェック
    python_version = sys.version_info
    checks['python_version'] = python_version.major == 3 and python_version.minor >= 8
    logger.info(f"[PYTHON] Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if checks['python_version']:
        logger.info("[OK] Python version is compatible")
    else:
        logger.warning("[WARNING] Python 3.8+ required")
    
    logger.info("="*80)
    
    return checks


def install_dependencies(show_progress: bool = True) -> bool:
    """依存関係を自動インストール"""
    if not AUTO_INSTALL_AVAILABLE:
        logger.warning("[INSTALL] Auto install not available, skipping")
        return False
    
    logger.info("="*80)
    logger.info("Installing Dependencies")
    logger.info("="*80)
    
    try:
        results = check_and_install_all(show_progress=show_progress)
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        
        if success_count == total_count:
            logger.info("[OK] All dependencies installed successfully")
            return True
        else:
            logger.warning(f"[WARNING] Some dependencies failed: {success_count}/{total_count}")
            return False
    except Exception as e:
        logger.error(f"[ERROR] Dependency installation failed: {e}")
        return False


def run_pipeline(config_path: Path, resume: bool = True) -> bool:
    """パイプラインを実行"""
    if not PIPELINE_AVAILABLE:
        logger.error("[ERROR] Pipeline not available")
        return False
    
    logger.info("="*80)
    logger.info("Starting Production Pipeline")
    logger.info("="*80)
    
    # 設定読み込み
    import yaml
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        logger.error(f"[ERROR] Config file not found: {config_path}")
        return False
    
    try:
        # パイプライン実行
        pipeline = CompleteDataPipeline(config)
        pipeline.run_pipeline(resume=resume)
        
        logger.info("[OK] Pipeline completed successfully")
        return True
        
    except KeyboardInterrupt:
        logger.warning("[WARNING] Pipeline interrupted by user")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Pipeline failed: {e}", exc_info=True)
        return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Start Production Pipeline")
    parser.add_argument(
        '--config',
        type=Path,
        default=PROJECT_ROOT / 'configs' / 'production_pipeline_config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--skip-install',
        action='store_true',
        help='Skip dependency installation'
    )
    parser.add_argument(
        '--skip-check',
        action='store_true',
        help='Skip environment check'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from checkpoint'
    )
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Do not play audio notification'
    )
    
    args = parser.parse_args()
    
    # シグナルハンドラー設定
    def signal_handler(signum, frame):
        logger.warning(f"Received signal {signum}, exiting...")
        if not args.no_audio:
            play_audio_notification()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)
    
    try:
        # 1. 環境チェック
        if not args.skip_check:
            checks = check_environment()
            if not all(checks.values()):
                logger.warning("[WARNING] Some environment checks failed, but continuing...")
        
        # 2. 依存関係インストール
        if not args.skip_install:
            install_dependencies()
        
        # 3. パイプライン実行
        success = run_pipeline(args.config, resume=not args.no_resume)
        
        # 4. 音声通知
        if not args.no_audio:
            play_audio_notification()
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"[ERROR] Fatal error: {e}", exc_info=True)
        if not args.no_audio:
            play_audio_notification()
        return 1


if __name__ == '__main__':
    sys.exit(main())












