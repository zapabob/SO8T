#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T完全バックグラウンド並列パイプラインマネージャー セットアップスクリプト

Windowsタスクスケジューラへの自動登録、依存関係の事前チェックを行います。

Usage:
    python scripts/data/setup_parallel_pipeline_manager.py
"""

import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/setup_parallel_pipeline_manager.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_admin_privileges() -> bool:
    """管理者権限をチェック"""
    try:
        # Windowsの場合、net sessionコマンドで管理者権限をチェック
        result = subprocess.run(
            ["net", "session"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        # 非Windows環境やエラー時はFalseを返す
        return False


def check_dependencies() -> bool:
    """依存関係の事前チェック"""
    logger.info("="*80)
    logger.info("Checking Dependencies")
    logger.info("="*80)
    
    try:
        # 必要なパッケージのチェック
        required_packages = ['psutil', 'playwright']
        missing_packages = []
        
        for package_name in required_packages:
            try:
                __import__(package_name)
            except ImportError:
                missing_packages.append(package_name)
        
        if missing_packages:
            logger.warning(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Please install missing packages before running the pipeline")
        else:
            logger.info("[OK] All required packages are installed")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to check dependencies: {e}")
        return False


def setup_task_scheduler() -> bool:
    """Windowsタスクスケジューラへの登録"""
    logger.info("="*80)
    logger.info("Setting up Windows Task Scheduler")
    logger.info("="*80)
    
    if not check_admin_privileges():
        logger.error("[ERROR] Administrator privileges required")
        logger.error("Please run this script as administrator")
        return False
    
    try:
        # parallel_pipeline_manager.pyのsetup機能を使用
        script_path = PROJECT_ROOT / "scripts" / "data" / "parallel_pipeline_manager.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--setup"
        ]
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("[OK] Task scheduler setup completed")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Failed to setup task scheduler: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Exception during task scheduler setup: {e}")
        return False


def verify_setup() -> bool:
    """セットアップ完了確認"""
    logger.info("="*80)
    logger.info("Verifying Setup")
    logger.info("="*80)
    
    try:
        task_name = 'SO8T-ParallelPipelineManager-AutoStart'
        
        # タスクの存在確認
        result = subprocess.run(
            ["schtasks", "/query", "/tn", task_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"[OK] Task '{task_name}' is registered")
            
            # タスクの詳細を表示
            logger.info("Task details:")
            subprocess.run(
                ["schtasks", "/query", "/tn", task_name, "/fo", "list", "/v"],
                check=False
            )
            
            return True
        else:
            logger.error(f"[ERROR] Task '{task_name}' not found")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] Failed to verify setup: {e}")
        return False


def main():
    """メイン関数"""
    logger.info("="*80)
    logger.info("SO8T Parallel Pipeline Manager Setup")
    logger.info("="*80)
    
    # ステップ1: 依存関係チェック
    if not check_dependencies():
        logger.warning("[WARNING] Dependency check failed, but continuing...")
    
    # ステップ2: タスクスケジューラ登録
    if not setup_task_scheduler():
        logger.error("[FAILED] Task scheduler setup failed")
        return 1
    
    # ステップ3: セットアップ確認
    if not verify_setup():
        logger.error("[FAILED] Setup verification failed")
        return 1
    
    logger.info("="*80)
    logger.info("[SUCCESS] Setup completed successfully!")
    logger.info("="*80)
    logger.info("The parallel pipeline manager will automatically run on system startup")
    logger.info("To test manually, run: python scripts/data/parallel_pipeline_manager.py --run --daemon")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

