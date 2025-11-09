#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T完全自動化マスターパイプライン セットアップスクリプト

Windowsタスクスケジューラへの自動登録、設定ファイルの検証、依存関係の事前チェックを行います。

Usage:
    python scripts/pipelines/setup_master_automated_pipeline.py
"""

import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any

import yaml

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/setup_master_automated_pipeline.log', encoding='utf-8'),
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


def validate_config(config_path: Path) -> bool:
    """設定ファイルの検証"""
    logger.info("="*80)
    logger.info("Validating Configuration File")
    logger.info("="*80)
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 必須設定のチェック
        required_sections = [
            'pipeline',
            'checkpoint',
            'error_handling',
            'task_scheduler'
        ]
        
        for section in required_sections:
            if section not in config:
                logger.error(f"Required section '{section}' not found in config")
                return False
        
        logger.info("[OK] Configuration file is valid")
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate config: {e}")
        return False


def check_dependencies() -> bool:
    """依存関係の事前チェック"""
    logger.info("="*80)
    logger.info("Checking Dependencies")
    logger.info("="*80)
    
    try:
        from scripts.utils.auto_install_dependencies import check_package_installed, REQUIRED_PACKAGES
        
        missing_packages = []
        for package_name in REQUIRED_PACKAGES.keys():
            if not check_package_installed(package_name):
                missing_packages.append(package_name)
        
        if missing_packages:
            logger.warning(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Packages will be installed automatically when pipeline runs")
        else:
            logger.info("[OK] All required packages are installed")
        
        return True
        
    except ImportError:
        logger.warning("Dependencies module not available, skipping check")
        return True
    except Exception as e:
        logger.error(f"Failed to check dependencies: {e}")
        return False


def setup_task_scheduler(config_path: Path) -> bool:
    """Windowsタスクスケジューラへの登録"""
    logger.info("="*80)
    logger.info("Setting up Windows Task Scheduler")
    logger.info("="*80)
    
    if not check_admin_privileges():
        logger.error("[ERROR] Administrator privileges required")
        logger.error("Please run this script as administrator")
        return False
    
    try:
        # master_automated_pipeline.pyのsetup機能を使用
        script_path = PROJECT_ROOT / "scripts" / "pipelines" / "master_automated_pipeline.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--setup",
            "--config", str(config_path)
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


def verify_setup(config_path: Path) -> bool:
    """セットアップ完了確認"""
    logger.info("="*80)
    logger.info("Verifying Setup")
    logger.info("="*80)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        task_name = config.get('task_scheduler', {}).get('task_name', 'SO8T-MasterAutomatedPipeline-AutoStart')
        
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
    logger.info("SO8T Master Automated Pipeline Setup")
    logger.info("="*80)
    
    config_path = PROJECT_ROOT / "configs" / "master_automated_pipeline.yaml"
    
    # ステップ1: 設定ファイル検証
    if not validate_config(config_path):
        logger.error("[FAILED] Configuration validation failed")
        return 1
    
    # ステップ2: 依存関係チェック
    if not check_dependencies():
        logger.warning("[WARNING] Dependency check failed, but continuing...")
    
    # ステップ3: タスクスケジューラ登録
    if not setup_task_scheduler(config_path):
        logger.error("[FAILED] Task scheduler setup failed")
        return 1
    
    # ステップ4: セットアップ確認
    if not verify_setup(config_path):
        logger.error("[FAILED] Setup verification failed")
        return 1
    
    logger.info("="*80)
    logger.info("[SUCCESS] Setup completed successfully!")
    logger.info("="*80)
    logger.info("The pipeline will automatically run on system startup")
    logger.info("To test manually, run: python scripts/pipelines/master_automated_pipeline.py --run")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())






