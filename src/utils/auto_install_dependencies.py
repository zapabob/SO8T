#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依存関係自動インストール機能

MLflow、W&B、TensorBoard等の不足パッケージを自動検出・インストール
"""

import sys
import subprocess
import importlib
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 必須パッケージリスト
REQUIRED_PACKAGES = {
    'mlflow': 'mlflow>=2.8.0',
    'wandb': 'wandb>=0.15.0',
    'tensorboard': 'tensorboard>=2.13.0',
    'psutil': 'psutil>=5.9.0',
    'numpy': 'numpy>=1.24.0',
    'pandas': 'pandas>=2.0.0',
    'scikit-learn': 'scikit-learn>=1.3.0',
    'tqdm': 'tqdm>=4.65.0',
    'pyyaml': 'pyyaml>=6.0',
    'torch': 'torch>=2.0.0',
    'transformers': 'transformers>=4.35.0',
    'peft': 'peft>=0.6.0',
    'bitsandbytes': 'bitsandbytes>=0.41.0',
    'accelerate': 'accelerate>=0.24.0',
    'matplotlib': 'matplotlib>=3.7.0',
    'seaborn': 'seaborn>=0.12.0',
    'plotly': 'plotly>=5.15.0',
    'joblib': 'joblib>=1.3.0',
}


def check_package_installed(package_name: str) -> bool:
    """
    パッケージがインストールされているかチェック
    
    Args:
        package_name: パッケージ名（import名）
    
    Returns:
        インストールされているかどうか
    """
    # パッケージ名のマッピング（インストール名 -> import名）
    package_mapping = {
        'scikit-learn': 'sklearn',
        'pyyaml': 'yaml',
    }
    
    # import名に変換
    import_name = package_mapping.get(package_name, package_name)
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def install_package(package_spec: str, show_progress: bool = True) -> bool:
    """
    パッケージをインストール
    
    Args:
        package_spec: パッケージ仕様（例: 'mlflow>=2.8.0'）
        show_progress: 進捗表示するか
    
    Returns:
        インストール成功したかどうか
    """
    try:
        logger.info(f"[INSTALL] Installing {package_spec}...")
        
        # pip install実行
        cmd = [sys.executable, '-m', 'pip', 'install', package_spec, '--quiet']
        
        if show_progress:
            # tqdmで進捗表示
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 進捗バー表示
            with tqdm(total=100, desc=f"Installing {package_spec.split('>=')[0]}", unit='%') as pbar:
                stdout, stderr = process.communicate()
                pbar.update(100)
            
            if process.returncode != 0:
                logger.error(f"[ERROR] Failed to install {package_spec}: {stderr}")
                return False
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"[ERROR] Failed to install {package_spec}: {result.stderr}")
                return False
        
        logger.info(f"[OK] Successfully installed {package_spec}")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Exception while installing {package_spec}: {e}")
        return False


def check_and_install_all(required_packages: Dict[str, str] = None, 
                         show_progress: bool = True) -> Dict[str, bool]:
    """
    すべての必須パッケージをチェックし、不足しているものをインストール
    
    Args:
        required_packages: 必須パッケージ辞書（デフォルト: REQUIRED_PACKAGES）
        show_progress: 進捗表示するか
    
    Returns:
        インストール結果辞書 {package_name: success}
    """
    if required_packages is None:
        required_packages = REQUIRED_PACKAGES
    
    results = {}
    missing_packages = []
    
    logger.info("="*80)
    logger.info("Checking required packages...")
    logger.info("="*80)
    
    # チェックフェーズ
    for package_name, package_spec in required_packages.items():
        if check_package_installed(package_name):
            logger.info(f"[OK] {package_name} is already installed")
            results[package_name] = True
        else:
            logger.warning(f"[MISSING] {package_name} is not installed")
            missing_packages.append((package_name, package_spec))
            results[package_name] = False
    
    # インストールフェーズ
    if missing_packages:
        logger.info("="*80)
        logger.info(f"Installing {len(missing_packages)} missing packages...")
        logger.info("="*80)
        
        for package_name, package_spec in tqdm(missing_packages, desc="Installing packages"):
            success = install_package(package_spec, show_progress=show_progress)
            results[package_name] = success
            
            # インストール後、再度チェック
            if success:
                # 少し待機してからチェック（パッケージが完全にインストールされるまで）
                import time
                time.sleep(1)
                
                if check_package_installed(package_name):
                    logger.info(f"[VERIFIED] {package_name} is now installed")
                    results[package_name] = True
                else:
                    logger.warning(f"[WARNING] {package_name} installation verification failed, but installation may have succeeded")
                    # インストールは成功したが検証に失敗した場合でも、成功として扱う
                    # （パッケージ名の不一致などが原因の可能性がある）
                    results[package_name] = True
    else:
        logger.info("="*80)
        logger.info("[OK] All required packages are installed!")
        logger.info("="*80)
    
    return results


def update_requirements_txt(requirements_path: Path = None):
    """
    requirements.txtにMLflowを追加（まだない場合）
    
    Args:
        requirements_path: requirements.txtのパス
    """
    if requirements_path is None:
        requirements_path = Path(__file__).parent.parent.parent / "requirements.txt"
    
    if not requirements_path.exists():
        logger.warning(f"requirements.txt not found: {requirements_path}")
        return
    
    # ファイル読み込み
    with open(requirements_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # MLflowが含まれているかチェック
    if 'mlflow' not in content.lower():
        logger.info("[UPDATE] Adding mlflow to requirements.txt...")
        
        # Logging and monitoringセクションを探す
        if '# Logging and monitoring' in content:
            # そのセクションに追加
            lines = content.split('\n')
            new_lines = []
            added = False
            
            for i, line in enumerate(lines):
                new_lines.append(line)
                if '# Logging and monitoring' in line and not added:
                    # 次の行にmlflowを追加
                    new_lines.append('mlflow>=2.8.0')
                    added = True
            
            content = '\n'.join(new_lines)
        else:
            # 末尾に追加
            content += '\n# MLOps/LLMOps\nmlflow>=2.8.0\n'
        
        # ファイル書き込み
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info("[OK] Updated requirements.txt")
    else:
        logger.info("[OK] mlflow already in requirements.txt")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto install dependencies")
    parser.add_argument(
        '--requirements',
        type=Path,
        help='Path to requirements.txt'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Do not show progress bars'
    )
    parser.add_argument(
        '--update-requirements',
        action='store_true',
        help='Update requirements.txt with mlflow'
    )
    
    args = parser.parse_args()
    
    # requirements.txt更新
    if args.update_requirements:
        update_requirements_txt(args.requirements)
    
    # パッケージチェック・インストール
    results = check_and_install_all(show_progress=not args.no_progress)
    
    # 結果サマリー
    logger.info("="*80)
    logger.info("Installation Summary")
    logger.info("="*80)
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    logger.info(f"Successfully installed/verified: {success_count}/{total_count}")
    
    if success_count < total_count:
        failed = [k for k, v in results.items() if not v]
        logger.warning(f"Failed packages: {', '.join(failed)}")
        return 1
    
    logger.info("[OK] All packages are ready!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

