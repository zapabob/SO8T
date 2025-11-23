#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
既存ファイルのエンコーディング問題修正スクリプト

既存ファイルのエンコーディング問題を一括修正、バックアップ機能付き
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))

# エンコーディングユーティリティのインポート
try:
    from scripts.utils.encoding_utils import (
        fix_directory_encoding,
        validate_utf8_file,
        convert_to_utf8
    )
    ENCODING_UTILS_AVAILABLE = True
except ImportError as e:
    ENCODING_UTILS_AVAILABLE = False
    logging.error(f"Encoding utils not available: {e}")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fix_encoding_issues.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def fix_single_file(file_path: Path, create_backup: bool = True) -> bool:
    """
    単一ファイルのエンコーディングを修正
    
    Args:
        file_path: ファイルパス
        create_backup: バックアップを作成するか
    
    Returns:
        修正成功フラグ
    """
    if not file_path.exists():
        logger.error(f"[FIX] File not found: {file_path}")
        return False
    
    # UTF-8で既に正しく読み込める場合はスキップ
    if validate_utf8_file(file_path):
        logger.info(f"[FIX] File already UTF-8: {file_path}")
        return True
    
    logger.info(f"[FIX] Converting file to UTF-8: {file_path}")
    
    success = convert_to_utf8(file_path, create_backup=create_backup)
    
    if success:
        # 検証
        if validate_utf8_file(file_path):
            logger.info(f"[FIX] Successfully converted: {file_path}")
            return True
        else:
            logger.warning(f"[FIX] Conversion completed but validation failed: {file_path}")
            return False
    else:
        logger.error(f"[FIX] Failed to convert: {file_path}")
        return False


def fix_directory(
    directory_path: Path,
    backup_dir: Optional[Path] = None,
    file_pattern: str = "*.jsonl",
    recursive: bool = True
) -> dict:
    """
    ディレクトリ内のファイルのエンコーディングを修正
    
    Args:
        directory_path: ディレクトリパス
        backup_dir: バックアップディレクトリ
        file_pattern: ファイルパターン
        recursive: 再帰的に処理するか
    
    Returns:
        修正結果の辞書
    """
    if not ENCODING_UTILS_AVAILABLE:
        logger.error("[FIX] Encoding utils not available")
        return {
            'success': False,
            'error': 'Encoding utils not available'
        }
    
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        logger.error(f"[FIX] Directory not found: {directory_path}")
        return {
            'success': False,
            'error': f'Directory not found: {directory_path}'
        }
    
    logger.info("="*80)
    logger.info(f"[FIX] Starting encoding fix for directory: {directory_path}")
    logger.info(f"[FIX] File pattern: {file_pattern}")
    logger.info(f"[FIX] Recursive: {recursive}")
    if backup_dir:
        logger.info(f"[FIX] Backup directory: {backup_dir}")
    logger.info("="*80)
    
    # バックアップディレクトリの作成
    if backup_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = directory_path.parent / f"{directory_path.name}_backup_{timestamp}"
    
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[FIX] Backup directory created: {backup_dir}")
    
    # ディレクトリ内のファイルを修正
    results = fix_directory_encoding(
        directory_path=directory_path,
        backup_dir=backup_dir,
        file_pattern=file_pattern
    )
    
    logger.info("="*80)
    logger.info("[FIX] Encoding fix completed")
    logger.info(f"[FIX] Total files: {results['total_files']}")
    logger.info(f"[FIX] Converted files: {results['converted_files']}")
    logger.info(f"[FIX] Failed files: {results['failed_files']}")
    if results['errors']:
        logger.warning(f"[FIX] Errors: {len(results['errors'])}")
        for error in results['errors'][:10]:  # 最初の10個のエラーを表示
            logger.warning(f"[FIX]   - {error}")
    logger.info("="*80)
    
    return results


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Fix encoding issues in files")
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input file or directory path'
    )
    parser.add_argument(
        '--backup-dir',
        type=Path,
        help='Backup directory (default: auto-generated)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.jsonl',
        help='File pattern to process (default: *.jsonl)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='Process recursively (default: True)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        logger.error(f"[FIX] Path not found: {input_path}")
        return 1
    
    if input_path.is_file():
        # 単一ファイルの処理
        logger.info(f"[FIX] Processing single file: {input_path}")
        success = fix_single_file(input_path, create_backup=not args.no_backup)
        return 0 if success else 1
    
    elif input_path.is_dir():
        # ディレクトリの処理
        logger.info(f"[FIX] Processing directory: {input_path}")
        results = fix_directory(
            directory_path=input_path,
            backup_dir=args.backup_dir,
            file_pattern=args.pattern,
            recursive=args.recursive
        )
        return 0 if results.get('success', False) else 1
    
    else:
        logger.error(f"[FIX] Invalid path: {input_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())











































































































































