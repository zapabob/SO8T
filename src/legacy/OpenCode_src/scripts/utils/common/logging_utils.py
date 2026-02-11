# -*- coding: utf-8 -*-
"""
Common Logging Utilities
共通ロギングユーティリティ
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logging(
    name: str = "so8t",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    file_format: Optional[str] = None,
    console_format: Optional[str] = None
) -> logging.Logger:
    """
    ロギングのセットアップ

    Args:
        name: ロガー名
        level: ログレベル
        log_file: ログファイルパス（オプション）
        console: コンソール出力有効化
        file_format: ファイルログフォーマット
        console_format: コンソールログフォーマット

    Returns:
        設定済みロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 既存ハンドラをクリア
    logger.handlers.clear()

    # デフォルトフォーマット
    if not file_format:
        file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if not console_format:
        console_format = '%(levelname)s: %(message)s'

    # コンソールハンドラ
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(console_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # ファイルハンドラ
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    名前付きロガーを取得

    Args:
        name: ロガー名

    Returns:
        ロガーインスタンス
    """
    return logging.getLogger(name)


def log_function_call(func_name: str, args: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
    """
    関数呼び出しをログ出力

    Args:
        func_name: 関数名
        args: 引数（オプション）
        logger: ロガー（オプション、デフォルトはso8tロガー）
    """
    if logger is None:
        logger = get_logger("so8t")

    if args:
        logger.debug(f"Calling {func_name} with args: {args}")
    else:
        logger.debug(f"Calling {func_name}")


def log_performance(func_name: str, duration: float, logger: Optional[logging.Logger] = None):
    """
    パフォーマンス情報をログ出力

    Args:
        func_name: 関数名
        duration: 実行時間（秒）
        logger: ロガー（オプション）
    """
    if logger is None:
        logger = get_logger("so8t")

    logger.info(f"Performance: {func_name} completed in {duration:.3f}s")
