"""
ロギングユーティリティモジュール。

Features:
    - 構造化ロギングサポート
    - モジュール別ログレベル設定
    - ファイル出力とコンソール出力の両対応
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import coloredlogs


DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "so8t",
    *,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    colored: bool = True,
) -> logging.Logger:
    """ロガーをセットアップする。

    Args:
        name: ロガー名
        level: ログレベル (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: ログファイルパス（Noneの場合はファイル出力なし）
        log_format: ログフォーマット
        date_format: 日付フォーマット
        colored: カラー出力有効化

    Returns:
        セットアップ済みのロガー

    Example:
        >>> logger = setup_logger("kromhc", level="DEBUG")
        >>> logger.info("Hello, World!")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        log_format or DEFAULT_FORMAT,
        date_format or DEFAULT_DATE_FORMAT,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if colored and sys.platform != "win32":
        try:
            coloredlogs.install(
                logger=name,
                level=level,
                fmt=log_format or DEFAULT_FORMAT,
                datefmt=date_format or DEFAULT_DATE_FORMAT,
            )
        except Exception:
            pass

    return logger


def get_logger(name: str) -> logging.Logger:
    """ロガーを取得する。

    Args:
        name: ロガー名（モジュール名）

    Returns:
        ロガー

    Example:
        >>> logger = get_logger("kromhc.core")
        >>> logger.info("Info message")
    """
    return logging.getLogger(name)


class LoggerMixin:
    """ロガーミキシン。

    Attributes:
        logger: インスタンス固有のロガー
    """

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger


def log_execution_time(logger: Optional[logging.Logger] = None):
    """関数実行時間をログに記録するデコレータ。

    Args:
        logger: 使用するロガー（Noneの場合はルートロガー）

    Returns:
        デコレータ
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            import time

            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            log = logger or logging.getLogger(func.__module__)
            log.info(f"{func.__name__} 実行時間: {elapsed:.4f}秒")
            return result

        return wrapper

    return decorator
