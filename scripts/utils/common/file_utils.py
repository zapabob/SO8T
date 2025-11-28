# -*- coding: utf-8 -*-
"""
Common File Utilities
共通ファイルユーティリティ
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    ディレクトリが存在することを保証

    Args:
        path: ディレクトリパス

    Returns:
        Pathオブジェクト
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_file_write(content: str, file_path: Union[str, Path], encoding: str = 'utf-8') -> bool:
    """
    安全にファイルに書き込み（バックアップ作成）

    Args:
        content: 書き込む内容
        file_path: ファイルパス
        encoding: エンコーディング

    Returns:
        成功かどうか
    """
    file_path = Path(file_path)

    try:
        # バックアップ作成
        if file_path.exists():
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")

        # 書き込み
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)

        logger.info(f"Safely wrote to {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to write to {file_path}: {e}")
        return False


def atomic_write(content: str, file_path: Union[str, Path], encoding: str = 'utf-8') -> bool:
    """
    アトミックにファイル書き込み（一時ファイル使用）

    Args:
        content: 書き込む内容
        file_path: ファイルパス
        encoding: エンコーディング

    Returns:
        成功かどうか
    """
    file_path = Path(file_path)

    try:
        # 一時ファイル作成
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=file_path.parent,
            prefix=file_path.name + '.tmp.',
            delete=False,
            encoding=encoding
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        # アトミック移動
        temp_path.replace(file_path)

        logger.info(f"Atomically wrote to {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to atomically write to {file_path}: {e}")
        return False


def safe_remove(path: Union[str, Path]) -> bool:
    """
    安全にファイル/ディレクトリを削除

    Args:
        path: 削除するパス

    Returns:
        成功かどうか
    """
    path = Path(path)

    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

        logger.info(f"Safely removed {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to remove {path}: {e}")
        return False


def get_file_size(path: Union[str, Path]) -> Optional[int]:
    """
    ファイルサイズを取得

    Args:
        path: ファイルパス

    Returns:
        ファイルサイズ（バイト）またはNone
    """
    try:
        return Path(path).stat().st_size
    except Exception:
        return None


def get_directory_size(path: Union[str, Path]) -> int:
    """
    ディレクトリサイズを取得（再帰的）

    Args:
        path: ディレクトリパス

    Returns:
        合計サイズ（バイト）
    """
    total_size = 0
    path = Path(path)

    try:
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    except Exception:
        return 0


def find_files_by_extension(directory: Union[str, Path], extension: str) -> list[Path]:
    """
    指定拡張子のファイルを検索

    Args:
        directory: 検索ディレクトリ
        extension: 拡張子（ドット付き）

    Returns:
        ファイルパスのリスト
    """
    directory = Path(directory)
    return list(directory.rglob(f'*{extension}'))


def create_hardlink_or_copy(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    ハードリンクを作成、失敗したらコピー

    Args:
        src: ソースパス
        dst: デスティネーションパス

    Returns:
        成功かどうか
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        # ハードリンクを試行
        os.link(str(src), str(dst))
        logger.debug(f"Created hardlink: {src} -> {dst}")
        return True
    except OSError:
        try:
            # コピーにフォールバック
            shutil.copy2(str(src), str(dst))
            logger.debug(f"Copied file: {src} -> {dst}")
            return True
        except Exception as e:
            logger.error(f"Failed to create link or copy: {e}")
            return False


def is_file_locked(file_path: Union[str, Path]) -> bool:
    """
    ファイルがロックされているか確認

    Args:
        file_path: チェックするファイルパス

    Returns:
        ロックされているかどうか
    """
    file_path = Path(file_path)

    try:
        with open(file_path, 'r+'):
            return False
    except IOError:
        return True


