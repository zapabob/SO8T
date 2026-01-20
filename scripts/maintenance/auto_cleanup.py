#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8Tプロジェクト定期クリーニングスクリプト
大容量一時ファイルの自動削除とメンテナンス
"""

import os
import shutil
import glob
import logging
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SO8TAutoCleanup:
    """SO8T自動クリーニングクラス"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.cleanup_stats = {
            "files_removed": 0,
            "dirs_removed": 0,
            "space_freed_mb": 0.0
        }

    def run_full_cleanup(self):
        """完全クリーニング実行"""
        logger.info("🧹 Starting SO8T automatic cleanup...")

        start_time = datetime.now()

        # 各種クリーニング実行
        self.cleanup_temp_dirs()
        self.cleanup_cache_dirs()
        self.cleanup_old_logs()
        self.cleanup_build_artifacts()
        self.cleanup_large_temp_files()

        end_time = datetime.now()
        duration = end_time - start_time

        self.print_cleanup_report(duration)

    def cleanup_temp_dirs(self):
        """一時ディレクトリクリーニング"""
        logger.info("Cleaning temporary directories...")

        temp_patterns = [
            "temp_*",
            "tmp_*",
            "*.tmp",
            "__pycache__",
            ".pytest_cache",
            "node_modules/.cache"
        ]

        for pattern in temp_patterns:
            for path in self.project_root.rglob(pattern):
                if path.is_dir():
                    try:
                        size_mb = self.get_dir_size_mb(path)
                        shutil.rmtree(path)
                        self.cleanup_stats["dirs_removed"] += 1
                        self.cleanup_stats["space_freed_mb"] += size_mb
                        logger.info(f"  🗑️  Removed temp dir: {path} ({size_mb:.1f}MB)")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Failed to remove {path}: {e}")
                elif path.is_file():
                    try:
                        size_mb = path.stat().st_size / (1024 * 1024)
                        path.unlink()
                        self.cleanup_stats["files_removed"] += 1
                        self.cleanup_stats["space_freed_mb"] += size_mb
                        logger.info(f"  🗑️  Removed temp file: {path} ({size_mb:.1f}MB)")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Failed to remove {path}: {e}")

    def cleanup_cache_dirs(self):
        """キャッシュディレクトリクリーニング"""
        logger.info("Cleaning cache directories...")

        cache_dirs = [
            ".cache",
            ".pytest_cache",
            "__pycache__",
            "external/**/__pycache__",
            "models/**/__pycache__",
            "scripts/**/__pycache__"
        ]

        for cache_dir in cache_dirs:
            for path in self.project_root.glob(cache_dir):
                if path.is_dir():
                    try:
                        size_mb = self.get_dir_size_mb(path)
                        shutil.rmtree(path)
                        self.cleanup_stats["dirs_removed"] += 1
                        self.cleanup_stats["space_freed_mb"] += size_mb
                        logger.info(f"  🗑️  Removed cache dir: {path} ({size_mb:.1f}MB)")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Failed to remove cache {path}: {e}")

    def cleanup_old_logs(self, days_old=30):
        """古いログファイル削除"""
        logger.info(f"Cleaning logs older than {days_old} days...")

        log_patterns = [
            "**/*.log",
            "**/*.log.*",
            "logs/**/*.json",
            "_docs/**/*.md"
        ]

        cutoff_date = datetime.now() - timedelta(days=days_old)

        for pattern in log_patterns:
            for log_file in self.project_root.glob(pattern):
                if log_file.is_file():
                    try:
                        file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                        if file_mtime < cutoff_date:
                            # _docsは重要な実装ログなので慎重に
                            if "_docs" in str(log_file):
                                # 最新の実装ログ以外は削除
                                if not self.is_recent_implementation_log(log_file):
                                    size_mb = log_file.stat().st_size / (1024 * 1024)
                                    log_file.unlink()
                                    self.cleanup_stats["files_removed"] += 1
                                    self.cleanup_stats["space_freed_mb"] += size_mb
                                    logger.info(f"  🗑️  Removed old doc: {log_file} ({size_mb:.1f}MB)")
                            else:
                                size_mb = log_file.stat().st_size / (1024 * 1024)
                                log_file.unlink()
                                self.cleanup_stats["files_removed"] += 1
                                self.cleanup_stats["space_freed_mb"] += size_mb
                                logger.info(f"  🗑️  Removed old log: {log_file} ({size_mb:.1f}MB)")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Failed to check/remove log {log_file}: {e}")

    def cleanup_build_artifacts(self):
        """ビルドアーティファクトクリーニング"""
        logger.info("Cleaning build artifacts...")

        build_patterns = [
            "**/build",
            "**/dist",
            "**/*.egg-info",
            "**/.eggs",
            "**/*.so",
            "**/*.pyd",
            "**/*.dll"
        ]

        for pattern in build_patterns:
            for path in self.project_root.glob(pattern):
                if path.is_dir():
                    try:
                        size_mb = self.get_dir_size_mb(path)
                        shutil.rmtree(path)
                        self.cleanup_stats["dirs_removed"] += 1
                        self.cleanup_stats["space_freed_mb"] += size_mb
                        logger.info(f"  🗑️  Removed build dir: {path} ({size_mb:.1f}MB)")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Failed to remove build dir {path}: {e}")

    def cleanup_large_temp_files(self, size_threshold_mb=100):
        """大容量一時ファイルクリーニング"""
        logger.info(f"Cleaning files larger than {size_threshold_mb}MB...")

        large_files = []

        # 全ファイルを走査して大容量ファイルを特定
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    if size_mb > size_threshold_mb:
                        large_files.append((file_path, size_mb))
                except Exception:
                    continue

        # 大容量ファイルを削除（慎重に）
        for file_path, size_mb in large_files:
            file_str = str(file_path)

            # 重要なファイルは保護
            protected_patterns = [
                "README.md",
                ".git",
                "requirements.txt",
                "pyproject.toml",
                "models/**/config.json",
                "models/**/tokenizer.json"
            ]

            is_protected = any(pattern in file_str for pattern in protected_patterns)

            if not is_protected:
                try:
                    file_path.unlink()
                    self.cleanup_stats["files_removed"] += 1
                    self.cleanup_stats["space_freed_mb"] += size_mb
                    logger.info(f"  🗑️  Removed large file: {file_path} ({size_mb:.1f}MB)")
                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to remove large file {file_path}: {e}")
            else:
                logger.info(f"  🛡️  Protected large file: {file_path} ({size_mb:.1f}MB)")

    def is_recent_implementation_log(self, log_file):
        """最近の実装ログかどうか判定"""
        # 最新の10件の実装ログは保持
        try:
            filename = log_file.name
            if "実装完了ログ" in filename or "implementation" in filename.lower():
                # 日付部分を抽出して比較
                import re
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
                if date_match:
                    file_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                    days_old = (datetime.now() - file_date).days
                    return days_old <= 90  # 90日以内の実装ログは保持
        except Exception:
            pass
        return False

    def get_dir_size_mb(self, dir_path):
        """ディレクトリサイズ計算（MB）"""
        total_size = 0
        try:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        return total_size / (1024 * 1024)

    def print_cleanup_report(self, duration):
        """クリーニングレポート出力"""
        logger.info("\n" + "="*50)
        logger.info("🧹 SO8T Cleanup Report")
        logger.info("="*50)
        logger.info(f"Duration: {duration}")
        logger.info(f"Files removed: {self.cleanup_stats['files_removed']}")
        logger.info(f"Directories removed: {self.cleanup_stats['dirs_removed']}")
        logger.info(f"Space freed: {self.cleanup_stats['space_freed_mb']:.1f} MB")
        logger.info("="*50)

        # 大容量解放の場合は警告
        if self.cleanup_stats['space_freed_mb'] > 1000:
            logger.warning("⚠️  Large amount of space freed. Consider implementing automated cleanup schedule.")

def main():
    """メイン実行関数"""
    print("🧹 SO8T Automatic Cleanup Starting...")
    print("This will remove temporary files, caches, and old logs.")
    print("Important files will be preserved.")

    # ユーザー確認（実際の運用ではコメントアウト）
    # response = input("Continue? (y/N): ")
    # if response.lower() != 'y':
    #     print("Cleanup cancelled.")
    #     return

    cleaner = SO8TAutoCleanup()
    cleaner.run_full_cleanup()

    print("\n✅ SO8T cleanup completed successfully!")

if __name__ == "__main__":
    main()