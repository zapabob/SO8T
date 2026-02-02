#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Git LFS セットアップスクリプト
モデルファイル巨大化対策としてGit LFSを導入
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitLFSSetup:
    """Git LFSセットアップクラス"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent

    def check_git_lfs_installed(self):
        """Git LFSがインストールされているか確認"""
        try:
            result = subprocess.run(["git", "lfs", "version"],
                                  capture_output=True, text=True, check=True)
            version = result.stdout.strip()
            logger.info(f"✅ Git LFS is installed: {version}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Git LFS is not installed")
            logger.info("Please install Git LFS:")
            logger.info("  Windows: winget install --id GitHub.GitLFS")
            logger.info("  Or download from: https://git-lfs.github.io/")
            return False

    def initialize_git_lfs(self):
        """Git LFSを初期化"""
        try:
            logger.info("Initializing Git LFS...")
            subprocess.run(["git", "lfs", "install"], check=True, cwd=self.project_root)
            logger.info("✅ Git LFS initialized successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to initialize Git LFS: {e}")
            return False

    def track_large_files(self):
        """大容量ファイルをLFS追跡対象に設定"""
        large_file_patterns = [
            "*.gguf",
            "*.GGUF",
            "*.safetensors",
            "*.bin",
            "*.ckpt",
            "*.pth",
            "*.imatrix",
            "*.npy",
            "*.npz",
            "*.h5",
            "*.hdf5"
        ]

        logger.info("Setting up LFS tracking for large files...")

        for pattern in large_file_patterns:
            try:
                subprocess.run(["git", "lfs", "track", pattern],
                             check=True, cwd=self.project_root)
                logger.info(f"✅ Tracking: {pattern}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  Failed to track {pattern}: {e}")

    def migrate_existing_large_files(self):
        """既存の大容量ファイルをLFSに移行"""
        logger.info("Checking for existing large files that need LFS migration...")

        large_files = []

        # 既にGit管理されている大容量ファイルを検索
        try:
            result = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                                  check=True, cwd=self.project_root)
            tracked_files = result.stdout.strip().split('\n')

            for file_path in tracked_files:
                if file_path.strip():
                    full_path = self.project_root / file_path
                    if full_path.exists() and full_path.is_file():
                        size_mb = full_path.stat().st_size / (1024 * 1024)
                        if size_mb > 50:  # 50MB以上のファイル
                            large_files.append((file_path, size_mb))

        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to check tracked files: {e}")
            return

        if large_files:
            logger.info(f"Found {len(large_files)} large files that may need LFS migration:")
            for file_path, size_mb in large_files:
                logger.info(f"  📁 {file_path} ({size_mb:.1f}MB)")

            logger.info("\nTo migrate existing large files to LFS:")
            logger.info("1. Commit any pending changes")
            logger.info("2. Run: git lfs migrate import --include='*.gguf,*.safetensors'")
            logger.info("3. This will rewrite git history - use with caution!")
        else:
            logger.info("✅ No large files found that need migration")

    def verify_setup(self):
        """セットアップの検証"""
        logger.info("Verifying Git LFS setup...")

        # .gitattributesの存在確認
        gitattributes_path = self.project_root / ".gitattributes"
        if gitattributes_path.exists():
            logger.info("✅ .gitattributes file exists")
        else:
            logger.warning("⚠️  .gitattributes file not found")

        # LFS追跡設定の確認
        try:
            result = subprocess.run(["git", "lfs", "ls-files"],
                                  capture_output=True, text=True,
                                  check=True, cwd=self.project_root)
            lfs_files = result.stdout.strip()
            if lfs_files:
                file_count = len(lfs_files.split('\n'))
                logger.info(f"✅ {file_count} files are tracked by LFS")
            else:
                logger.info("ℹ️  No files currently tracked by LFS")
        except subprocess.CalledProcessError:
            logger.warning("⚠️  Could not verify LFS tracking")

    def create_lfs_maintenance_script(self):
        """LFSメンテナンススクリプト作成"""
        maintenance_script = '''#!/bin/bash
# SO8T Git LFS Maintenance Script

echo "🛠️  SO8T Git LFS Maintenance"
echo "=========================="

# LFSファイルの状態確認
echo "Checking LFS file status..."
git lfs ls-files

# LFSストレージ使用量確認
echo -e "\nChecking LFS storage usage..."
du -sh .git/lfs/objects || echo "LFS objects directory not found"

# 未コミットのLFSファイル確認
echo -e "\nChecking for uncommitted LFS files..."
git status --porcelain | grep -E "\\.(gguf|safetensors|bin)$" || echo "No uncommitted LFS files found"

echo -e "\n✅ LFS maintenance check completed"
'''

        script_path = self.project_root / "scripts" / "maintenance" / "lfs_maintenance.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)

        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(maintenance_script)

        # 実行権限付与（Windowsでは効果なしだが、互換性のため）
        try:
            script_path.chmod(0o755)
        except Exception:
            pass

        logger.info(f"✅ Created LFS maintenance script: {script_path}")

def main():
    """メイン実行関数"""
    print("🔧 SO8T Git LFS Setup Starting...")
    print("This will configure Git LFS for large model files.")

    setup = GitLFSSetup()

    # Git LFSインストール確認
    if not setup.check_git_lfs_installed():
        print("\n❌ Git LFS is required but not installed.")
        print("Please install Git LFS and run this script again.")
        sys.exit(1)

    # Git LFS初期化
    if not setup.initialize_git_lfs():
        print("\n❌ Failed to initialize Git LFS.")
        sys.exit(1)

    # 大容量ファイル追跡設定
    setup.track_large_files()

    # 既存ファイル移行確認
    setup.migrate_existing_large_files()

    # 検証
    setup.verify_setup()

    # メンテナンススクリプト作成
    setup.create_lfs_maintenance_script()

    print("\n✅ Git LFS setup completed successfully!")
    print("\nNext steps:")
    print("1. Review .gitattributes file")
    print("2. Consider migrating existing large files if needed")
    print("3. Run 'scripts/maintenance/lfs_maintenance.sh' for regular maintenance")

if __name__ == "__main__":
    main()