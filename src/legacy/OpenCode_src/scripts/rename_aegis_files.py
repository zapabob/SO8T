#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGISファイル名一括変更スクリプト
Rename AEGIS files in bulk
"""

import os
import re
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rename_aegis_files():
    """AEGIS関連ファイルを一括変更"""

    project_root = Path(__file__).parent.parent

    # 変更ルール
    rename_rules = {
        'AEGIS-v2.0-Phi3.5-thinking': 'AEGIS-v2.0-Phi3.5-thinking',
        'aegis-v2.0-phi3.5-thinking': 'aegis-v2.0-phi3.5-thinking',
        'aegis_v2_phi35_thinking': 'aegis_v2_phi35_thinking'
    }

    # 対象ディレクトリ
    target_dirs = [
        '_docs',
        'scripts',
        'configs',
        'huggingface_upload'
    ]

    # 対象ファイル拡張子
    target_extensions = ['.md', '.py', '.yaml', '.yml', '.json', '.bat', '.sh']

    total_files_processed = 0
    total_changes_made = 0

    for target_dir in target_dirs:
        dir_path = project_root / target_dir
        if not dir_path.exists():
            continue

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = Path(root) / file

                # 拡張子チェック
                if file_path.suffix.lower() not in target_extensions:
                    continue

                try:
                    # ファイル内容変更
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    original_content = content
                    changes_in_file = 0

                    # 各ルールで置換
                    for old_name, new_name in rename_rules.items():
                        if old_name in content:
                            content = content.replace(old_name, new_name)
                            changes_in_file += content.count(new_name) - original_content.count(new_name)

                    # 内容が変更された場合のみ保存
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)

                        total_changes_made += changes_in_file
                        logger.info(f"[CHANGED] {file_path.relative_to(project_root)} ({changes_in_file} replacements)")

                    total_files_processed += 1

                except Exception as e:
                    logger.error(f"[ERROR] Failed to process {file_path}: {e}")

    # ファイル名変更
    renamed_files = []

    for target_dir in target_dirs:
        dir_path = project_root / target_dir
        if not dir_path.exists():
            continue

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = Path(root) / file

                # 新しいファイル名生成
                new_filename = file
                for old_name, new_name in rename_rules.items():
                    # ファイル名の大文字小文字を考慮した置換
                    patterns = [
                        old_name,
                        old_name.lower(),
                        old_name.upper(),
                        re.sub(r'[^a-zA-Z0-9]', '_', old_name.lower())
                    ]

                    for pattern in patterns:
                        if pattern in new_filename:
                            new_filename = new_filename.replace(pattern, re.sub(r'[^a-zA-Z0-9]', '_', new_name.lower()))
                            break

                if new_filename != file:
                    new_file_path = file_path.parent / new_filename

                    try:
                        shutil.move(str(file_path), str(new_file_path))
                        renamed_files.append((file_path.relative_to(project_root), new_file_path.relative_to(project_root)))
                        logger.info(f"[RENAMED] {file_path.relative_to(project_root)} -> {new_file_path.relative_to(project_root)}")
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to rename {file_path}: {e}")

    logger.info("\n[SUMMARY] AEGIS file rename completed")
    logger.info(f"Files processed: {total_files_processed}")
    logger.info(f"Content changes made: {total_changes_made}")
    logger.info(f"Files renamed: {len(renamed_files)}")

    if renamed_files:
        logger.info("\nRenamed files:")
        for old_path, new_path in renamed_files:
            logger.info(f"  {old_path} -> {new_path}")

if __name__ == '__main__':
    rename_aegis_files()
