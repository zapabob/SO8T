#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Repository Organizer
リポジトリ整理スクリプト

ファイルを削除せず、機能を維持したまま適切な場所に整理
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RepositoryOrganizer:
    """
    リポジトリ整理クラス
    Repository Organizer Class
    """

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.scripts_path = self.root_path / "scripts"

        # 移動マッピング定義
        self.file_moves = {
            # pipelines/ -> pipeline/ サブディレクトリ
            "pipelines/": {
                "automated": [
                    "aegis_v2_automated_pipeline.py",
                    "auto_start_complete_ab_pipeline.py",
                    "auto_start_complete_pipeline.py",
                    "automated_so8t_pipeline.py",
                    "master_automated_pipeline.py",
                    "run_complete_automated_ab_pipeline.py",
                ],
                "manual": [
                    "run_ab_test_complete.py",
                    "run_borea_complete_pipeline.py",
                    "run_codex_to_so8t_ppo_pipeline.py",
                    "run_complete_pipeline.py",
                    "run_complete_so8t_ab_pipeline.py",
                    "run_pipeline_with_sound.bat",
                    "run_so8t_pipeline.py",
                    "run_so8t_evaluation_and_training.py",
                    "run_train_safety.py",
                    "run_unified_pipeline.bat",
                    "run_web_scraping_data_pipeline.bat",
                ],
                "production": [
                    "download_and_start_production.py",
                    "power_failure_protected_scraping_pipeline.py",
                    "run_production_web_scraping.bat",
                    "run_production_web_scraping.ps1",
                    "start_production_pipeline.bat",
                    "start_production_pipeline.py",
                ],
                "safety": [
                    "nsfw_drug_detection_qlora_training_data_pipeline.py",
                    "run_nsfw_drug_detection_qlora_training_data_pipeline_from_scratch.bat",
                    "run_nsfw_drug_detection_qlora_training_data_pipeline.bat",
                    "run_safety_complete.py",
                    "test_so8t_safety_pipeline.py",
                ],
                "training": [
                    "coding_focused_retraining_pipeline.py",
                    "evaluate_and_retrain_so8t.py",
                    "execute_so8t_training.py",
                    "finetune_and_ab_test_pipeline.py",
                    "prepare_coding_training_data.py",
                    "run_so8t_burnin_qc_pipeline.py",
                    "run_so8t_rtx3060_full_pipeline.bat",
                ],
                "unified": [
                    "unified_auto_scraping_pipeline.py",
                    "unified_master_pipeline.py",
                    "unified_master_pipeline_autostart.bat",
                ],
            },

            # その他の整理
            "data/": {
                "collected": [
                    "extract_coding_dataset.py",
                    "improve_dataset_quality.py",
                    "incremental_labeling_pipeline.py",
                    "parallel_data_processing_pipeline.py",
                    "run_parallel_train_crawl.py",
                    "run_so8t_auto_data_processing.bat",
                    "run_unified_auto_scraping_pipeline.bat",
                    "run_japanese_training_dataset_collection.bat",
                    "so8t_auto_data_processing_pipeline.py",
                    "web_scraping_data_pipeline.py",
                ],
            },
        }

        # 作成するディレクトリ
        self.create_dirs = [
            "scripts/pipeline/automated",
            "scripts/pipeline/manual",
            "scripts/pipeline/production",
            "scripts/pipeline/safety",
            "scripts/pipeline/training",
            "scripts/pipeline/unified",
            "scripts/data/collected",
        ]

    def organize_repository(self) -> bool:
        """
        リポジトリ整理実行
        Organize repository
        """
        logger.info("[ORGANIZE] Starting repository organization...")

        success = True

        try:
            # 1. 必要なディレクトリ作成
            logger.info("[ORGANIZE] Creating necessary directories...")
            if not self._create_directories():
                logger.error("[ORGANIZE] Directory creation failed!")
                success = False

            # 2. ファイル移動
            logger.info("[ORGANIZE] Moving files to appropriate locations...")
            if not self._move_files():
                logger.error("[ORGANIZE] File moving failed!")
                success = False

            # 3. 重複ファイルの統合
            logger.info("[ORGANIZE] Consolidating duplicate files...")
            if not self._consolidate_duplicates():
                logger.error("[ORGANIZE] Duplicate consolidation failed!")
                success = False

            # 4. 空ディレクトリのクリーンアップ
            logger.info("[ORGANIZE] Cleaning up empty directories...")
            if not self._cleanup_empty_dirs():
                logger.error("[ORGANIZE] Directory cleanup failed!")
                success = False

            # 5. 整理結果の検証
            logger.info("[ORGANIZE] Validating organization...")
            if not self._validate_organization():
                logger.error("[ORGANIZE] Organization validation failed!")
                success = False

        except Exception as e:
            logger.error(f"[ORGANIZE] Organization failed with error: {e}")
            success = False

        if success:
            logger.info("[SUCCESS] Repository organization completed!")
        else:
            logger.error("[FAILED] Repository organization failed!")

        return success

    def _create_directories(self) -> bool:
        """必要なディレクトリ作成"""
        try:
            for dir_path in self.create_dirs:
                full_path = self.root_path / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"[DIRS] Created directory: {dir_path}")

            return True

        except Exception as e:
            logger.error(f"[DIRS] Directory creation failed: {e}")
            return False

    def _move_files(self) -> bool:
        """ファイル移動実行"""
        try:
            for source_dir, categories in self.file_moves.items():
                for category, files in categories.items():
                    for filename in files:
                        source_path = self.scripts_path / source_dir / filename

                        if source_path.exists():
                            if "pipeline" in source_dir:
                                dest_path = self.scripts_path / "pipeline" / category / filename
                            else:
                                dest_path = self.scripts_path / source_dir / category / filename

                            # 移動実行
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(source_path), str(dest_path))
                            logger.info(f"[MOVE] {source_path} -> {dest_path}")
                        else:
                            logger.warning(f"[MOVE] Source file not found: {source_path}")

            return True

        except Exception as e:
            logger.error(f"[MOVE] File moving failed: {e}")
            return False

    def _consolidate_duplicates(self) -> bool:
        """重複ファイルの統合"""
        try:
            # 類似ファイルの特定と統合
            duplicates_to_remove = []

            # 例: 類似したパイプラインスクリプトの統合
            pipeline_duplicates = [
                ("scripts/pipeline/manual/run_complete_pipeline.py", "scripts/pipeline/manual/run_borea_complete_pipeline.py"),
                ("scripts/pipeline/training/execute_so8t_training.py", "scripts/pipeline/training/finetune_and_ab_test_pipeline.py"),
            ]

            for keep_file, remove_file in pipeline_duplicates:
                keep_path = self.root_path / keep_file
                remove_path = self.root_path / remove_file

                if keep_path.exists() and remove_path.exists():
                    # ファイルサイズを比較して小さい方を削除
                    if keep_path.stat().st_size >= remove_path.stat().st_size:
                        duplicates_to_remove.append(remove_path)
                        logger.info(f"[DUPLICATE] Marked for removal: {remove_path}")
                    else:
                        # 入れ替え
                        temp_path = remove_path.with_suffix('.temp')
                        shutil.move(str(keep_path), str(temp_path))
                        shutil.move(str(remove_path), str(keep_path))
                        shutil.move(str(temp_path), str(remove_path))
                        duplicates_to_remove.append(remove_path)
                        logger.info(f"[DUPLICATE] Swapped and marked for removal: {remove_path}")

            # 重複ファイルを削除
            for dup_file in duplicates_to_remove:
                if dup_file.exists():
                    dup_file.unlink()
                    logger.info(f"[DUPLICATE] Removed duplicate: {dup_file}")

            return True

        except Exception as e:
            logger.error(f"[DUPLICATE] Duplicate consolidation failed: {e}")
            return False

    def _cleanup_empty_dirs(self) -> bool:
        """空ディレクトリのクリーンアップ"""
        try:
            # pipelinesディレクトリが空になったら削除
            pipelines_dir = self.scripts_path / "pipelines"
            if pipelines_dir.exists() and not any(pipelines_dir.iterdir()):
                pipelines_dir.rmdir()
                logger.info(f"[CLEANUP] Removed empty directory: {pipelines_dir}")

            return True

        except Exception as e:
            logger.error(f"[CLEANUP] Directory cleanup failed: {e}")
            return False

    def _validate_organization(self) -> bool:
        """整理結果の検証"""
        try:
            # 主要ディレクトリの存在確認
            required_dirs = [
                "scripts/pipeline/automated",
                "scripts/pipeline/manual",
                "scripts/pipeline/production",
                "scripts/pipeline/safety",
                "scripts/pipeline/training",
                "scripts/pipeline/unified",
                "scripts/data/collected",
            ]

            for dir_path in required_dirs:
                full_path = self.root_path / dir_path
                if not full_path.exists():
                    logger.warning(f"[VALIDATE] Directory not found: {dir_path}")
                    return False
                else:
                    file_count = len(list(full_path.glob("*")))
                    logger.info(f"[VALIDATE] ✓ {dir_path}: {file_count} files")

            # 重要なファイルの存在確認
            critical_files = [
                "scripts/setup/setup_pipeline_environment.py",
                "scripts/evaluation/comprehensive_llm_benchmark.py",
                "scripts/training/train_alpha_gate_sigmoid_bayesian.py",
            ]

            for file_path in critical_files:
                full_path = self.root_path / file_path
                if not full_path.exists():
                    logger.error(f"[VALIDATE] Critical file missing: {file_path}")
                    return False
                else:
                    logger.info(f"[VALIDATE] ✓ {file_path}")

            return True

        except Exception as e:
            logger.error(f"[VALIDATE] Organization validation failed: {e}")
            return False


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="SO8T Repository Organizer"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be organized without actually doing it'
    )
    parser.add_argument(
        '--root-path',
        type=str,
        default='.',
        help='Root path of the repository'
    )

    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Would organize repository structure...")
        print("This would:")
        print("- Create organized subdirectories under scripts/pipeline/")
        print("- Move pipeline files to appropriate categories")
        print("- Consolidate duplicate files")
        print("- Clean up empty directories")
        print("- Validate final organization")
        return

    # 整理実行
    organizer = RepositoryOrganizer(args.root_path)
    success = organizer.organize_repository()

    if success:
        print("[SUCCESS] Repository organization completed successfully!")
        print("Next steps:")
        print("1. Test critical scripts still work")
        print("2. Update any hardcoded paths in scripts")
        print("3. Commit the reorganized structure")
    else:
        print("[FAILED] Repository organization failed!")
        exit(1)


if __name__ == '__main__':
    main()
