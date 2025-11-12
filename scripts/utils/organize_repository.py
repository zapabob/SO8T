"""
リポジトリ整理スクリプト

scripts/ディレクトリを用途別に分類して整理
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List

# プロジェクトルート
ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = ROOT / "scripts"

# 分類マッピング
CATEGORIES = {
    "training": [
        "train_*.py",
        "finetune_*.py",
        "burnin_*.py",
        "burn_in_*.py",
    ],
    "inference": [
        "demo_*.py",
        "infer_*.py",
        "run_*_test.py",
        "test_*.py",
    ],
    "conversion": [
        "convert_*.py",
        "integrate_*.py",
    ],
    "evaluation": [
        "evaluate_*.py",
        "ab_test_*.py",
        "compare_*.py",
    ],
    "data": [
        "clean_*.py",
        "split_*.py",
        "label_*.py",
        "collect_*.py",
    ],
    "api": [
        "serve_*.py",
    ],
    "utils": [
        "check_*.py",
        "setup_*.py",
        "fix_*.py",
        "debug_*.py",
    ],
    "pipelines": [
        "complete_*.py",
        "run_*_pipeline.py",
        "*_pipeline.py",
    ],
}

# 個別ファイルマッピング（パターンマッチで分類できない場合）
SPECIFIC_FILES = {
    "training": [
        "train_so8t_transformer.py",
        "train_so8t_qwen.py",
        "train_so8t_recovery.py",
        "train_phi4_so8t_japanese.py",
        "finetune_borea_japanese.py",
        "burnin_borea_so8t_pet.py",
        "burn_in_and_convert_gguf.py",
    ],
    "inference": [
        "demo_infer.py",
        "demo_so8t_external.py",
        "demo_complete_so8t_system.py",
        "demo_so8t_progress.py",
    ],
    "conversion": [
        "convert_borea_to_gguf.py",
        "convert_distilled_to_gguf.py",
        "convert_hf_to_gguf.py",
        "convert_lightweight_to_gguf.py",
        "convert_qwen2vl_to_so8t.py",
        "convert_qwen2vl_to_so8t_simple.py",
        "convert_qwen3_so8t_8bit_gguf.py",
        "convert_qwen_so8t_8bit_gguf.py",
        "convert_qwen_so8t_lightweight.py",
        "integrate_phi4_so8t.py",
        "integrate_phi4_so8t_lightweight.py",
    ],
    "evaluation": [
        "evaluate_four_class.py",
        "evaluate_model_a_baseline.py",
        "ab_test_borea_phi35.py",
        "compare_swa.py",
    ],
    "data": [
        "clean_japanese_dataset.py",
        "split_dataset.py",
        "label_four_class_dataset.py",
        "collect_japanese_data.py",
        "collect_public_datasets.py",
    ],
    "api": [
        "serve_fastapi.py",
        "serve_think_api.py",
    ],
    "utils": [
        "check_memory.py",
        "setup_auto_resume.py",
        "setup_auto_resume.bat",
        "setup_auto_resume.ps1",
        "setup_auto_resume_quick.ps1",
        "fix_borea_git_lfs.ps1",
        "fix_borea_git_lfs_simple.ps1",
        "debug_cleaning.py",
    ],
    "pipelines": [
        "complete_so8t_pipeline.py",
        "run_so8t_burnin_qc_pipeline.py",
        "run_borea_complete_pipeline.py",
        "run_ab_test_complete.py",
        "run_parallel_train_crawl.py",
    ],
}


def categorize_file(filename: str) -> str:
    """ファイルをカテゴリに分類"""
    # 個別ファイルマッピングを確認
    for category, files in SPECIFIC_FILES.items():
        if filename in files:
            return category
    
    # パターンマッチング
    for category, patterns in CATEGORIES.items():
        for pattern in patterns:
            # 簡単なワイルドカードマッチング
            pattern_regex = pattern.replace("*", ".*")
            import re
            if re.match(pattern_regex, filename):
                return category
    
    # デフォルト: そのまま残す
    return None


def organize_scripts(dry_run: bool = True):
    """scripts/ディレクトリを整理"""
    print(f"[INFO] Organizing scripts directory (dry_run={dry_run})")
    
    # 新しいディレクトリを作成
    for category in CATEGORIES.keys():
        new_dir = SCRIPTS_DIR / category
        if not dry_run:
            new_dir.mkdir(exist_ok=True)
        print(f"[INFO] Created directory: {new_dir}")
    
    # ファイルを分類
    files_to_move = {}
    for file_path in SCRIPTS_DIR.glob("*.py"):
        filename = file_path.name
        category = categorize_file(filename)
        if category:
            if category not in files_to_move:
                files_to_move[category] = []
            files_to_move[category].append(file_path)
    
    # batファイルとps1ファイルも処理
    for ext in ["*.bat", "*.ps1"]:
        for file_path in SCRIPTS_DIR.glob(ext):
            filename = file_path.name
            category = categorize_file(filename)
            if category:
                if category not in files_to_move:
                    files_to_move[category] = []
                files_to_move[category].append(file_path)
    
    # 移動計画を表示
    print("\n[INFO] Files to move:")
    for category, files in files_to_move.items():
        print(f"\n{category}/:")
        for file_path in files:
            print(f"  {file_path.name}")
    
    if dry_run:
        print("\n[INFO] This is a dry run. Set dry_run=False to actually move files.")
        return
    
    # ファイルを移動（git mvを使用）
    import subprocess
    for category, files in files_to_move.items():
        for file_path in files:
            dest = SCRIPTS_DIR / category / file_path.name
            print(f"[INFO] Moving {file_path.name} to {category}/")
            try:
                # git mvを使用して移動
                subprocess.run(
                    ["git", "mv", str(file_path), str(dest)],
                    check=True,
                    cwd=ROOT,
                )
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to move {file_path.name}: {e}")
                # フォールバック: 通常の移動
                shutil.move(str(file_path), str(dest))


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    organize_scripts(dry_run=dry_run)

