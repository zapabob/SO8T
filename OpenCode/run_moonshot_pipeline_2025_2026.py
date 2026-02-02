#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ムーンショットパイプライン再稼働スクリプト
2025-2026最新手法統合版: DeepseekGLPO、mHC多様体、SO8T
"""

import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
import os
import shutil

# --- CRITICAL: Environment setup BEFORE any other imports ---
# Windows specific: Disable torch.compile (Dynamo) entirely
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "0"
os.environ["LC_ALL"] = "C"  # Fix encoding issues often seen on Windows

# Force Unsloth/Transformers to NOT use compiled backend if possible
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1" 

# Clear compilation cache immediately (Main process only)
if __name__ == "__main__":
    cache_dir = Path("unsloth_compiled_cache")
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print("Notice: 🧹 Cleared Unsloth compilation cache.")
        except Exception as e:
            # On Windows, this might fail if a process is still hanging; we ignore and proceed.
            print(f"Notice: ⚠️ Failed to clear cache (might be in use): {e}")

# Monkeypatch torch.compile to be a no-op identity function
# This prevents libraries (like Unsloth) from forcing compilation
import torch
# Monkeypatch torch.compile to be a no-op identity function
import torch
import torch._dynamo
def _no_op_compile(model=None, *args, **kwargs):
    if model is None:
        def decorator(func):
            return func
        return decorator
    return model

# Apply patches GLOBALLY so sub-processes inherit them
torch.compile = _no_op_compile
torch._dynamo.disable = lambda fn=None, recursive=True, **kwargs: fn if fn else (lambda f: f)

# ------------------------------------------------------------

from scripts.pipeline.integrated_moonshot_pipeline_2025_2026 import IntegratedMoonshotPipeline2025_2026
from scripts.utils.startup_manager import StartupManager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('moonshot_pipeline_2025_2026.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="ムーンショットパイプライン再稼働")
    parser.add_argument(
        "--use-existing-datasets",
        action="store_true",
        default=True,
        help="既存データセットを使用（デフォルト: True）"
    )
    parser.add_argument(
        "--collect-new-data",
        action="store_true",
        help="新しいデータを収集（--use-existing-datasetsを無効化）"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="既存データセットを一覧表示"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🚀 ムーンショットパイプライン再稼働開始")
    logger.info("📅 実行日時: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=" * 80)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    pipeline = IntegratedMoonshotPipeline2025_2026()
    startup = StartupManager(Path(__file__))
    
    if args.list_datasets:
        datasets = pipeline.discover_existing_datasets()
        print("\n=== 検出された既存データセット ===")
        for cat, files in datasets.items():
            print(f"\n[{cat.upper()}]")
            for f in files:
                print(f"  - {f}")
        return

    # スタートアップに登録（電源断に備える）
    startup.register()

    try:
        use_existing = args.use_existing_datasets and not args.collect_new_data
        pipeline.execute_full_pipeline(use_existing_datasets=use_existing)
        
        # 正常に終了した場合、スタートアップから削除
        # (execute_full_pipeline 内でチェックポイントのクリーンアップは実施済み)
        logger.info("🎉 全工程が正常に終了しました。自動再開設定を解除します。")
        startup.unregister()
        
    except Exception as e:
        logger.error(f"❌ 実行中にエラーが発生しました: {e}")
        logger.info("ℹ️ 次回の電源投入時に自動的に再開が試行されます。")
        sys.exit(1)

if __name__ == "__main__":
    main()
