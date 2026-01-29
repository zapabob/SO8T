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

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

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
