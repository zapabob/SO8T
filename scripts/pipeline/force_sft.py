#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Force Run SFT Phase
SFTフェーズがスキップされる問題へのリカバリースクリプト
"""
import sys
import os
from pathlib import Path
import logging

# プロジェクトルートの設定
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.infrastructure.pipeline.integrated_moonshot_pipeline_2025_2026 import IntegratedMoonshotPipeline2025_2026

# ロギング設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("FORCE START: SFT PHASE (Recovery Mode)")
    logger.info("="*60)
    
    try:
        pipeline = IntegratedMoonshotPipeline2025_2026()
        
        # データセット収集（既存のものを使用）
        logger.info("Discovering datasets...")
        datasets = pipeline.discover_existing_datasets()
        dataset_paths = []
        for k, v in datasets.items():
            dataset_paths.extend(v)
        
        logger.info(f"Found {len(dataset_paths)} dataset files.")
        
        # SFT強制実行
        logger.info("Executing SFT phase directly...")
        pipeline.execute_sft(dataset_paths)
        
        logger.info("SFT Phase execution request completed.")
        
    except Exception as e:
        logger.error(f"Force SFT failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
