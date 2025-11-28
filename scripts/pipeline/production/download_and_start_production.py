#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大規模データダウンロード + 本番環境起動スクリプト

D:/webdatasetに大規模データをダウンロードしてから本番環境を起動
"""

import sys
import logging
import argparse
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 大規模データダウンローダー
try:
    from scripts.data.download_large_datasets import LargeDatasetDownloader
    DOWNLOADER_AVAILABLE = True
except ImportError:
    DOWNLOADER_AVAILABLE = False
    logger.warning("Large dataset downloader not available")

# 本番環境起動スクリプト
try:
    from scripts.pipelines.start_production_pipeline import main as start_production_main
    PRODUCTION_AVAILABLE = True
except ImportError:
    PRODUCTION_AVAILABLE = False
    logger.error("Production pipeline not available")


def download_datasets(datasets: list = None, max_samples: int = None):
    """大規模データセットをダウンロード"""
    if not DOWNLOADER_AVAILABLE:
        logger.error("[ERROR] Dataset downloader not available")
        return False
    
    logger.info("="*80)
    logger.info("Downloading Large Datasets to D:\\webdataset")
    logger.info("="*80)
    
    downloader = LargeDatasetDownloader(base_dir=Path(r"D:\webdataset"))
    
    if datasets is None:
        # デフォルトデータセットをダウンロード
        datasets = ['wikipedia_ja', 'cc100_ja']
    
    try:
        from scripts.data.download_large_datasets import DEFAULT_DATASETS
        
        configs = []
        for dataset_name in datasets:
            if dataset_name in DEFAULT_DATASETS:
                configs.append(DEFAULT_DATASETS[dataset_name])
            else:
                logger.warning(f"[WARNING] Unknown dataset: {dataset_name}")
        
        if configs:
            results = downloader.download_multiple_datasets(
                configs,
                max_samples_per_dataset=max_samples
            )
            logger.info(f"[OK] Downloaded {len(results)} datasets")
            return True
        else:
            logger.warning("[WARNING] No valid datasets to download")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] Failed to download datasets: {e}")
        return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Download Datasets and Start Production Pipeline")
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download datasets, do not start pipeline'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip dataset download, start pipeline directly'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['wikipedia_ja', 'cc100_ja', 'oscar_ja', 'mc4_ja'],
        default=['wikipedia_ja', 'cc100_ja'],
        help='Datasets to download'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum samples per dataset'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=PROJECT_ROOT / 'configs' / 'production_pipeline_config.yaml',
        help='Production pipeline config file'
    )
    
    args = parser.parse_args()
    
    # データセットダウンロード
    if not args.skip_download:
        success = download_datasets(args.datasets, args.max_samples)
        if not success:
            logger.warning("[WARNING] Dataset download failed, but continuing...")
    
    if args.download_only:
        logger.info("[OK] Dataset download completed")
        return 0
    
    # 本番環境起動
    if not PRODUCTION_AVAILABLE:
        logger.error("[ERROR] Production pipeline not available")
        return 1
    
    logger.info("="*80)
    logger.info("Starting Production Pipeline")
    logger.info("="*80)
    
    # 本番環境起動スクリプトを実行
    import sys
    sys.argv = ['start_production_pipeline.py', '--config', str(args.config)]
    return start_production_main()


if __name__ == '__main__':
    sys.exit(main())

