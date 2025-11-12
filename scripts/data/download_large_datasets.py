#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大規模データセットダウンロードスクリプト

D:/webdatasetに大規模データセットをダウンロード
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# HuggingFace datasets
try:
    from datasets import load_dataset, DatasetDict
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("datasets library not available")

# リクエスト
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available")


# デフォルトデータセット設定
DEFAULT_DATASETS = {
    "wikipedia_ja": {
        "name": "wikipedia",
        "config": "20231101.ja",
        "split": "train",
        "description": "Wikipedia日本語版（2023年11月1日版）"
    },
    "cc100_ja": {
        "name": "cc100",
        "config": "ja",
        "split": "train",
        "description": "CC-100日本語コーパス"
    },
    "oscar_ja": {
        "name": "oscar",
        "config": "unshuffled_deduplicated_ja",
        "split": "train",
        "description": "OSCAR日本語コーパス"
    },
    "mc4_ja": {
        "name": "mc4",
        "config": "ja",
        "split": "train",
        "description": "mC4日本語コーパス"
    }
}


class LargeDatasetDownloader:
    """大規模データセットダウンローダー"""
    
    def __init__(self, base_dir: Path = None):
        """
        Args:
            base_dir: ベースディレクトリ（デフォルト: D:\webdataset）
        """
        if base_dir is None:
            # D:\webdatasetを使用
            self.base_dir = Path(r"D:\webdataset")
        else:
            self.base_dir = Path(base_dir)
        
        # データセット保存ディレクトリ
        self.datasets_dir = self.base_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[DOWNLOADER] Initialized")
        logger.info(f"  Base directory: {self.base_dir}")
        logger.info(f"  Datasets directory: {self.datasets_dir}")
    
    def download_huggingface_dataset(
        self,
        dataset_name: str,
        config_name: Optional[str] = None,
        split: str = "train",
        output_dir: Optional[Path] = None,
        streaming: bool = False,
        max_samples: Optional[int] = None
    ) -> Path:
        """
        HuggingFaceデータセットをダウンロード
        
        Args:
            dataset_name: データセット名
            config_name: コンフィグ名（オプション）
            split: スプリット名（デフォルト: train）
            output_dir: 出力ディレクトリ（デフォルト: datasets_dir/dataset_name）
            streaming: ストリーミングモード（メモリ効率的）
            max_samples: 最大サンプル数（オプション）
        
        Returns:
            出力ディレクトリのパス
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library not available. Install with: pip install datasets")
        
        logger.info("="*80)
        logger.info(f"Downloading HuggingFace Dataset: {dataset_name}")
        if config_name:
            logger.info(f"  Config: {config_name}")
        logger.info(f"  Split: {split}")
        logger.info("="*80)
        
        # 出力ディレクトリ設定
        if output_dir is None:
            if config_name:
                output_dir = self.datasets_dir / f"{dataset_name}_{config_name}"
            else:
                output_dir = self.datasets_dir / dataset_name
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # データセット読み込み
            if config_name:
                dataset = load_dataset(
                    dataset_name,
                    config_name,
                    split=split,
                    streaming=streaming
                )
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=streaming)
            
            # JSONL形式で保存
            output_file = output_dir / f"{split}.jsonl"
            
            logger.info(f"Saving to {output_file}...")
            
            if streaming:
                # ストリーミングモード
                with open(output_file, 'w', encoding='utf-8') as f:
                    count = 0
                    for sample in tqdm(dataset, desc="Downloading"):
                        if max_samples and count >= max_samples:
                            break
                        
                        # テキストフィールドを取得
                        text = sample.get('text', sample.get('content', ''))
                        if not text:
                            continue
                        
                        # JSONL形式で保存
                        sample_dict = {
                            'text': text,
                            'source': dataset_name,
                            'config': config_name,
                            'split': split
                        }
                        f.write(json.dumps(sample_dict, ensure_ascii=False) + '\n')
                        count += 1
                
                logger.info(f"[OK] Downloaded {count} samples (streaming mode)")
            else:
                # 通常モード
                samples = []
                for sample in tqdm(dataset, desc="Processing"):
                    if max_samples and len(samples) >= max_samples:
                        break
                    
                    text = sample.get('text', sample.get('content', ''))
                    if not text:
                        continue
                    
                    sample_dict = {
                        'text': text,
                        'source': dataset_name,
                        'config': config_name,
                        'split': split
                    }
                    samples.append(sample_dict)
                
                # 一括保存
                with open(output_file, 'w', encoding='utf-8') as f:
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
                logger.info(f"[OK] Downloaded {len(samples)} samples")
            
            # メタデータ保存
            metadata = {
                'dataset_name': dataset_name,
                'config_name': config_name,
                'split': split,
                'output_file': str(output_file),
                'samples': len(samples) if not streaming else 'streaming'
            }
            
            metadata_file = output_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"[OK] Dataset saved to {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to download dataset: {e}")
            raise
    
    def download_multiple_datasets(
        self,
        dataset_configs: List[Dict[str, any]],
        max_samples_per_dataset: Optional[int] = None
    ) -> Dict[str, Path]:
        """
        複数のデータセットをダウンロード
        
        Args:
            dataset_configs: データセット設定のリスト
            max_samples_per_dataset: データセットあたりの最大サンプル数
        
        Returns:
            データセット名とパスの辞書
        """
        results = {}
        
        for config in tqdm(dataset_configs, desc="Downloading datasets"):
            dataset_name = config['name']
            config_name = config.get('config')
            split = config.get('split', 'train')
            
            try:
                output_dir = self.download_huggingface_dataset(
                    dataset_name=dataset_name,
                    config_name=config_name,
                    split=split,
                    max_samples=max_samples_per_dataset,
                    streaming=True  # メモリ効率的
                )
                results[dataset_name] = output_dir
            except Exception as e:
                logger.error(f"[ERROR] Failed to download {dataset_name}: {e}")
                continue
        
        return results
    
    def list_available_datasets(self) -> List[str]:
        """利用可能なデータセットリストを返す"""
        return list(DEFAULT_DATASETS.keys())


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Download Large Datasets")
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path(r"D:\webdataset"),
        help='Base directory for datasets (default: D:\\webdataset)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(DEFAULT_DATASETS.keys()) + ['all'],
        help='Dataset to download'
    )
    parser.add_argument(
        '--custom-dataset',
        type=str,
        help='Custom HuggingFace dataset name'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Dataset config name'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split (default: train)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to download'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets'
    )
    
    args = parser.parse_args()
    
    downloader = LargeDatasetDownloader(base_dir=args.base_dir)
    
    if args.list:
        logger.info("Available datasets:")
        for name, config in DEFAULT_DATASETS.items():
            logger.info(f"  {name}: {config['description']}")
        return 0
    
    if args.dataset == 'all':
        # すべてのデフォルトデータセットをダウンロード
        configs = list(DEFAULT_DATASETS.values())
        results = downloader.download_multiple_datasets(
            configs,
            max_samples_per_dataset=args.max_samples
        )
        logger.info(f"[OK] Downloaded {len(results)} datasets")
        return 0
    
    if args.custom_dataset:
        # カスタムデータセットをダウンロード
        output_dir = downloader.download_huggingface_dataset(
            dataset_name=args.custom_dataset,
            config_name=args.config,
            split=args.split,
            max_samples=args.max_samples,
            streaming=True
        )
        logger.info(f"[OK] Custom dataset downloaded to {output_dir}")
        return 0
    
    if args.dataset and args.dataset in DEFAULT_DATASETS:
        # デフォルトデータセットをダウンロード
        config = DEFAULT_DATASETS[args.dataset]
        output_dir = downloader.download_huggingface_dataset(
            dataset_name=config['name'],
            config_name=config.get('config'),
            split=config.get('split', 'train'),
            max_samples=args.max_samples,
            streaming=True
        )
        logger.info(f"[OK] Dataset downloaded to {output_dir}")
        return 0
    
    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())

