#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセット作成モジュール

クレンジング済みデータからデータセットを生成します。

Usage:
    python scripts/pipelines/dataset_creator.py --input D:/webdataset/cleaned --output D:/webdataset/datasets
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dataset_creator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatasetCreator:
    """データセット作成クラス"""
    
    def __init__(self, output_dir: Path):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dataset(self, samples: List[Dict], dataset_name: str = "so8t_dataset") -> Path:
        """
        データセットを作成
        
        Args:
            samples: サンプルリスト
            dataset_name: データセット名
        
        Returns:
            dataset_path: データセットファイルパス
        """
        logger.info(f"[DATASET] Creating dataset '{dataset_name}' from {len(samples)} samples...")
        
        # データセットメタデータを収集
        metadata = self._collect_metadata(samples)
        
        # データセットファイルを作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_file = self.output_dir / f"{dataset_name}_{timestamp}.jsonl"
        
        # JSONL形式で保存
        with open(dataset_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[OK] Dataset saved to {dataset_file.name}")
        
        # データセット情報ファイルを作成
        info_file = self.output_dir / f"{dataset_name}_{timestamp}_info.json"
        dataset_info = {
            'dataset_name': dataset_name,
            'created_at': datetime.now().isoformat(),
            'total_samples': len(samples),
            'metadata': metadata
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[OK] Dataset info saved to {info_file.name}")
        
        return dataset_file
    
    def _collect_metadata(self, samples: List[Dict]) -> Dict:
        """
        メタデータを収集
        
        Args:
            samples: サンプルリスト
        
        Returns:
            metadata: メタデータ
        """
        logger.info("[DATASET] Collecting metadata...")
        
        # 統計情報を収集
        by_category = Counter(sample.get('category', 'unknown') for sample in samples)
        by_language = Counter(sample.get('language', 'unknown') for sample in samples)
        by_domain = Counter(sample.get('domain', 'unknown') for sample in samples)
        
        # 分類統計
        classifications = []
        for sample in samples:
            classification = sample.get('classification', {})
            if classification:
                classifications.append(classification.get('decision', 'unknown'))
        by_classification = Counter(classifications)
        
        # テキスト長統計
        text_lengths = [
            len(sample.get('cleaned_text', sample.get('text', '')))
            for sample in samples
        ]
        
        metadata = {
            'by_category': dict(by_category),
            'by_language': dict(by_language),
            'by_domain': dict(by_domain),
            'by_classification': dict(by_classification),
            'text_length_stats': {
                'min': min(text_lengths) if text_lengths else 0,
                'max': max(text_lengths) if text_lengths else 0,
                'avg': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
                'total': sum(text_lengths)
            }
        }
        
        logger.info(f"[OK] Metadata collected: {len(by_category)} categories, {len(by_language)} languages")
        
        return metadata
    
    def create_train_test_split(
        self,
        samples: List[Dict],
        train_ratio: float = 0.8,
        dataset_name: str = "so8t_dataset"
    ) -> Tuple[Path, Path]:
        """
        訓練/テスト分割データセットを作成
        
        Args:
            samples: サンプルリスト
            train_ratio: 訓練データ比率
            dataset_name: データセット名
        
        Returns:
            (train_path, test_path): 訓練/テストデータセットパス
        """
        logger.info(f"[DATASET] Creating train/test split (ratio: {train_ratio})...")
        
        # シャッフル
        import random
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        
        # 分割
        split_idx = int(len(shuffled_samples) * train_ratio)
        train_samples = shuffled_samples[:split_idx]
        test_samples = shuffled_samples[split_idx:]
        
        logger.info(f"[DATASET] Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")
        
        # データセット作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_file = self.output_dir / f"{dataset_name}_train_{timestamp}.jsonl"
        test_file = self.output_dir / f"{dataset_name}_test_{timestamp}.jsonl"
        
        # 訓練データセット保存
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # テストデータセット保存
        with open(test_file, 'w', encoding='utf-8') as f:
            for sample in test_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[OK] Train dataset saved to {train_file.name}")
        logger.info(f"[OK] Test dataset saved to {test_file.name}")
        
        return train_file, test_file


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Dataset Creator")
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input directory (cleaned data)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory (datasets)'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='so8t_dataset',
        help='Dataset name'
    )
    parser.add_argument(
        '--train-test-split',
        action='store_true',
        help='Create train/test split'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Train ratio for split (default: 0.8)'
    )
    
    args = parser.parse_args()
    
    # 入力データを読み込み
    logger.info(f"[MAIN] Loading cleaned data from {args.input}...")
    samples = []
    
    # JSONLファイルを読み込み
    for jsonl_file in Path(args.input).glob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"[OK] Loaded {len(samples)} samples")
    
    # データセット作成
    creator = DatasetCreator(output_dir=args.output)
    
    if args.train_test_split:
        train_file, test_file = creator.create_train_test_split(
            samples=samples,
            train_ratio=args.train_ratio,
            dataset_name=args.dataset_name
        )
        logger.info(f"[MAIN] Train/test split created: {train_file.name}, {test_file.name}")
    else:
        dataset_file = creator.create_dataset(
            samples=samples,
            dataset_name=args.dataset_name
        )
        logger.info(f"[MAIN] Dataset created: {dataset_file.name}")


if __name__ == "__main__":
    main()

