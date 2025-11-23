#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセット分割スクリプト

統計的ランダム分割で訓練/検証/テストに分割

Usage:
    python scripts/split_dataset.py --input data/labeled --output data/splits
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetSplitter:
    """データセット分割クラス"""
    
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify: bool = True,
        seed: int = 42
    ):
        """
        Args:
            train_ratio: 訓練データ比率
            val_ratio: 検証データ比率
            test_ratio: テストデータ比率
            stratify: 層化サンプリングを使用するか
            seed: 乱数シード
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.stratify = stratify
        self.seed = seed
        
        logger.info("Dataset Splitter initialized")
        logger.info(f"  Train: {train_ratio*100:.1f}%")
        logger.info(f"  Val: {val_ratio*100:.1f}%")
        logger.info(f"  Test: {test_ratio*100:.1f}%")
        logger.info(f"  Stratify: {stratify}")
    
    def split_dataset(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict[str, int]:
        """データセット分割実行"""
        logger.info("="*80)
        logger.info("Dataset Splitting")
        logger.info("="*80)
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 入力ファイル検索
        input_files = list(input_dir.glob("*.jsonl"))
        if not input_files:
            logger.error(f"No JSONL files found in {input_dir}")
            return {}
        
        logger.info(f"Found {len(input_files)} input files")
        
        # 全サンプル読み込み
        all_samples: List[Dict] = []
        for input_file in input_files:
            logger.info(f"Loading {input_file.name}...")
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Loading {input_file.name}"):
                    try:
                        sample = json.loads(line.strip())
                        all_samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"Total samples: {len(all_samples):,}")
        
        # ラベル抽出（層化サンプリング用）
        labels = [s.get("label", "ALLOW") for s in all_samples]
        
        # 訓練/検証+テストに分割
        stratify_labels = labels if self.stratify else None
        train_samples, temp_samples = train_test_split(
            all_samples,
            test_size=(self.val_ratio + self.test_ratio),
            stratify=stratify_labels,
            random_state=self.seed
        )
        
        # 検証/テストに分割
        temp_labels = [s.get("label", "ALLOW") for s in temp_samples]
        stratify_temp = temp_labels if self.stratify else None
        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=(self.test_ratio / (self.val_ratio + self.test_ratio)),
            stratify=stratify_temp,
            random_state=self.seed
        )
        
        # 統計
        stats = {
            "total": len(all_samples),
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples)
        }
        
        # 出力ファイルに保存
        splits = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples
        }
        
        for split_name, samples in splits.items():
            output_file = output_dir / f"{split_name}.jsonl"
            logger.info(f"Saving {split_name} split to {output_file}...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in tqdm(samples, desc=f"Writing {split_name}"):
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # 統計レポート
        logger.info("="*80)
        logger.info("Split Statistics")
        logger.info("="*80)
        logger.info(f"Total: {stats['total']:,}")
        logger.info(f"Train: {stats['train']:,} ({stats['train']/stats['total']*100:.1f}%)")
        logger.info(f"Val: {stats['val']:,} ({stats['val']/stats['total']*100:.1f}%)")
        logger.info(f"Test: {stats['test']:,} ({stats['test']/stats['total']*100:.1f}%)")
        logger.info("="*80)
        
        return stats


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing JSONL files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for split datasets"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training data ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation data ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test data ratio (default: 0.1)"
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified sampling"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # 分割実行
    splitter = DatasetSplitter(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify=not args.no_stratify,
        seed=args.seed
    )
    
    try:
        stats = splitter.split_dataset(
            input_dir=Path(args.input),
            output_dir=Path(args.output)
        )
        
        logger.info("[SUCCESS] Dataset splitting completed")
        return 0
        
    except Exception as e:
        logger.error(f"[FAILED] Dataset splitting failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

