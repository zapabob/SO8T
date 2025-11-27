#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四値分類ラベル付けスクリプト

ALLOW/ESCALATION/DENY/REFUSEの自動ラベル付けを実行

Usage:
    python scripts/label_four_class_dataset.py --input data/cleaned --output data/labeled
    python scripts/label_four_class_dataset.py --huggingface --input D:/webdataset/datasets --output D:/webdataset/labeled --test-size 0.2 --val-size 0.1
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm
import random

from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FourClassLabeler:
    """四値分類ラベル付けクラス"""
    
    # キーワード定義
    ALLOW_KEYWORDS = [
        "説明します", "一般的", "基本的", "公開", "教育", "参考", "通常",
        "標準", "公式", "ドキュメント", "一般的な", "基本的な"
    ]
    
    ESCALATION_KEYWORDS = [
        "確認", "照会", "上位", "専門", "判断", "エスカレーション", "検討",
        "協議", "承認", "許可", "要確認", "要検討"
    ]
    
    DENY_KEYWORDS = [
        "できません", "禁止", "回答を控え", "開示できません", "危険", "機密",
        "秘密", "非公開", "制限", "拒否", "不可", "不適切"
    ]
    
    REFUSE_KEYWORDS = [
        "拒否", "断る", "お断り", "対応不可", "応答不可", "回答拒否",
        "拒否します", "お断りします", "対応できません"
    ]
    
    def __init__(self, balance_classes: bool = True):
        """
        Args:
            balance_classes: クラスバランスを調整するか
        """
        self.balance_classes = balance_classes
        logger.info("Four Class Labeler initialized")
    
    def classify_text(self, text: str) -> str:
        """テキストを四値分類"""
        text_lower = text.lower()
        
        # DENY判定（最優先）
        if any(kw in text for kw in self.DENY_KEYWORDS):
            return "DENY"
        
        # REFUSE判定
        if any(kw in text for kw in self.REFUSE_KEYWORDS):
            return "REFUSE"
        
        # ESCALATION判定
        if any(kw in text for kw in self.ESCALATION_KEYWORDS):
            return "ESCALATION"
        
        # ALLOW判定
        if any(kw in text for kw in self.ALLOW_KEYWORDS):
            return "ALLOW"
        
        # デフォルト: 応答長で判定
        if len(text) < 50:
            return "DENY"  # 短すぎる応答は拒否とみなす
        else:
            return "ALLOW"  # デフォルトは許可
    
    def balance_dataset(self, samples: List[Dict]) -> List[Dict]:
        """データセットのクラスバランスを調整"""
        if not self.balance_classes:
            return samples
        
        # クラス別に分類
        class_samples = {"ALLOW": [], "ESCALATION": [], "DENY": [], "REFUSE": []}
        for sample in samples:
            label = sample.get("label", "ALLOW")
            if label in class_samples:
                class_samples[label].append(sample)
        
        # 各クラスのサンプル数
        class_counts = {k: len(v) for k, v in class_samples.items()}
        logger.info(f"Class distribution before balancing: {class_counts}")
        
        # 最小サンプル数に合わせる
        min_count = min(class_counts.values()) if class_counts.values() else 0
        if min_count == 0:
            logger.warning("Some classes have no samples, skipping balancing")
            return samples
        
        # 各クラスから最小サンプル数をランダムサンプリング
        balanced_samples = []
        for label, samples_list in class_samples.items():
            if len(samples_list) > min_count:
                balanced_samples.extend(random.sample(samples_list, min_count))
            else:
                balanced_samples.extend(samples_list)
        
        # シャッフル
        random.shuffle(balanced_samples)
        
        class_counts_after = Counter(s["label"] for s in balanced_samples)
        logger.info(f"Class distribution after balancing: {dict(class_counts_after)}")
        
        return balanced_samples
    
    def label_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        huggingface_mode: bool = False,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict[str, int]:
        """データセットにラベル付け"""
        logger.info("="*80)
        logger.info("Four Class Labeling")
        logger.info("="*80)

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if huggingface_mode:
            return self._label_huggingface_datasets(input_dir, output_dir, test_size, val_size)
        else:
            return self._label_jsonl_files(input_dir, output_dir, test_size, val_size)

    def _label_jsonl_files(self, input_dir: Path, output_dir: Path, test_size: float, val_size: float) -> Dict[str, int]:
        """JSONLファイルからラベル付け"""
        # 入力ファイル検索
        input_files = list(input_dir.glob("*.jsonl"))
        if not input_files:
            logger.error(f"No JSONL files found in {input_dir}")
            return {}

        logger.info(f"Found {len(input_files)} input files")
        return self._process_files(input_files, output_dir, "JSONL", test_size, val_size)

    def _label_huggingface_datasets(self, input_dir: Path, output_dir: Path, test_size: float, val_size: float) -> Dict[str, int]:
        """HuggingFaceデータセットからラベル付け"""
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return {}

        # データセットディレクトリを検索
        dataset_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not dataset_dirs:
            logger.error(f"No dataset directories found in {input_dir}")
            return {}

        logger.info(f"Found {len(dataset_dirs)} dataset directories")

        # 各データセットからファイルを収集
        all_files = []
        for dataset_dir in dataset_dirs:
            # JSON/JSONLファイルを検索
            json_files = list(dataset_dir.glob("*.json")) + list(dataset_dir.glob("*.jsonl"))
            if json_files:
                all_files.extend(json_files)
                logger.info(f"  {dataset_dir.name}: {len(json_files)} files")

        if not all_files:
            logger.error("No JSON/JSONL files found in any dataset directory")
            return {}

        logger.info(f"Total files to process: {len(all_files)}")
        return self._process_files(all_files, output_dir, "HuggingFace", test_size, val_size)

    def _process_files(self, input_files: List[Path], output_dir: Path, source_type: str,
                      test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> Dict[str, int]:
        """共通のファイル処理ロジック"""
        # 統計
        stats = {
            "total": 0,
            "ALLOW": 0,
            "ESCALATION": 0,
            "DENY": 0,
            "REFUSE": 0
        }

        labeled_samples: List[Dict] = []

        # 全ファイルを処理
        for input_file in input_files:
            logger.info(f"Processing {input_file.name}...")

            try:
                if input_file.suffix == ".json":
                    # JSONファイルの場合
                    with open(input_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        for sample in tqdm(data, desc=f"Labeling {input_file.name}"):
                            self._process_sample(sample, stats, labeled_samples)
                    elif isinstance(data, dict):
                        self._process_sample(data, stats, labeled_samples)

                elif input_file.suffix == ".jsonl":
                    # JSONLファイルの場合
                    with open(input_file, 'r', encoding='utf-8') as f:
                        for line in tqdm(f, desc=f"Labeling {input_file.name}"):
                            try:
                                sample = json.loads(line.strip())
                                self._process_sample(sample, stats, labeled_samples)
                            except json.JSONDecodeError:
                                continue

            except Exception as e:
                logger.warning(f"Error processing file {input_file}: {e}")
                continue

        # クラスバランス調整
        if self.balance_classes:
            logger.info("Balancing classes...")
            labeled_samples = self.balance_dataset(labeled_samples)

        # データ分割 (train/val/test)
        total_split_size = test_size + val_size
        if total_split_size > 0 and total_split_size < 1.0:
            logger.info(f"Splitting dataset (test_size={test_size}, val_size={val_size})...")

            # stratify用のラベルを取得
            labels = [s["label"] for s in labeled_samples]

            if val_size > 0:
                # train/val/testに3分割
                train_samples, temp_samples = train_test_split(
                    labeled_samples,
                    test_size=test_size + val_size,
                    stratify=labels,
                    random_state=random_state
                )

                # valとtestに分割
                val_ratio = val_size / (test_size + val_size)
                val_samples, test_samples = train_test_split(
                    temp_samples,
                    test_size=1 - val_ratio,
                    stratify=[s["label"] for s in temp_samples],
                    random_state=random_state
                )
            else:
                # train/testに2分割
                train_samples, test_samples = train_test_split(
                    labeled_samples,
                    test_size=test_size,
                    stratify=labels,
                    random_state=random_state
                )
                val_samples = []

            logger.info(f"Split results: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

            # 分割データを保存
            self._save_split_data(train_samples, output_dir, f"train_{source_type.lower()}.jsonl")
            if val_samples:
                self._save_split_data(val_samples, output_dir, f"val_{source_type.lower()}.jsonl")
            if test_samples:
                self._save_split_data(test_samples, output_dir, f"test_{source_type.lower()}.jsonl")
        else:
            # 分割なしで保存
            output_file = output_dir / f"labeled_four_class_dataset_{source_type.lower()}.jsonl"
            logger.info(f"Saving labeled dataset to {output_file}...")

            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in labeled_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # 統計レポート
        logger.info("="*80)
        logger.info("Labeling Statistics")
        logger.info("="*80)
        logger.info(f"Total samples: {stats['total']:,}")
        logger.info(f"ALLOW: {stats['ALLOW']:,}")
        logger.info(f"ESCALATION: {stats['ESCALATION']:,}")
        logger.info(f"DENY: {stats['DENY']:,}")
        logger.info(f"REFUSE: {stats['REFUSE']:,}")

        if test_size > 0 or val_size > 0:
            logger.info(f"Data split: test_size={test_size}, val_size={val_size}")
            logger.info("Output files:")
            logger.info(f"  - train_{source_type.lower()}.jsonl")
            if val_size > 0:
                logger.info(f"  - val_{source_type.lower()}.jsonl")
            if test_size > 0:
                logger.info(f"  - test_{source_type.lower()}.jsonl")
        else:
            logger.info(f"Output file: labeled_four_class_dataset_{source_type.lower()}.jsonl")

        logger.info("="*80)

        return stats

    def _save_split_data(self, samples: List[Dict], output_dir: Path, filename: str):
        """分割データを保存"""
        output_file = output_dir / filename
        logger.info(f"Saving {len(samples)} samples to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    def _process_sample(self, sample: Dict, stats: Dict[str, int], labeled_samples: List[Dict]):
        """個別のサンプルを処理"""
        stats["total"] += 1

        # テキスト抽出（複数のフィールドから）
        text = ""
        text_fields = ["text", "content", "instruction", "input", "output", "response", "prompt"]
        for field in text_fields:
            if field in sample and isinstance(sample[field], str):
                text += " " + sample[field]

        text = text.strip()
        if not text:
            return

        # ラベル付け
        label = self.classify_text(text)
        stats[label] += 1

        # ラベル付きサンプル
        labeled_sample = {
            **sample,
            "label": label,
            "original_text": text[:500]  # デバッグ用にテキストの一部を保存
        }
        labeled_samples.append(labeled_sample)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Label dataset with four classes (ALLOW/ESCALATION/DENY/REFUSE)"
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
        help="Output directory for labeled dataset"
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable class balancing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--huggingface",
        action="store_true",
        help="Process HuggingFace datasets instead of JSONL files"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation set size (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # シード設定
    random.seed(args.seed)
    
    # ラベル付け実行
    labeler = FourClassLabeler(balance_classes=not args.no_balance)
    
    try:
        stats = labeler.label_dataset(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            huggingface_mode=args.huggingface,
            test_size=args.test_size,
            val_size=args.val_size
        )
        
        logger.info("[SUCCESS] Dataset labeling completed")
        return 0
        
    except Exception as e:
        logger.error(f"[FAILED] Dataset labeling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

