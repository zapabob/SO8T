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
import re
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

    # キーワード定義 - 日本語と英語両対応
    ALLOW_KEYWORDS = [
        # 日本語
        "説明します", "一般", "基本", "公開", "教育", "参考", "通常",
        "標準", "公式", "ドキュメント", "一般的な", "基本的な", "教えます",
        "紹介します", "概要", "基礎", "入門", "使い方", "方法", "手順",
        "例", "サンプル", "チュートリアル", "ガイド", "ヘルプ", "サポート",
        "情報", "データ", "統計", "分析", "結果", "レポート", "まとめ",
        # 英語
        "explain", "general", "basic", "public", "education", "reference",
        "normal", "standard", "official", "documentation", "common",
        "fundamental", "teach", "introduce", "overview", "foundation",
        "beginner", "usage", "method", "procedure", "example", "sample",
        "tutorial", "guide", "help", "support", "information", "data",
        "statistics", "analysis", "result", "report", "summary"
    ]

    ESCALATION_KEYWORDS = [
        # 日本語
        "確認", "調査", "上司", "専門", "判断", "エスカレーション", "検討",
        "協議", "承認", "許可", "要確認", "要検討", "要相談", "要承認",
        "専門家", "管理者", "スーパーバイザー", "責任者", "上司", "マネージャー",
        "レビュー", "審査", "評価", "査読", "検証", "認証", "許可申請",
        # 英語
        "confirm", "inquiry", "superior", "expert", "judgment", "escalation",
        "review", "discussion", "approval", "permission", "verification",
        "consultation", "specialist", "administrator", "supervisor",
        "manager", "boss", "review", "audit", "assessment", "validation",
        "authentication", "authorization"
    ]

    DENY_KEYWORDS = [
        # 日本語
        "できません", "禁止", "回答を控え", "開示できません", "危険", "機密",
        "秘密", "非公開", "制限", "拒否", "不可", "不適切", "禁止事項",
        "制限事項", "アクセス不可", "利用不可", "実行不可", "許可なし",
        "セキュリティ", "プライバシー", "個人情報", "機密情報", "内部情報",
        "非公開情報", "制限付き", "機密扱い", "危険行為", "有害", "有害性",
        # 英語
        "cannot", "prohibited", "decline", "unable", "dangerous", "confidential",
        "secret", "private", "restricted", "denied", "inappropriate",
        "forbidden", "not allowed", "access denied", "security", "privacy",
        "personal information", "classified", "internal", "sensitive",
        "hazardous", "harmful", "risky", "unsafe"
    ]

    REFUSE_KEYWORDS = [
        # 日本語
        "拒否", "断る", "お断り", "対応不可", "応答不可", "回答拒否",
        "拒否します", "お断りします", "対応できません", "お断りいたします",
        "ご遠慮ください", "お引き受けできません", "お手伝いできません",
        "協力できません", "参加できません", "関与できません",
        # 英語
        "refuse", "decline", "reject", "deny", "unable to respond",
        "cannot assist", "will not help", "refusing", "declining",
        "rejecting", "cannot participate", "cannot cooperate",
        "cannot engage", "will not accept"
    ]
    
    def __init__(self, balance_classes: bool = True):
        """
        Args:
            balance_classes: クラスバランスを調整するぁE        """
        self.balance_classes = balance_classes
        logger.info("Four Class Labeler initialized")
    
    def classify_text(self, text: str) -> str:
        """テキストを四値分類（改良版）"""
        text_lower = text.lower()
        text_length = len(text.strip())

        # スコアリング方式で分類
        scores = {
            "ALLOW": 0,
            "ESCALATION": 0,
            "DENY": 0,
            "REFUSE": 0
        }

        # キーワードマッチングでスコア加算
        for kw in self.ALLOW_KEYWORDS:
            if kw in text_lower:
                scores["ALLOW"] += 1

        for kw in self.ESCALATION_KEYWORDS:
            if kw in text_lower:
                scores["ESCALATION"] += 2  # エスカレーションは重み付け

        for kw in self.DENY_KEYWORDS:
            if kw in text_lower:
                scores["DENY"] += 3  # DENYは高重み付け

        for kw in self.REFUSE_KEYWORDS:
            if kw in text_lower:
                scores["REFUSE"] += 4  # REFUSEは最高重み付け

        # 追加のヒューリスティック判定
        # 1. テキスト長による判定
        if text_length < 20:
            scores["DENY"] += 1  # 短すぎるテキストはDENY寄り
        elif text_length > 1000:
            scores["ALLOW"] += 1  # 長いテキストはALLOW寄り

        # 2. 日本語比率による判定
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text))
        if japanese_chars > 0:
            japanese_ratio = japanese_chars / len(text)
            if japanese_ratio > 0.8:
                scores["ALLOW"] += 0.5  # 日本語コンテンツはALLOW寄り

        # 3. 質問形式の判定
        if any(q in text_lower for q in ["?", "？", "ですか", "でしょうか", "どう", "何", "なぜ", "いつ"]):
            scores["ESCALATION"] += 1  # 質問はESCALATION寄り

        # 4. 命令形の判定
        if any(cmd in text_lower for cmd in ["して", "ください", "お願い", "do", "please", "help"]):
            scores["ESCALATION"] += 0.5

        # 5. 否定的表現の判定
        if any(neg in text_lower for neg in ["ない", "できません", "だめ", "no", "cannot", "unable"]):
            scores["DENY"] += 1

        # 最高スコアのクラスを返す
        max_score = max(scores.values())
        if max_score > 0:
            # 同点の場合は優先順位: REFUSE > DENY > ESCALATION > ALLOW
            for label in ["REFUSE", "DENY", "ESCALATION", "ALLOW"]:
                if scores[label] == max_score:
                    return label

        # デフォルト判定（スコアが全て0の場合）
        if text_length < 50:
            return "DENY"  # 短すぎる応答は拒否
        elif "?" in text or "？" in text:
            return "ESCALATION"  # 質問はエスカレーション
        else:
            return "ALLOW"  # デフォルトは許可

    def balance_dataset(self, samples: List[Dict]) -> List[Dict]:
        """データセットのクラスバランスを調整"""
        if not getattr(self, "balance_classes", False):
            return samples
        
        # クラス別に振り分け
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
        """チE�EタセチE��にラベル付け"""
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
            
        # 入力ファイル検索
        input_files = list(input_dir.glob("*.jsonl"))
        json_files = list(input_dir.glob("*.json"))

        if not input_files and not json_files and input_dir.is_file():
            # 単一ファイルが指定された場吁E            if input_dir.suffix == '.jsonl':
                input_files = [input_dir]
            elif input_dir.suffix == '.json':
                json_files = [input_dir]

        if not input_files and not json_files:
            logger.error(f"No JSONL or JSON files found in {input_dir}")
            return {}

        all_files = input_files + json_files
        logger.info(f"Found {len(all_files)} input files ({len(input_files)} JSONL, {len(json_files)} JSON)")
        return self._process_files(all_files, output_dir, "JSON", test_size, val_size)

    def _label_huggingface_datasets(self, input_dir: Path, output_dir: Path, test_size: float, val_size: float) -> Dict[str, int]:
        """HuggingFaceチE�EタセチE��からラベル付け"""
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return {}

        # チE�EタセチE��チE��レクトリを検索
        dataset_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not dataset_dirs:
            logger.error(f"No dataset directories found in {input_dir}")
            return {}

        logger.info(f"Found {len(dataset_dirs)} dataset directories")

        # 吁E��ータセチE��からファイルを収雁E        all_files = []
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
        """共通�Eファイル処琁E��ジチE��"""
        # 統訁E        stats = {
            "total": 0,
            "ALLOW": 0,
            "ESCALATION": 0,
            "DENY": 0,
            "REFUSE": 0
        }

        labeled_samples: List[Dict] = []

        # 全ファイルを�E琁E        for input_file in input_files:
            logger.info(f"Processing {input_file.name}...")

            try:
                if input_file.suffix == ".json":
                    # JSONファイルの場吁E                    with open(input_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        for sample in tqdm(data, desc=f"Labeling {input_file.name}"):
                            self._process_sample(sample, stats, labeled_samples)
                    elif isinstance(data, dict):
                        # SO8TチE�EタセチE��形式�E場吁E                        if 'training_data' in data:
                            for sample in tqdm(data['training_data'], desc=f"Labeling {input_file.name}"):
                                self._process_sample(sample, stats, labeled_samples)
                        elif 'data' in data:
                            # 別のチE�EタセチE��形弁E                            for sample in tqdm(data['data'], desc=f"Labeling {input_file.name}"):
                                self._process_sample(sample, stats, labeled_samples)
                        else:
                            # 単一サンプルとして扱ぁE                            self._process_sample(data, stats, labeled_samples)

                elif input_file.suffix == ".jsonl":
                    # JSONLファイルの場吁E                    with open(input_file, 'r', encoding='utf-8') as f:
                        for line in tqdm(f, desc=f"Labeling {input_file.name}"):
                            try:
                                sample = json.loads(line.strip())
                                self._process_sample(sample, stats, labeled_samples)
                            except json.JSONDecodeError:
                                logger.warning(f"JSON decode error in file {input_file.name}, skipping line.")
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
        min_samples_for_split = 100  # 分割に必要な最小サンプル数

        if total_split_size > 0 and total_split_size < 1.0 and len(labeled_samples) >= min_samples_for_split:
            logger.info(f"Splitting dataset (test_size={test_size}, val_size={val_size})...")

            # stratify用のラベルを取得
            labels = [s["label"] for s in labeled_samples]

            # 各クラスの最小サンプル数をチェック
            from collections import Counter
            label_counts = Counter(labels)
            min_class_count = min(label_counts.values())

            if min_class_count >= 2:
                # stratifyを使用可能
                use_stratify = True
                stratify_param = labels
            else:
                # stratifyを使用できない場合
                use_stratify = False
                stratify_param = None
                logger.warning(f"Some classes have too few samples for stratification (min: {min_class_count}), disabling stratify")

            if val_size > 0:
                # train/val/testに3分割
                # まずtrainと(temp = val + test)に分割
                train_samples, temp_samples = train_test_split(
                    labeled_samples,
                    test_size=test_size + val_size,
                    stratify=stratify_param,
                    random_state=random_state
                )

                # 次にtempをvalとtestに分割
                temp_labels = [s["label"] for s in temp_samples] if use_stratify else None
                val_ratio = val_size / (test_size + val_size)
                val_samples, test_samples = train_test_split(
                    temp_samples,
                    test_size=1 - val_ratio,  # test_sizeは残りのうちの割合
                    stratify=temp_labels,
                    random_state=random_state
                )
            else:
                # train/testに2分割
                train_samples, test_samples = train_test_split(
                    labeled_samples,
                    test_size=test_size,
                    stratify=stratify_param,
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
            if len(labeled_samples) < min_samples_for_split:
                logger.info(f"Dataset too small for splitting ({len(labeled_samples)} < {min_samples_for_split}), skipping split")
            # 分割なしで保存
            output_file = output_dir / f"labeled_four_class_dataset_{source_type.lower()}.jsonl"
            logger.info(f"Saving labeled dataset to {output_file}...")

            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in labeled_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # 統計レポ�EチE        logger.info("="*80)
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
        """刁E��チE�Eタを保孁E""
        output_file = output_dir / filename
        logger.info(f"Saving {len(samples)} samples to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    def _load_json_file(self, file_path: Path) -> List[Dict]:
        """JSONファイルを読み込み、サンプルリストに変換"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # JSONファイルの構造に応じてチE�Eタを抽出
            if isinstance(data, list):
                # 直接リスト形弁E                samples = data
            elif isinstance(data, dict) and 'training_data' in data:
                # SO8TチE�EタセチE��形弁E                samples = data['training_data']
            elif isinstance(data, dict) and 'data' in data:
                # 別のチE�EタセチE��形弁E                samples = data['data']
            else:
                # そ�E他�E構造は単一サンプルとして扱ぁE                samples = [data]

            logger.info(f"Loaded {len(samples)} samples from JSON file: {file_path}")
            return samples

        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return []

    def _process_sample(self, sample: Dict, stats: Dict[str, int], labeled_samples: List[Dict]):
        """個別のサンプルを処理"""
        stats["total"] += 1

        # テキスト抽出 - 複数のフィールドから優先順位で
        text = ""
        text_fields = ["chosen", "rejected", "text", "content", "instruction", "input", "output", "response", "prompt"]
        for field in text_fields:
            if field in sample and isinstance(sample[field], str) and sample[field].strip():
                text = sample[field].strip()
                break  # 最初に見つかったフィールドを使用

        if not text:
            return

        # 既存のラベルがある場合はそれを使用
        existing_label = sample.get("four_class_label", "").upper()
        if existing_label in ["ALLOW", "ESCALATION", "DENY", "REFUSE"]:
            label = existing_label
        else:
            # 新規ラベル付け
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
    
    # シード設宁E    random.seed(args.seed)
    
    # ラベル付け実衁E    labeler = FourClassLabeler(balance_classes=not args.no_balance)
    
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
