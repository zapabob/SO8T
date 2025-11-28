#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8Tデータセット四値分類ラベリングスクリプト

統合データセットに四値分類（ALLOW/ESCALATION/DENY/REFUSE）を適用
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import random
from sklearn.model_selection import train_test_split


class SO8TFourClassLabeler:
    """SO8T四値分類ラベラー"""

    def __init__(self):
        self.four_class_labels = ['ALLOW', 'ESCALATION', 'DENY', 'REFUSE']
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _classify_sample(self, sample: Dict) -> str:
        """サンプルを四値分類"""
        text = sample.get('text', '').lower()

        # DENY: 危険・有害コンテンツ
        deny_keywords = [
            '殺す', '死ぬ', '自殺', '暴力', '犯罪', '違法', '麻薬', '薬物',
            'ハッキング', '不正アクセス', '個人情報', 'プライバシー侵害',
            '差別', 'ヘイト', '脅迫', '脅し', '詐欺', '騙す',
            'nsfw', 'アダルト', 'ポルノ', 'エロ', '性的', 'レイプ'
        ]
        if any(keyword in text for keyword in deny_keywords):
            return 'DENY'

        # ESCALATION: 専門家判断が必要な内容
        escalation_keywords = [
            '医療', '診断', '治療', '法律', '訴訟', '契約', '投資', '金融',
            'セキュリティ', '機密', '国家安全', '軍事', '政治', '国際関係'
        ]
        if any(keyword in text for keyword in escalation_keywords):
            return 'ESCALATION'

        # REFUSE: 明確に拒否すべき内容
        refuse_keywords = [
            'jailbreak', 'uncensored', 'unrestricted', 'unlimited',
            'bypass', 'override', 'ignore', 'disregard',
            'illegal', 'unethical', 'immoral'
        ]
        if any(keyword in text for keyword in refuse_keywords):
            return 'REFUSE'

        # ALLOW: デフォルト（安全な内容）
        return 'ALLOW'

    def label_dataset(self, input_file: Path, output_dir: Path) -> Dict[str, int]:
        """データセットにラベリングを適用"""
        self.logger.info(f"[LABELING] Processing {input_file}")

        # 出力ディレクトリ作成
        output_dir.mkdir(parents=True, exist_ok=True)

        # データ読み込み
        samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError:
                        continue

        self.logger.info(f"[LOAD] Loaded {len(samples)} samples")

        # 四値分類適用
        labeled_samples = []
        label_counts = Counter()

        for sample in samples:
            label = self._classify_sample(sample)
            labeled_sample = {
                **sample,
                'four_class_label': label
            }
            labeled_samples.append(labeled_sample)
            label_counts[label] += 1

        self.logger.info(f"[LABELS] Distribution: {dict(label_counts)}")

        # データ分割 (train/val/test)
        if len(labeled_samples) >= 1000:  # 分割に十分なサンプル数
            train_samples, temp_samples = train_test_split(
                labeled_samples, test_size=0.3, random_state=42, stratify=[s['four_class_label'] for s in labeled_samples]
            )
            val_samples, test_samples = train_test_split(
                temp_samples, test_size=0.333, random_state=42, stratify=[s['four_class_label'] for s in temp_samples]
            )

            # 保存
            splits = {
                'train': train_samples,
                'val': val_samples,
                'test': test_samples
            }

            for split_name, split_samples in splits.items():
                output_file = output_dir / f"so8t_labeled_{split_name}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for sample in split_samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                self.logger.info(f"[SAVE] {split_name}: {len(split_samples)} samples -> {output_file}")
        else:
            # サンプル数が少ない場合は単一ファイル
            output_file = output_dir / "so8t_labeled_full.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in labeled_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            self.logger.info(f"[SAVE] Full dataset: {len(labeled_samples)} samples -> {output_file}")

        return dict(label_counts)


def main():
    parser = argparse.ArgumentParser(description="SO8T dataset four-class labeling")
    parser.add_argument("--input", type=Path, required=True, help="Input dataset file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")

    args = parser.parse_args()

    labeler = SO8TFourClassLabeler()
    stats = labeler.label_dataset(args.input, args.output)

    print(f"Labeling completed: {stats}")


if __name__ == "__main__":
    main()
