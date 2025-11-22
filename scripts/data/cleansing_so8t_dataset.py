#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T/thinkingデータセットのクレンジングと四値分類スクリプト
- 重複除去
- thinking向きでないノイズデータ除去
- 四値分類（薬物、エロのNSFWデータ含む検知目的データ）
"""

import json
import hashlib
import re
from collections import defaultdict, Counter
from pathlib import Path
import os

class SO8TDatasetCleanser:
    """SO8T/thinkingデータセットのクレンジングクラス"""

    def __init__(self):
        # 四値分類のキーワード
        self.classification_keywords = {
            'drug_detection': [
                '麻薬', '薬物', '覚醒剤', '大麻', 'コカイン', 'ヘロイン', 'LSD', 'MDMA', 'エクスタシー',
                'drug', 'marijuana', 'cocaine', 'heroin', 'methamphetamine', 'amphetamine',
                'opioid', 'narcotic', 'controlled substance', 'illicit drug', 'designer drug'
            ],
            'nsfw_erotic': [
                'エロ', 'アダルト', 'ポルノ', '性的', 'ヌード', 'SM', 'フェティシ', 'Hentai',
                'erotic', 'porn', 'sexual', 'nude', 'adult content', 'xxx', 'nsfw',
                'BDSM', 'fetish', 'hentai', 'ecchi'
            ],
            'nsfw_violence': [
                '暴力', '殺人', '自殺', '虐待', 'テロ', '戦争', '犯罪', '脅迫',
                'violence', 'murder', 'suicide', 'abuse', 'terrorism', 'war', 'crime'
            ],
            'safety_detection': [
                '検知', '分類', '識別', '判定', '評価', 'チェック', '検証',
                'detection', 'classification', 'identification', 'judgment', 'evaluation'
            ]
        }

        # thinking向きでないノイズパターン
        self.noise_patterns = [
            r'^[\s\n]*$',  # 空行のみ
            r'^https?://[^\s]+$',  # URLのみ
            r'^[^\w\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]{10,}$',  # 記号のみ
            r'^(.)\1{20,}$',  # 同じ文字の繰り返し
            r'^\d{10,}$',  # 数字のみ
        ]

    def calculate_text_hash(self, text: str) -> str:
        """テキストのハッシュを計算"""
        return hashlib.md5(text.strip().encode('utf-8')).hexdigest()

    def is_noise_text(self, text: str) -> bool:
        """thinking向きでないノイズテキストかを判定（緩和版）"""
        if len(text.strip()) < 5:  # 5文字未満（緩和）
            return True

        # より緩やかなノイズ判定
        for pattern in self.noise_patterns:
            if re.search(pattern, text.strip()):
                return True

        return False

    def classify_sample(self, text: str) -> dict:
        """サンプルを四値分類"""
        text_lower = text.lower()
        classifications = {
            'drug_detection': False,
            'nsfw_erotic': False,
            'nsfw_violence': False,
            'safety_detection': False
        }

        for category, keywords in self.classification_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    classifications[category] = True
                    break

        return classifications

    def cleanse_dataset(self, input_file: str, output_file: str):
        """データセットのクレンジングを実行"""
        print(f"データセットクレンジング開始: {input_file}")

        seen_hashes = set()
        cleansed_samples = []
        classification_stats = defaultdict(int)
        removed_stats = defaultdict(int)

        total_samples = 0

        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    total_samples += 1
                    data = json.loads(line.strip())

                    text = data.get('text', '').strip()

                    # 1. ノイズテキスト除去
                    if self.is_noise_text(text):
                        removed_stats['noise_text'] += 1
                        continue

                    # 2. 重複除去（オフ）
                    # text_hash = self.calculate_text_hash(text)
                    # if text_hash in seen_hashes:
                    #     removed_stats['duplicate'] += 1
                    #     continue
                    # seen_hashes.add(text_hash)

                    # 3. 四値分類
                    classifications = self.classify_sample(text)

                    # 分類結果をメタデータに追加
                    if 'metadata' not in data:
                        data['metadata'] = {}
                    data['metadata']['classifications'] = classifications

                    # 分類統計を更新
                    for category, detected in classifications.items():
                        if detected:
                            classification_stats[category] += 1

                    cleansed_samples.append(data)

                    if total_samples % 10000 == 0:
                        print(f"処理済み: {total_samples} サンプル")

                except json.JSONDecodeError as e:
                    print(f"JSONエラー line {line_num}: {e}")
                    removed_stats['json_error'] += 1
                    continue

        # クレンジング済みデータを保存
        print(f"クレンジング済みデータを保存: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in cleansed_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')

        # 統計レポート
        self.generate_cleansing_report(
            total_samples,
            len(cleansed_samples),
            removed_stats,
            classification_stats,
            output_file.replace('.jsonl', '_cleansing_report.md')
        )

        return len(cleansed_samples)

    def generate_cleansing_report(self, total_samples, cleansed_samples, removed_stats, classification_stats, report_file):
        """クレンジングレポートを生成"""
        report = f"""# SO8T/thinkingデータセット クレンジングレポート

## 概要
- **処理前サンプル数**: {total_samples:,}
- **処理後サンプル数**: {cleansed_samples:,}
- **除去サンプル数**: {total_samples - cleansed_samples:,}
- **保持率**: {(cleansed_samples / total_samples * 100):.1f}%

## 除去統計
"""

        for reason, count in removed_stats.items():
            report += f"- **{reason}**: {count:,} サンプル ({count / total_samples * 100:.1f}%)\n"

        report += "\n## 四値分類統計\n"
        for category, count in classification_stats.items():
            percentage = count / cleansed_samples * 100 if cleansed_samples > 0 else 0
            report += f"- **{category}**: {count:,} サンプル ({percentage:.1f}%)\n"

        report += "\n## 分類詳細\n"
        report += "- **drug_detection**: 薬物関連コンテンツ（検知目的）\n"
        report += "- **nsfw_erotic**: エロティック/アダルトコンテンツ（検知目的）\n"
        report += "- **nsfw_violence**: 暴力/有害コンテンツ（検知目的）\n"
        report += "- **safety_detection**: 安全判定/分類関連コンテンツ\n"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"クレンジングレポート保存: {report_file}")

    def apply_phi35_labeling_and_weighting(self, input_file: str, output_file: str):
        """Phi3.5ラベル体系での重みづけを適用"""
        print(f"Phi3.5ラベル体系での重みづけ適用: {input_file}")

        # Phi3.5ラベル体系での重み定義
        phi35_weights = {
            'drug_detection': 2.0,     # 薬物検知: 高重み（安全性重要）
            'nsfw_erotic': 1.5,        # エロ検知: 中高重み
            'nsfw_violence': 1.8,      # 暴力検知: 高重み（安全性重要）
            'safety_detection': 2.5,   # 安全判定: 最高重み
            'general': 1.0,            # 一般: 基準重み
            'mathematical': 1.2,       # 数学: 中重み
            'logical': 1.3,            # 論理: 中高重み
            'geometric': 1.4,          # 幾何: 中高重み
            'scientific': 1.2          # 科学: 中重み
        }

        weighted_samples = []
        total_weight = 0

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())

                    # Phi3.5ラベル決定
                    phi35_label = self._determine_phi35_label(data)
                    weight = phi35_weights.get(phi35_label, 1.0)

                    # 重み情報を追加
                    if 'metadata' not in data:
                        data['metadata'] = {}
                    data['metadata']['phi35_label'] = phi35_label
                    data['metadata']['weight'] = weight
                    data['metadata']['is_weighted'] = True

                    weighted_samples.append(data)
                    total_weight += weight

                except json.JSONDecodeError:
                    continue

        # 重みづけ済みデータを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in weighted_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')

        # 重みづけレポート生成
        self.generate_phi35_weighting_report(weighted_samples, phi35_weights, output_file.replace('.jsonl', '_phi35_weighting_report.md'))

        return len(weighted_samples)

    def _determine_phi35_label(self, data: dict) -> str:
        """Phi3.5ラベルを決定"""
        classifications = data.get('metadata', {}).get('classifications', {})
        reasoning_type = data.get('reasoning_type', 'general')

        # 四値分類が優先
        if classifications.get('safety_detection'):
            return 'safety_detection'
        elif classifications.get('drug_detection'):
            return 'drug_detection'
        elif classifications.get('nsfw_violence'):
            return 'nsfw_violence'
        elif classifications.get('nsfw_erotic'):
            return 'nsfw_erotic'
        else:
            # reasoning_typeに基づく
            return reasoning_type

    def generate_phi35_weighting_report(self, samples, weights, report_file):
        """Phi3.5重みづけレポートを生成"""
        label_counts = {}
        total_weight = 0

        for sample in samples:
            label = sample['metadata']['phi35_label']
            weight = sample['metadata']['weight']
            label_counts[label] = label_counts.get(label, {'count': 0, 'weight': 0})
            label_counts[label]['count'] += 1
            label_counts[label]['weight'] += weight
            total_weight += weight

        report = f"""# Phi3.5ラベル体系 重みづけレポート

## 概要
- **総サンプル数**: {len(samples):,}
- **総重み**: {total_weight:.1f}
- **平均重み**: {total_weight / len(samples):.3f}

## 重み定義
"""

        for label, weight in weights.items():
            report += f"- **{label}**: {weight:.1f}\n"

        report += "\n## ラベル分布\n"
        for label, stats in label_counts.items():
            percentage = stats['count'] / len(samples) * 100
            avg_weight = stats['weight'] / stats['count']
            report += f"- **{label}**: {stats['count']:,} サンプル ({percentage:.1f}%), 平均重み: {avg_weight:.3f}\n"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Phi3.5重みづけレポート保存: {report_file}")

def main():
    """メイン関数"""
    input_file = 'data/so8t_thinking_large_train.jsonl'
    cleansed_file = 'data/so8t_thinking_cleansed_train.jsonl'
    weighted_file = 'data/so8t_thinking_phi35_weighted_train.jsonl'

    if not os.path.exists(input_file):
        print(f"入力ファイルが存在しません: {input_file}")
        return

    cleanser = SO8TDatasetCleanser()

    # 1. クレンジング実行
    print("=== Phase 1: データセットクレンジング ===")
    cleansed_count = cleanser.cleanse_dataset(input_file, cleansed_file)
    print(f"クレンジング完了: {cleansed_count} サンプル")

    # 2. Phi3.5ラベル体系での重みづけ適用
    print("\n=== Phase 2: Phi3.5ラベル体系重みづけ ===")
    weighted_count = cleanser.apply_phi35_labeling_and_weighting(cleansed_file, weighted_file)
    print(f"Phi3.5重みづけ完了: {weighted_count} サンプル")

    print("\n[OK] SO8T/thinkingデータセット処理完了!")
    print(f"最終データセット: {weighted_file}")
    print("Phi3.5ラベルで重みづけされたSO8T/thinkingモデル構築用データセットが準備できました。")

if __name__ == '__main__':
    main()
