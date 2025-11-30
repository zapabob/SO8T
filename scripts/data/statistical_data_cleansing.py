#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統計的なデータクレンジングスクリプト
ノーベル賞・フィールズ賞級CoTデータセットの品質向上のための高度なフィルタリング
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import statistics
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# パス設定
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root / "utils"))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StatisticalDataCleanser:
    """統計的なデータクレンジング器"""

    def __init__(self, quality_threshold: float = 0.6, similarity_threshold: float = 0.85):
        self.quality_threshold = quality_threshold
        self.similarity_threshold = similarity_threshold

        # 統計追跡
        self.stats = {
            'original_samples': 0,
            'quality_filtered': 0,
            'deduplicated': 0,
            'final_samples': 0,
            'quality_scores': [],
            'similarity_scores': [],
            'domain_distribution': defaultdict(int),
            'inference_type_distribution': defaultdict(int)
        }

        # 重複検出用
        self.content_hashes: Set[str] = set()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    def load_dataset(self, input_file: str) -> List[Dict[str, Any]]:
        """データセットを読み込み"""
        logger.info(f"Loading dataset from {input_file}")
        samples = []

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line.strip()))

        self.stats['original_samples'] = len(samples)
        logger.info(f"Loaded {len(samples)} samples")
        return samples

    def calculate_quality_score(self, sample: Dict[str, Any]) -> float:
        """包括的な品質スコアを計算"""
        score = 0.5  # ベーススコア
        score_components = {}

        # 1. コンテンツの長さと密度
        content_parts = []
        for key in ['instruction', 'problem_statement', 'solution', 'output']:
            if key in sample and sample[key]:
                content_parts.append(str(sample[key]))

        content = ' '.join(content_parts)
        content_length = len(content)

        # 長さスコア（適切な長さのコンテンツを重視）
        if 100 <= content_length <= 10000:
            length_score = 1.0
        elif content_length < 100:
            length_score = content_length / 100 * 0.5
        else:
            length_score = max(0.1, 1.0 - (content_length - 10000) / 50000)

        score_components['length'] = length_score * 0.15

        # 2. 科学的・数学的コンテンツの品質
        scientific_indicators = [
            r'\\[a-zA-Z]+\{',  # LaTeX数式
            r'theorem|lemma|proof',  # 数学用語
            r'equation|formula|algorithm',  # 科学用語
            r'証明|定理|証明',  # 日本語数学用語
            r'experiment|measurement|data',  # 実験・測定
            r'仮説|理論|モデル',  # 日本語科学用語
        ]

        scientific_score = 0
        for pattern in scientific_indicators:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            scientific_score += min(matches * 0.1, 0.2)  # 最大0.2

        score_components['scientific'] = scientific_score

        # 3. 構造的完全性（Phi-3.5タグの有無と完全性）
        structural_score = 0
        output = str(sample.get('output', ''))

        if '<think>' in output and '<final>' in output:
            structural_score += 0.3

        # 四値分類タグのチェック
        quad_tags = ['<|observation|>', '<|deduction|>', '<|abduction|>', '<|integration|>']
        tag_count = sum(1 for tag in quad_tags if tag in output)
        structural_score += (tag_count / len(quad_tags)) * 0.2

        score_components['structural'] = structural_score

        # 4. 言語的品質
        language_score = 0

        # 基本的な文構造チェック
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])

        if 5 <= avg_sentence_length <= 50:
            language_score += 0.2

        # 専門用語の適切な使用
        technical_terms = ['therefore', 'thus', 'hence', 'consequently', 'accordingly']
        technical_count = sum(1 for term in technical_terms if term in content.lower())
        language_score += min(technical_count * 0.05, 0.1)

        score_components['language'] = language_score

        # 5. 一貫性と完全性
        consistency_score = 0

        # 問題と解決の対応
        if 'instruction' in sample and 'solution' in sample:
            consistency_score += 0.1

        # メタデータの完全性
        required_fields = ['inference_type', 'domain']
        metadata_completeness = sum(1 for field in required_fields if field in sample) / len(required_fields)
        consistency_score += metadata_completeness * 0.1

        score_components['consistency'] = consistency_score

        # 6. ドメイン固有の品質チェック
        domain = sample.get('domain') or sample.get('category', 'general')
        domain_score = self._calculate_domain_specific_score(content, domain)
        score_components['domain'] = domain_score

        # 総スコア計算（重み付き）
        weights = {
            'scientific': 0.3,
            'structural': 0.25,
            'length': 0.15,
            'language': 0.15,
            'consistency': 0.1,
            'domain': 0.05
        }

        final_score = sum(score_components[comp] * weights[comp] for comp in score_components)

        # 統計追跡
        sample['_quality_components'] = score_components
        sample['_quality_score'] = final_score

        return final_score

    def _calculate_domain_specific_score(self, content: str, domain: str) -> float:
        """ドメイン固有の品質スコアを計算"""
        score = 0.1  # ベーススコア

        domain_keywords = {
            'mathematics': ['theorem', 'proof', 'equation', 'integral', 'derivative', 'matrix'],
            'physics': ['force', 'energy', 'quantum', 'relativity', 'particle', 'field'],
            'chemistry': ['reaction', 'molecule', 'bond', 'compound', 'catalyst', 'equilibrium'],
            'biology': ['cell', 'dna', 'protein', 'evolution', 'species', 'ecosystem'],
            'computer_science': ['algorithm', 'data', 'computation', 'complexity', 'optimization']
        }

        if domain in domain_keywords:
            keywords = domain_keywords[domain]
            keyword_count = sum(1 for keyword in keywords if keyword in content.lower())
            score += min(keyword_count * 0.05, 0.2)

        return score

    def deduplicate_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """高度な重複除去を実行"""
        logger.info("Starting deduplication process...")

        # 1. ハッシュベースの重複除去
        unique_samples = []
        seen_hashes = set()

        for sample in samples:
            # コンテンツの正規化とハッシュ化
            content_parts = []
            for key in ['instruction', 'problem_statement', 'solution']:
                if key in sample:
                    # 正規化：空白の除去、小文字化、特殊文字の除去
                    content = re.sub(r'\s+', ' ', str(sample[key]).lower())
                    content = re.sub(r'[^\w\s]', '', content)
                    content_parts.append(content)

            content_hash = hashlib.md5(' '.join(content_parts).encode()).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_samples.append(sample)

        logger.info(f"Hash-based deduplication: {len(samples)} -> {len(unique_samples)}")

        # 2. 意味的類似度ベースの重複除去（オプション）
        if len(unique_samples) > 1000:  # 大規模データセットの場合のみ
            unique_samples = self._semantic_deduplication(unique_samples)

        self.stats['deduplicated'] = len(unique_samples)
        return unique_samples

    def _semantic_deduplication(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """意味的類似度ベースの重複除去"""
        logger.info("Performing semantic deduplication...")

        # コンテンツ抽出
        contents = []
        for sample in samples:
            content_parts = []
            for key in ['instruction', 'problem_statement']:
                if key in sample:
                    content_parts.append(str(sample[key]))
            contents.append(' '.join(content_parts))

        # TF-IDFベクトル化
        try:
            tfidf_matrix = self.vectorizer.fit_transform(contents)

            # 類似度行列の計算（メモリ効率のためバッチ処理）
            n_samples = len(samples)
            to_remove = set()

            batch_size = 1000
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_matrix = tfidf_matrix[i:end_idx]

                # バッチ内類似度計算
                similarity_matrix = cosine_similarity(batch_matrix)

                # 類似サンプル検出
                for j in range(len(similarity_matrix)):
                    for k in range(j + 1, len(similarity_matrix)):
                        if similarity_matrix[j, k] > self.similarity_threshold:
                            # 品質の低い方を除去
                            idx_j = i + j
                            idx_k = i + k

                            quality_j = samples[idx_j].get('_quality_score', 0)
                            quality_k = samples[idx_k].get('_quality_score', 0)

                            if quality_j >= quality_k:
                                to_remove.add(idx_k)
                            else:
                                to_remove.add(idx_j)

            # 重複除去
            deduplicated = [s for idx, s in enumerate(samples) if idx not in to_remove]
            logger.info(f"Semantic deduplication: {len(samples)} -> {len(deduplicated)}")

            return deduplicated

        except Exception as e:
            logger.warning(f"Semantic deduplication failed: {e}. Using hash-based only.")
            return samples

    def apply_statistical_filters(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """統計的なフィルタを適用"""
        logger.info("Applying statistical filters...")

        # 品質スコアの統計計算
        quality_scores = [s.get('_quality_score', 0) for s in samples if '_quality_score' in s]

        if quality_scores:
            mean_quality = statistics.mean(quality_scores)
            std_quality = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0

            # 動的な閾値設定
            dynamic_threshold = max(self.quality_threshold, mean_quality - 0.5 * std_quality)

            logger.info(f"Quality statistics: mean={mean_quality:.3f}, std={std_quality:.3f}, threshold={dynamic_threshold:.3f}")

            # フィルタリング
            filtered_samples = []
            for sample in samples:
                quality = sample.get('_quality_score', 0)
                if quality >= dynamic_threshold:
                    filtered_samples.append(sample)
                else:
                    logger.debug(f"Filtered out sample with quality {quality:.3f}")

            logger.info(f"Quality filtering: {len(samples)} -> {len(filtered_samples)}")
            samples = filtered_samples

        # 分類分布のバランスチェック
        inference_types = [s.get('inference_type', 'unknown') for s in samples]
        type_counts = Counter(inference_types)

        # 最小サンプル数の確保
        min_samples_per_type = max(10, len(samples) // 20)  # データセットの5% or 10

        balanced_samples = []
        for sample in samples:
            inf_type = sample.get('inference_type', 'unknown')
            if type_counts[inf_type] > min_samples_per_type:
                balanced_samples.append(sample)
                type_counts[inf_type] -= 1

        logger.info(f"Balance filtering: {len(samples)} -> {len(balanced_samples)}")

        return balanced_samples

    def cleanse_dataset(self, input_file: str, output_file: str) -> List[Dict[str, Any]]:
        """データクレンジングのメイン処理"""
        logger.info("Starting statistical data cleansing...")

        # 1. データセット読み込み
        samples = self.load_dataset(input_file)

        # 2. 品質スコア計算
        logger.info("Calculating quality scores...")
        for sample in tqdm(samples, desc="Quality scoring"):
            quality_score = self.calculate_quality_score(sample)
            self.stats['quality_scores'].append(quality_score)

        # 3. 重複除去
        samples = self.deduplicate_samples(samples)

        # 4. 統計的フィルタ適用
        samples = self.apply_statistical_filters(samples)

        # 5. 最終統計更新
        self.stats['final_samples'] = len(samples)
        self._update_final_statistics(samples)

        # 6. 結果保存
        self.save_cleansed_dataset(samples, output_file)

        logger.info("Data cleansing completed!")
        logger.info(f"Original: {self.stats['original_samples']} -> Final: {self.stats['final_samples']}")

        return samples

    def _update_final_statistics(self, samples: List[Dict[str, Any]]):
        """最終統計を更新"""
        for sample in samples:
            domain = sample.get('domain') or sample.get('category', 'unknown')
            self.stats['domain_distribution'][domain] += 1

            inf_type = sample.get('inference_type', 'unknown')
            self.stats['inference_type_distribution'][inf_type] += 1

    def save_cleansed_dataset(self, samples: List[Dict[str, Any]], output_file: str):
        """クレンジング済みデータセットを保存"""
        logger.info(f"Saving cleansed dataset to {output_file}")

        # 統計情報を含むメタデータ
        metadata = {
            'cleansing_stats': self.stats,
            'parameters': {
                'quality_threshold': self.quality_threshold,
                'similarity_threshold': self.similarity_threshold
            }
        }

        # メインのデータセット保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')

        # 統計情報の保存
        stats_file = output_file.replace('.jsonl', '_cleansing_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(samples)} cleansed samples and statistics")

def main():
    parser = argparse.ArgumentParser(description="統計的なデータクレンジング")
    parser.add_argument("--input_file", type=str, required=True,
                       help="入力データセットファイル")
    parser.add_argument("--output_file", type=str, required=True,
                       help="出力データセットファイル")
    parser.add_argument("--quality_threshold", type=float, default=0.6,
                       help="品質閾値")
    parser.add_argument("--similarity_threshold", type=float, default=0.85,
                       help="類似度閾値")

    args = parser.parse_args()

    # クレンジング実行
    cleanser = StatisticalDataCleanser(
        quality_threshold=args.quality_threshold,
        similarity_threshold=args.similarity_threshold
    )

    cleansed_samples = cleanser.cleanse_dataset(args.input_file, args.output_file)

    logger.info(f"Cleansing completed! Processed {len(cleansed_samples)} samples")

if __name__ == "__main__":
    main()
