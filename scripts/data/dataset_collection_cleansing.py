#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセット収集・クレンジングパイプライン
Dataset Collection and Cleansing Pipeline

NSFWデータ含むマルチモーダル日英データ、数学・科学データ収集
四値分類と統計処理によるデータクレンジング
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from tqdm import tqdm
import re
import hashlib
from datetime import datetime
import requests
from urllib.parse import urlparse
import warnings
warnings.filterwarnings("ignore")

# ライブラリインポート（オプション）
try:
    import datasets
    from datasets import load_dataset, Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from PIL import Image
    import cv2
    HAS_CV = True
except ImportError:
    HAS_CV = False

logger = logging.getLogger(__name__)

class DataQualityClassifier:
    """
    四値分類データ品質分類器
    4-class data quality classifier
    """

    def __init__(self):
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'acceptable': 0.5,
            'poor': 0.0
        }

        # 品質評価基準
        self.criteria = {
            'text_length': {'min': 10, 'max': 2048},
            'diversity': {'min': 0.3, 'max': 1.0},
            'coherence': {'min': 0.4, 'max': 1.0},
            'relevance': {'min': 0.5, 'max': 1.0},
            'toxicity': {'max': 0.3},  # 毒性スコアの上限
            'nsfw_content': {'max': 0.7}  # NSFWコンテンツ許容度
        }

    def classify_quality(self, sample: Dict[str, Any]) -> str:
        """
        サンプル品質を四値分類
        Classify sample quality into 4 classes
        """
        scores = self._compute_quality_scores(sample)
        overall_score = np.mean(list(scores.values()))

        if overall_score >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif overall_score >= self.quality_thresholds['good']:
            return 'good'
        elif overall_score >= self.quality_thresholds['acceptable']:
            return 'acceptable'
        else:
            return 'poor'

    def _compute_quality_scores(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """品質スコア計算"""
        scores = {}

        # テキスト長スコア
        if 'text' in sample:
            text_len = len(sample['text'].split())
            scores['text_length'] = self._normalize_score(
                text_len,
                self.criteria['text_length']['min'],
                self.criteria['text_length']['max']
            )

        # 多様性スコア（ユニーク単語率）
        if 'text' in sample:
            words = sample['text'].lower().split()
            unique_ratio = len(set(words)) / len(words) if words else 0
            scores['diversity'] = min(1.0, unique_ratio / 0.5)  # 50%ユニークを基準

        # 一貫性スコア（簡易版：文の論理的一貫性）
        if 'text' in sample:
            coherence = self._compute_coherence(sample['text'])
            scores['coherence'] = coherence

        # 関連性スコア
        if 'text' in sample and 'domain' in sample:
            relevance = self._compute_relevance(sample['text'], sample['domain'])
            scores['relevance'] = relevance

        # 毒性スコア（簡易版：禁止単語チェック）
        if 'text' in sample:
            toxicity = self._compute_toxicity(sample['text'])
            scores['toxicity'] = 1.0 - toxicity  # 低毒性が良い

        # NSFWコンテンツスコア
        if 'text' in sample or 'image_path' in sample:
            nsfw_score = self._compute_nsfw_score(sample)
            scores['nsfw_content'] = 1.0 - nsfw_score  # 低NSFWが良い（学習目的）

        return scores

    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """スコア正規化"""
        if value <= min_val:
            return 0.0
        elif value >= max_val:
            return 1.0
        else:
            return (value - min_val) / (max_val - min_val)

    def _compute_coherence(self, text: str) -> float:
        """一貫性スコア計算（簡易版）"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5

        # 文間類似度（簡易版：共通単語数）
        coherence_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())
            similarity = len(words1 & words2) / len(words1 | words2) if (words1 | words2) else 0
            coherence_scores.append(similarity)

        return np.mean(coherence_scores) if coherence_scores else 0.5

    def _compute_relevance(self, text: str, domain: str) -> float:
        """関連性スコア計算"""
        domain_keywords = {
            'mathematics': ['math', 'equation', 'theorem', 'proof', 'algebra', 'calculus', 'geometry'],
            'science': ['experiment', 'hypothesis', 'theory', 'data', 'analysis', 'research'],
            'programming': ['code', 'function', 'variable', 'algorithm', 'debug', 'compile'],
            'general': ['understand', 'explain', 'how', 'what', 'why']
        }

        keywords = domain_keywords.get(domain.lower(), domain_keywords['general'])
        text_lower = text.lower()

        matches = sum(1 for keyword in keywords if keyword in text_lower)
        return min(1.0, matches / len(keywords))

    def _compute_toxicity(self, text: str) -> float:
        """毒性スコア計算（簡易版）"""
        toxic_words = [
            'hate', 'stupid', 'idiot', 'dumb', 'worthless', 'terrible',
            'awful', 'horrible', 'disgusting', 'repulsive'
        ]

        text_lower = text.lower()
        toxic_count = sum(1 for word in toxic_words if word in text_lower)

        return min(1.0, toxic_count / 10)  # 最大10個の毒性単語で1.0

    def _compute_nsfw_score(self, sample: Dict[str, Any]) -> float:
        """NSFWスコア計算（検出目的のみ）"""
        nsfw_indicators = [
            'nsfw', 'adult', 'porn', 'sex', 'nude', 'erotic',
            '18+', 'xxx', 'mature', 'explicit'
        ]

        score = 0.0

        # テキストチェック
        if 'text' in sample:
            text_lower = sample['text'].lower()
            matches = sum(1 for indicator in nsfw_indicators if indicator in text_lower)
            score += min(1.0, matches / 3)

        # メタデータチェック
        if 'tags' in sample:
            tags_lower = [tag.lower() for tag in sample['tags']]
            matches = sum(1 for tag in tags_lower for indicator in nsfw_indicators if indicator in tag)
            score += min(1.0, matches / 2)

        # ファイル名チェック
        if 'image_path' in sample and sample['image_path']:
            filename = os.path.basename(sample['image_path']).lower()
            matches = sum(1 for indicator in nsfw_indicators if indicator in filename)
            score += min(1.0, matches / 2)

        return min(1.0, score)


class DatasetCollectionCleansing:
    """
    データセット収集・クレンジングシステム
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_classifier = DataQualityClassifier()

        # 出力ディレクトリ
        self.output_dir = Path(config.get('output_dir', 'D:/webdataset/datasets'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 収集設定
        self.max_samples_per_source = config.get('max_samples_per_source', 10000)
        self.license_filter = config.get('license_filter', ['mit', 'apache-2.0'])
        self.include_nsfw = config.get('include_nsfw', True)  # 検出目的のみ

        # クレンジング設定
        self.quality_thresholds = config.get('quality_thresholds', {
            'excellent': 0.9,
            'good': 0.7,
            'acceptable': 0.5
        })

        # 統計情報
        self.collection_stats = {
            'sources_processed': 0,
            'samples_collected': 0,
            'quality_distribution': {},
            'domain_distribution': {},
            'language_distribution': {}
        }

    def collect_and_cleansing_datasets(self) -> Dict[str, Any]:
        """
        データセット収集・クレンジング実行
        """
        logger.info("[DATASET] Starting dataset collection and cleansing...")

        # ターゲットデータセット
        target_datasets = self._get_target_datasets()

        all_samples = []
        quality_stats = {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0}

        for dataset_info in target_datasets:
            try:
                samples = self._collect_from_source(dataset_info)
                logger.info(f"[DATASET] Collected {len(samples)} samples from {dataset_info['name']}")

                # 品質分類とクレンジング
                cleansed_samples = []
                for sample in tqdm(samples, desc=f"Cleansing {dataset_info['name']}"):
                    quality_class = self.quality_classifier.classify_quality(sample)
                    quality_stats[quality_class] += 1

                    # 品質閾値によるフィルタリング
                    if self._should_keep_sample(sample, quality_class):
                        cleansed_samples.append({
                            **sample,
                            'quality_class': quality_class,
                            'source': dataset_info['name'],
                            'collected_at': datetime.now().isoformat()
                        })

                all_samples.extend(cleansed_samples)
                self.collection_stats['sources_processed'] += 1

            except Exception as e:
                logger.error(f"[DATASET] Failed to collect from {dataset_info['name']}: {e}")

        self.collection_stats['samples_collected'] = len(all_samples)
        self.collection_stats['quality_distribution'] = quality_stats

        # データセット保存
        result = self._save_cleansed_dataset(all_samples)

        logger.info(f"[DATASET] Collection completed: {len(all_samples)} samples")
        logger.info(f"[DATASET] Quality distribution: {quality_stats}")

        return result

    def _get_target_datasets(self) -> List[Dict[str, Any]]:
        """ターゲットデータセット定義"""
        datasets = []

        # HuggingFaceデータセット（MIT/Apache限定）
        hf_datasets = [
            {
                'name': 'elyza_tasks_100',
                'hf_path': 'elyza/ELYZA-tasks-100',
                'domain': 'multilingual_qa',
                'language': 'ja',
                'license': 'mit'
            },
            {
                'name': 'truthful_qa',
                'hf_path': 'truthful_qa',
                'domain': 'reasoning',
                'language': 'en',
                'license': 'apache-2.0'
            },
            {
                'name': 'math_qa',
                'hf_path': 'math_qa',
                'domain': 'mathematics',
                'language': 'en',
                'license': 'mit'
            },
            {
                'name': 'sciq',
                'hf_path': 'sciq',
                'domain': 'science',
                'language': 'en',
                'license': 'cc-by-4.0'  # 教育目的で使用
            },
            {
                'name': 'code_search_net',
                'hf_path': 'code_search_net',
                'domain': 'programming',
                'language': 'en',
                'license': 'mit'
            }
        ]

        # マルチモーダルデータセット（安全なもの）
        multimodal_datasets = [
            {
                'name': 'coco_captions',
                'hf_path': 'HuggingFaceM4/COCO',
                'domain': 'multimodal',
                'language': 'en',
                'license': 'cc-by-4.0',
                'has_images': True
            },
            {
                'name': 'flickr30k',
                'hf_path': 'nlphuji/flickr30k',
                'domain': 'multimodal',
                'language': 'en',
                'license': 'cc-by-4.0',
                'has_images': True
            }
        ]

        # NSFWデータセット（検出目的のみ）
        if self.include_nsfw:
            nsfw_datasets = [
                {
                    'name': 'nsfw_detection_corpus',
                    'hf_path': 'FredZhang7/nsfw-detector',
                    'domain': 'nsfw_detection',
                    'language': 'en',
                    'license': 'mit',
                    'purpose': 'detection_only'
                }
            ]
            datasets.extend(nsfw_datasets)

        datasets.extend(hf_datasets)
        datasets.extend(multimodal_datasets)

        return datasets

    def _collect_from_source(self, dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """データソースからの収集"""
        samples = []

        try:
            if HAS_DATASETS and 'hf_path' in dataset_info:
                # HuggingFaceデータセット
                dataset = load_dataset(dataset_info['hf_path'], split='train', streaming=True)

                for i, item in enumerate(dataset):
                    if i >= self.max_samples_per_source:
                        break

                    sample = self._convert_hf_sample(item, dataset_info)
                    if sample:
                        samples.append(sample)

            else:
                # ローカルデータまたはAPI
                logger.warning(f"No collection method for {dataset_info['name']}")

        except Exception as e:
            logger.error(f"Failed to collect from {dataset_info['name']}: {e}")

        return samples

    def _convert_hf_sample(self, item: Any, dataset_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """HuggingFaceサンプル変換"""
        try:
            sample = {
                'domain': dataset_info['domain'],
                'language': dataset_info['language'],
                'license': dataset_info.get('license', 'unknown')
            }

            # テキスト抽出
            if hasattr(item, 'text'):
                sample['text'] = item.text
            elif hasattr(item, 'question') and hasattr(item, 'answer'):
                sample['text'] = f"Q: {item.question}\nA: {item.answer}"
            elif hasattr(item, 'input') and hasattr(item, 'output'):
                sample['text'] = f"Input: {item.input}\nOutput: {item.output}"
            else:
                # 利用可能なフィールドを全て結合
                text_parts = []
                for field in ['instruction', 'input', 'query', 'context', 'response']:
                    if hasattr(item, field) and getattr(item, field):
                        text_parts.append(f"{field.title()}: {getattr(item, field)}")
                sample['text'] = '\n'.join(text_parts) if text_parts else None

            if not sample.get('text'):
                return None

            # 画像処理（マルチモーダル）
            if dataset_info.get('has_images', False):
                if hasattr(item, 'image') and item.image:
                    # 画像パスを保存（実際の画像は別途処理）
                    sample['image_path'] = f"images/{hashlib.md5(sample['text'].encode()).hexdigest()}.jpg"
                    sample['has_image'] = True
                else:
                    sample['has_image'] = False

            # NSFW関連メタデータ
            if dataset_info.get('purpose') == 'detection_only':
                sample['nsfw_purpose'] = 'detection'
                sample['nsfw_content'] = True

            return sample

        except Exception as e:
            logger.error(f"Failed to convert HF sample: {e}")
            return None

    def _should_keep_sample(self, sample: Dict[str, Any], quality_class: str) -> bool:
        """サンプル保持判定"""
        # 品質クラスによる基本フィルタ
        quality_scores = self.quality_classifier.quality_thresholds
        min_score = self.quality_thresholds.get('acceptable', 0.5)

        # 品質スコア計算
        scores = self.quality_classifier._compute_quality_scores(sample)
        overall_score = np.mean(list(scores.values()))

        if overall_score < min_score:
            return False

        # ライセンスチェック
        if sample.get('license', '').lower() not in [l.lower() for l in self.license_filter]:
            return False

        # NSFWコンテンツチェック（検出目的のみ許可）
        if sample.get('nsfw_content', False):
            if not self.include_nsfw:
                return False
            if sample.get('nsfw_purpose') != 'detection':
                return False

        # テキスト長チェック
        if 'text' in sample and len(sample['text']) < 10:
            return False

        return True

    def _save_cleansed_dataset(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """クレンジング済みデータセット保存"""
        # データフレーム変換
        df = pd.DataFrame(samples)

        # 統計情報追加
        quality_stats = df['quality_class'].value_counts().to_dict()
        domain_stats = df['domain'].value_counts().to_dict()
        language_stats = df['language'].value_counts().to_dict()

        # 保存
        output_path = self.output_dir / "cleansed_multimodal_dataset.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')

        # 統計情報
        stats_path = self.output_dir / "dataset_statistics.json"
        stats = {
            'total_samples': len(samples),
            'quality_distribution': quality_stats,
            'domain_distribution': domain_stats,
            'language_distribution': language_stats,
            'collection_stats': self.collection_stats,
            'generated_at': datetime.now().isoformat()
        }

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        result = {
            'dataset_path': str(output_path),
            'stats_path': str(stats_path),
            'statistics': stats
        }

        logger.info(f"[DATASET] Saved cleansed dataset to {output_path}")
        logger.info(f"[DATASET] Statistics: {stats}")

        return result

    def generate_quality_report(self) -> str:
        """品質レポート生成"""
        report = f"""
# Dataset Collection and Cleansing Report

## Overview
- Total samples collected: {self.collection_stats['samples_collected']}
- Sources processed: {self.collection_stats['sources_processed']}
- Quality distribution: {self.collection_stats['quality_distribution']}

## Quality Thresholds
- Excellent: >={self.quality_thresholds['excellent']}
- Good: >={self.quality_thresholds['good']}
- Acceptable: >={self.quality_thresholds['acceptable']}

## Data Domains
- Mathematics
- Science
- Programming
- Multilingual QA
- Multimodal (image-text pairs)
- NSFW Detection (for safety purposes only)

## Licenses
Filtered to include only: {', '.join(self.license_filter)}

## Notes
- NSFW content is included only for detection and safety assessment purposes
- All data is cleansed using 4-class quality classification
- Multimodal data includes both text and image components
- Statistics-based filtering removes poor quality samples
"""

        return report


def create_ppo_training_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    PPOトレーニングデータセット作成
    Create PPO training dataset based on model parameters
    """
    # Phi3.5パラメータ数（推定）
    phi35_params = 3.8e9  # 38億パラメータ

    # Borea-phi3.5-instinct-jpパラメータ数
    borea_params = 3.8e9  # 同じベース

    # 学習発散を防ぐためのデータセットサイズ計算
    # 経験則: パラメータ数の1/1000倍程度
    base_samples = int((phi35_params + borea_params) / 1000)

    # ドメインごとの割合
    domain_ratios = {
        'mathematics': 0.3,
        'science': 0.25,
        'programming': 0.2,
        'multimodal': 0.15,
        'reasoning': 0.1
    }

    dataset_config = {
        'total_samples': base_samples,
        'domain_ratios': domain_ratios,
        'include_nsfw': True,  # 検出目的のみ
        'quality_thresholds': {'acceptable': 0.6},  # PPO用に少し緩和
        'license_filter': ['mit', 'apache-2.0'],
        'output_dir': 'D:/webdataset/datasets/ppo_training'
    }

    # データセット収集・クレンジング
    collector = DatasetCollectionCleansing(dataset_config)
    result = collector.collect_and_cleansing_datasets()

    # PPO特化の追加処理
    ppo_samples = []

    for sample in result.get('samples', []):
        # PPO用のフォーマット変換
        ppo_sample = {
            'prompt': sample.get('text', ''),
            'domain': sample.get('domain', 'general'),
            'quality_score': sample.get('quality_score', 0.5),
            'language': sample.get('language', 'en'),
            'has_image': sample.get('has_image', False)
        }

        # 内部推論強化用のメタデータ追加
        ppo_sample['thinking_tokens'] = len(ppo_sample['prompt'].split()) // 10  # 推定
        ppo_sample['complexity_score'] = len(set(ppo_sample['prompt'].lower().split())) / len(ppo_sample['prompt'].split())

        ppo_samples.append(ppo_sample)

    # PPOデータセット保存
    ppo_output_path = Path(dataset_config['output_dir']) / "ppo_training_dataset.jsonl"
    ppo_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ppo_output_path, 'w', encoding='utf-8') as f:
        for sample in ppo_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

    result['ppo_dataset_path'] = str(ppo_output_path)
    result['ppo_samples'] = len(ppo_samples)

    logger.info(f"[PPO] Created PPO training dataset with {len(ppo_samples)} samples")

    return result


if __name__ == '__main__':
    # テスト実行
    config = {
        'output_dir': 'D:/webdataset/datasets/test_cleansing',
        'max_samples_per_source': 1000,
        'license_filter': ['mit', 'apache-2.0'],
        'include_nsfw': True,
        'quality_thresholds': {
            'excellent': 0.9,
            'good': 0.7,
            'acceptable': 0.5
        }
    }

    # データセット収集・クレンジング実行
    collector = DatasetCollectionCleansing(config)
    result = collector.collect_and_cleansing_datasets()

    # 品質レポート生成
    report = collector.generate_quality_report()
    with open('dataset_quality_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("Dataset collection and cleansing completed!")
    print(f"Results: {result}")
