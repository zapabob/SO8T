#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データクレンジングスクリプト
4値分類後の統計的有意なデータクレンジング
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings

logger = logging.getLogger(__name__)

class StatisticalDataCleanser:
    """統計的有意なデータクレンジング"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_level = config.get('confidence_level', 0.95)
        self.outlier_threshold = config.get('outlier_threshold', 0.05)

    def cleanse_dataset(self, data_path: str, output_path: str) -> Dict[str, Any]:
        """データセットクレンジング実行"""
        logger.info(f"データクレンジング開始: {data_path}")

        # データ読み込み
        data = self.load_data(data_path)
        original_count = len(data)

        # 統計的クレンジング適用
        cleansed_data = self.apply_statistical_cleansing(data)

        # 結果保存
        self.save_data(cleansed_data, output_path)

        # 統計レポート
        report = {
            'original_count': original_count,
            'cleansed_count': len(cleansed_data),
            'removed_count': original_count - len(cleansed_data),
            'removal_rate': (original_count - len(cleansed_data)) / original_count,
            'cleansing_methods': self.config.get('cleansing_methods', [])
        }

        logger.info(f"データクレンジング完了: {original_count} -> {len(cleansed_data)}")
        return report

    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """データ読み込み"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def save_data(self, data: List[Dict[str, Any]], output_path: str):
        """データ保存"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def apply_statistical_cleansing(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """統計的クレンジング適用"""
        cleansing_methods = self.config.get('cleansing_methods', [
            'outlier_detection',
            'distribution_check',
            'quality_filtering'
        ])

        cleansed_data = data.copy()

        for method in cleansing_methods:
            if method == 'outlier_detection':
                cleansed_data = self.detect_outliers(cleansed_data)
            elif method == 'distribution_check':
                cleansed_data = self.check_distribution(cleansed_data)
            elif method == 'quality_filtering':
                cleansed_data = self.quality_filtering(cleansed_data)

        return cleansed_data

    def detect_outliers(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """外れ値検出"""
        logger.info("外れ値検出実行")

        # スコア抽出
        scores = [item.get('score', 0) for item in data]

        if len(scores) < 10:
            logger.warning("データ数が少なすぎて外れ値検出をスキップ")
            return data

        # Isolation Forestによる外れ値検出
        scores_reshaped = np.array(scores).reshape(-1, 1)

        # 標準化
        scaler = StandardScaler()
        scores_scaled = scaler.fit_transform(scores_reshaped)

        # Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.outlier_threshold,
            random_state=42
        )

        outlier_labels = iso_forest.fit_predict(scores_scaled)

        # 外れ値を除去
        cleansed_data = [
            item for item, label in zip(data, outlier_labels)
            if label == 1  # 1: 正常値, -1: 外れ値
        ]

        removed_count = len(data) - len(cleansed_data)
        logger.info(f"外れ値除去: {removed_count}件")

        return cleansed_data

    def check_distribution(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分布チェック"""
        logger.info("分布チェック実行")

        # ドメインごとのスコア分布をチェック
        domain_scores = {}
        for item in data:
            domain = item.get('domain', 'unknown')
            score = item.get('score', 0)
            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(score)

        # 各ドメインの分布をチェック
        cleansed_data = []
        for item in data:
            domain = item.get('domain', 'unknown')
            score = item.get('score', 0)
            domain_data = domain_scores[domain]

            if len(domain_data) >= 5:
                # z-scoreによる異常値検出
                mean_score = np.mean(domain_data)
                std_score = np.std(domain_data)

                if std_score > 0:
                    z_score = abs(score - mean_score) / std_score
                    # z-scoreが3を超えるものは除去
                    if z_score <= 3.0:
                        cleansed_data.append(item)
                else:
                    cleansed_data.append(item)
            else:
                cleansed_data.append(item)

        removed_count = len(data) - len(cleansed_data)
        logger.info(f"分布チェック除去: {removed_count}件")

        return cleansed_data

    def quality_filtering(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """品質フィルタリング"""
        logger.info("品質フィルタリング実行")

        cleansed_data = []

        for item in data:
            # 品質チェック
            is_quality = True

            # 必須フィールドチェック
            required_fields = ['id', 'domain', 'score']
            for field in required_fields:
                if field not in item:
                    is_quality = False
                    break

            # スコア範囲チェック
            score = item.get('score', 0)
            if not (0.0 <= score <= 1.0):
                is_quality = False

            # SO(8)スコアチェック
            metadata = item.get('metadata', {})
            so8t_scores = metadata.get('so8t_scores', {})
            if not so8t_scores:
                is_quality = False

            # テキスト長チェック
            text = item.get('text', '')
            if len(text) < 10:
                is_quality = False

            if is_quality:
                cleansed_data.append(item)

        removed_count = len(data) - len(cleansed_data)
        logger.info(f"品質フィルタリング除去: {removed_count}件")

        return cleansed_data

class AdvancedDataCleanser:
    """高度データクレンジング"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def apply_advanced_cleansing(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """高度クレンジング適用"""
        logger.info("高度データクレンジング実行")

        # 統計的クレンジング
        statistical_cleanser = StatisticalDataCleanser(self.config)
        cleansed_data = statistical_cleanser.apply_statistical_cleansing(data)

        # クラスター分析による異常検出
        cleansed_data = self.cluster_based_outlier_detection(cleansed_data)

        # 重複除去（高度版）
        cleansed_data = self.advanced_deduplication(cleansed_data)

        # レポート作成
        report = {
            'original_count': len(data),
            'final_count': len(cleansed_data),
            'total_removed': len(data) - len(cleansed_data),
            'methods_applied': [
                'statistical_cleansing',
                'cluster_outlier_detection',
                'advanced_deduplication'
            ]
        }

        return cleansed_data, report

    def cluster_based_outlier_detection(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """クラスター分析による外れ値検出"""
        logger.info("クラスター分析外れ値検出")

        # 特徴量抽出
        features = []
        for item in data:
            score = item.get('score', 0)
            text_len = len(item.get('text', ''))
            domain_encoded = hash(item.get('domain', '')) % 1000 / 1000.0  # 簡易エンコード
            features.append([score, text_len, domain_encoded])

        if len(features) < 5:
            return data

        # DBSCANによるクラスタリング
        features_array = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)

        dbscan = DBSCAN(eps=0.5, min_samples=3)
        cluster_labels = dbscan.fit_predict(features_scaled)

        # ノイズ（-1）を除去
        cleansed_data = [
            item for item, label in zip(data, cluster_labels)
            if label != -1
        ]

        removed_count = len(data) - len(cleansed_data)
        logger.info(f"クラスター外れ値除去: {removed_count}件")

        return cleansed_data

    def advanced_deduplication(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """高度重複除去"""
        logger.info("高度重複除去実行")

        seen_texts = set()
        seen_ids = set()
        cleansed_data = []

        for item in data:
            # ID重複チェック
            item_id = item.get('id', '')
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            # テキスト類似度チェック（簡易版）
            text = item.get('text', '')
            text_hash = hash(text) % 10000  # 簡易ハッシュ

            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                cleansed_data.append(item)

        removed_count = len(data) - len(cleansed_data)
        logger.info(f"高度重複除去: {removed_count}件")

        return cleansed_data

def create_cleansing_config() -> Dict[str, Any]:
    """クレンジング設定作成"""
    return {
        'confidence_level': 0.95,
        'outlier_threshold': 0.05,
        'cleansing_methods': [
            'outlier_detection',
            'distribution_check',
            'quality_filtering'
        ],
        'advanced_methods': [
            'cluster_outlier_detection',
            'advanced_deduplication'
        ]
    }

def main():
    """メイン処理"""
    print("🧹 Data Cleansing Pipeline")
    print("=" * 50)

    config = create_cleansing_config()

    # クレンジング対象データセット
    datasets = [
        ('data/train_sft_enhanced.jsonl', 'data/train_sft_cleansed.jsonl'),
        ('data/train_ppo_integrated.jsonl', 'data/train_ppo_cleansed.jsonl'),
        ('data/test_eval.jsonl', 'data/test_eval_cleansed.jsonl')
    ]

    total_report = {}

    for input_path, output_path in datasets:
        if Path(input_path).exists():
            print(f"\n処理中: {input_path}")

            # 高度クレンジング実行
            cleanser = AdvancedDataCleanser(config)
            with open(input_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]

            cleansed_data, report = cleanser.apply_advanced_cleansing(data)

            # 保存
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in cleansed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            print(f"クレンジング完了: {len(data)} -> {len(cleansed_data)}")
            total_report[input_path] = report

    print("\n=== 全体クレンジング結果 ===")
    for path, report in total_report.items():
        print(f"{path}:")
        print(f"  元: {report['original_count']}件")
        print(f"  最終: {report['final_count']}件")
        print(".1%")
    print("=" * 50)

if __name__ == "__main__":
    main()

