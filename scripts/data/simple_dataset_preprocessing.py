#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シンプルなデータセット前処理スクリプト
ライブラリ依存を最小限に抑えた基本的なデータセット収集・前処理
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import yaml
import argparse

def create_mock_datasets():
    """モックデータセットを生成"""
    print("Creating mock datasets for testing...")

    # 出力ディレクトリ作成
    output_base = Path("D:/webdataset/datasets")
    output_base.mkdir(parents=True, exist_ok=True)

    datasets = {
        'elyza_tasks_100': {
            'domain': 'multilingual_qa',
            'language': 'ja',
            'license': 'mit',
            'samples': []
        },
        'truthful_qa': {
            'domain': 'reasoning',
            'language': 'en',
            'license': 'apache-2.0',
            'samples': []
        },
        'math_qa': {
            'domain': 'mathematics',
            'language': 'en',
            'license': 'mit',
            'samples': []
        },
        'sciq': {
            'domain': 'science',
            'language': 'en',
            'license': 'cc-by-4.0',
            'samples': []
        },
        'code_search_net': {
            'domain': 'programming',
            'language': 'en',
            'license': 'mit',
            'samples': []
        }
    }

    # 各データセットにモックサンプルを生成
    for dataset_name, config in datasets.items():
        print(f"Generating mock samples for {dataset_name}...")

        num_samples = 100  # テスト用

        for i in range(num_samples):
            sample = {
                'text': f"Mock sample {i} from {dataset_name}: This is a {config['domain']} sample in {config['language']} language.",
                'domain': config['domain'],
                'language': config['language'],
                'license': config['license'],
                'quality_score': 0.7 + np.random.normal(0, 0.1),  # 品質スコア
                'has_image': False,
                'nsfw_content': False,
                'id': f"{dataset_name}_{i}"
            }
            config['samples'].append(sample)

        # データセットをJSONL形式で保存
        output_file = output_base / f"{dataset_name}_processed.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in config['samples']:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')

        print(f"Saved {len(config['samples'])} samples to {output_file}")

    return datasets

def create_soul_weights_dataset(output_dir: str = "D:/webdataset/datasets/soul_weights"):
    """魂の重みデータセットを生成"""
    print("Creating soul weights dataset...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_samples = 500

    # Alpha Gate値の生成（シグモイド関数ベース）
    alphas = []
    for i in range(num_samples):
        # シグモイド関数でアニーリングをシミュレート
        t = i / (num_samples - 1)
        sigmoid_value = 1.0 / (1.0 + np.exp(-5.0 * (t - 0.5)))
        alpha = sigmoid_value
        alphas.append(alpha)

    # SO(8)回転行列の生成（簡易版）
    so8_rotations = {
        'r_safe': torch.randn(num_samples, 64, 64).tolist(),  # メモリ節約のため小サイズ
        'r_cmd': torch.randn(num_samples, 64, 64).tolist(),
        'r_total': torch.randn(num_samples, 64, 64).tolist()
    }

    # 魂の柱データ生成
    soul_pillars = {
        'safety_head': torch.randn(num_samples, 768).tolist(),
        'task_head': torch.randn(num_samples, 768).tolist(),
        'dual_heads': torch.randn(num_samples, 2, 768).tolist(),
        'pet_regularization': torch.randn(num_samples, 768).tolist()
    }

    # LoRAアダプタ重み生成
    lora_adapter = {
        'lora_A': torch.randn(num_samples, 768, 32).tolist(),
        'lora_B': torch.randn(num_samples, 32, 768).tolist()
    }

    # 統合データセット作成
    soul_samples = []
    for i in range(num_samples):
        sample = {
            'alpha_gate': alphas[i],
            'so8_rotations': {
                'r_safe': so8_rotations['r_safe'][i],
                'r_cmd': so8_rotations['r_cmd'][i],
                'r_total': so8_rotations['r_total'][i]
            },
            'soul_pillars': {
                'safety_head': soul_pillars['safety_head'][i],
                'task_head': soul_pillars['task_head'][i],
                'dual_heads': soul_pillars['dual_heads'][i],
                'pet_regularization': soul_pillars['pet_regularization'][i]
            },
            'lora_adapter': {
                'lora_A': lora_adapter['lora_A'][i],
                'lora_B': lora_adapter['lora_B'][i]
            },
            'id': f"soul_weight_{i}"
        }
        soul_samples.append(sample)

    # 保存
    output_file = output_path / "soul_weights_dataset.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in soul_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

    print(f"Soul weights dataset saved: {output_file} ({len(soul_samples)} samples)")

    return str(output_file)

def perform_data_cleansing():
    """データクレンジングを実行"""
    print("Performing data cleansing...")

    # 四値分類基準
    quality_criteria = {
        'excellent': lambda score: score >= 0.9,
        'good': lambda score: 0.7 <= score < 0.9,
        'acceptable': lambda score: 0.5 <= score < 0.7,
        'poor': lambda score: score < 0.5
    }

    # 統計処理
    cleansing_stats = {
        'total_samples': 0,
        'quality_distribution': {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0},
        'language_distribution': {},
        'domain_distribution': {}
    }

    # 各データセットを処理
    datasets_dir = Path("D:/webdataset/datasets")
    if not datasets_dir.exists():
        print("No datasets directory found. Creating mock datasets first...")
        create_mock_datasets()

    for jsonl_file in datasets_dir.glob("*_processed.jsonl"):
        print(f"Processing {jsonl_file.name}...")

        samples = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        cleansed_samples = []
        for sample in samples:
            cleansing_stats['total_samples'] += 1

            # 品質分類
            score = sample.get('quality_score', 0.5)
            for quality, criterion in quality_criteria.items():
                if criterion(score):
                    sample['quality_class'] = quality
                    cleansing_stats['quality_distribution'][quality] += 1
                    break

            # 言語・ドメイン統計
            lang = sample.get('language', 'unknown')
            cleansing_stats['language_distribution'][lang] = cleansing_stats['language_distribution'].get(lang, 0) + 1

            domain = sample.get('domain', 'unknown')
            cleansing_stats['domain_distribution'][domain] = cleansing_stats['domain_distribution'].get(domain, 0) + 1

            cleansed_samples.append(sample)

        # クレンジング済みデータを保存
        cleansed_file = jsonl_file.with_name(f"{jsonl_file.stem}_cleansed{jsonl_file.suffix}")
        with open(cleansed_file, 'w', encoding='utf-8') as f:
            for sample in cleansed_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')

        print(f"Cleansed {len(cleansed_samples)} samples from {jsonl_file.name}")

    # 統計を保存
    stats_file = datasets_dir / "cleansing_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(cleansing_stats, f, indent=2, ensure_ascii=False)

    print(f"Cleansing statistics saved: {stats_file}")
    print(f"Total samples processed: {cleansing_stats['total_samples']}")
    print(f"Quality distribution: {cleansing_stats['quality_distribution']}")

    return cleansing_stats

def main():
    """メイン処理"""
    print("Simple Dataset Preprocessing Script")
    print("=" * 50)

    # コマンドライン引数
    parser = argparse.ArgumentParser(description="Simple dataset preprocessing")
    parser.add_argument('--create-mock', action='store_true', help='Create mock datasets')
    parser.add_argument('--create-soul-weights', action='store_true', help='Create soul weights dataset')
    parser.add_argument('--cleansing', action='store_true', help='Perform data cleansing')
    parser.add_argument('--all', action='store_true', help='Run all preprocessing steps')

    args = parser.parse_args()

    if args.all or not any([args.create_mock, args.create_soul_weights, args.cleansing]):
        print("Running all preprocessing steps...")
        args.create_mock = args.create_soul_weights = args.cleansing = True

    # 各ステップ実行
    if args.create_mock:
        create_mock_datasets()

    if args.create_soul_weights:
        create_soul_weights_dataset()

    if args.cleansing:
        perform_data_cleansing()

    print("\nPreprocessing completed successfully!")

if __name__ == '__main__':
    main()
