#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセット収集テストスクリプト
Test script for dataset collection and preprocessing
"""

import os
import sys
import json
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_dataset_collection():
    """データセット収集テスト"""
    print("Testing dataset collection and preprocessing...")

    try:
        from scripts.data.dataset_collection_cleansing import DatasetCollectionCleansing

        # テスト設定
        config = {
            'output_dir': 'D:/webdataset/datasets/test_collection',
            'max_samples_per_source': 500,  # テスト用に制限
            'license_filter': ['mit', 'apache-2.0', 'cc-by-4.0'],
            'include_nsfw': False,  # テスト時は無効
            'soul_weights_samples': 200,  # テスト用に制限
            'quality_thresholds': {'acceptable': 0.6}
        }

        print(f"Config: {config}")

        # データセットコレクター初期化
        collector = DatasetCollectionCleansing(config)

        # ターゲットデータセット確認
        target_datasets = collector._get_target_datasets()
        print(f"Target datasets: {len(target_datasets)}")
        for i, ds in enumerate(target_datasets):
            print(f"  {i+1}. {ds['name']} ({ds.get('domain', 'unknown')})")

        # データセット収集実行
        print("\nStarting dataset collection...")
        result = collector.collect_and_cleansing_datasets()

        print(f"\nCollection completed!")
        print(f"Results: {result}")

        # 結果を保存
        output_file = 'dataset_collection_test_result.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print(f"Results saved to: {output_file}")

        return result

    except Exception as e:
        print(f"Error during dataset collection: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_soul_weights_generation():
    """魂の重み生成テスト"""
    print("\nTesting soul weights generation...")

    try:
        # 魂の重み生成スクリプトを直接実行
        import subprocess
        import sys

        result = subprocess.run([
            sys.executable, 'scripts/data/generate_soul_weights_dataset.py',
            '--output_dir', 'D:/webdataset/datasets/test_soul_weights'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)

        if result.returncode == 0:
            print("Soul weights dataset created successfully!")
            print(result.stdout)
            return "D:/webdataset/datasets/test_soul_weights"
        else:
            print(f"Soul weights generation failed: {result.stderr}")
            return None

    except Exception as e:
        print(f"Error during soul weights generation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    print("Dataset Collection Test Script")
    print("=" * 50)

    # ライブラリチェック
    print("Checking libraries...")

    try:
        import datasets
        print("✓ datasets available")
        has_datasets = True
    except ImportError as e:
        print(f"✗ datasets not available: {e}")
        has_datasets = False

    try:
        from PIL import Image
        print("✓ PIL available")
        has_pil = True
    except ImportError as e:
        print(f"✗ PIL not available: {e}")
        has_pil = False

    # 利用可能な機能のみテスト
    if not has_datasets:
        print("Skipping dataset collection test (datasets library not available)")
        collection_result = None
    else:
        collection_result = test_dataset_collection()

    # 魂の重み生成は常にテスト
    soul_result = test_soul_weights_generation()

    # 魂の重み生成テスト
    soul_result = test_soul_weights_generation()

    # データセット収集テスト
    collection_result = test_dataset_collection()

    print("\n" + "=" * 50)
    print("Test completed!")
    print(f"Soul weights: {'✓' if soul_result else '✗'}")
    print(f"Collection: {'✓' if collection_result else '✗'}")
