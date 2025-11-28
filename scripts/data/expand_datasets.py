#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T学習用データセット拡張スクリプト

HFから複数の大規模データセットをダウンロードして、
SO8T学習のデータ量を確保する
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.download_large_datasets import LargeDatasetDownloader


def main():
    print('=== SO8T学習用データセット拡張 ===')

    # 必要なデータセットリスト
    datasets_to_download = [
        {'name': 'microsoft/orca-math-word-problems-200k', 'category': 'mathematics', 'split': 'train'},
        {'name': 'allenai/math_qa', 'category': 'mathematics', 'split': 'train'},
        {'name': 'deepmind/code_contests', 'category': 'coding', 'split': 'train'},
        {'name': 'Anthropic/hh-rlhf', 'category': 'safety', 'split': 'train'},
        {'name': 'OpenAssistant/oasst1', 'category': 'conversation', 'split': 'train'},
        {'name': 'databricks/databricks-dolly-15k', 'category': 'instruction', 'split': 'train'},
    ]

    downloader = LargeDatasetDownloader()

    for dataset_info in datasets_to_download:
        try:
            print(f'Downloading {dataset_info["name"]} ({dataset_info["category"]})...')
            output_path = downloader.download_huggingface_dataset(
                dataset_name=dataset_info['name'],
                split=dataset_info['split'],
                output_dir=Path('D:/webdataset/datasets'),
                category=dataset_info['category']
            )
            if output_path:
                print(f'  SUCCESS: {output_path}')
            else:
                print(f'  FAILED: {dataset_info["name"]}')
        except Exception as e:
            print(f'  ERROR: {dataset_info["name"]} - {e}')


if __name__ == "__main__":
    main()
