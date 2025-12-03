#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセット確認スクリプト
"""

import json
from pathlib import Path

# データセット確認
datasets = [
    'data/aegis_v2_mathematical_enhanced_dataset.jsonl',
    'data/science_reasoning_dataset_final.jsonl',
    'data/train_sft_enhanced.jsonl'
]

total_samples = 0
for dataset_path in datasets:
    path = Path(dataset_path)
    if path.exists():
        count = 0
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception as e:
            print(f'Error reading {dataset_path}: {e}')
            continue

        print(f'{dataset_path}: {count} samples')
        total_samples += count
    else:
        print(f'{dataset_path}: NOT FOUND')

print(f'Total expected samples: {total_samples}')

# サンプルデータ確認
print('\nFirst few samples from first dataset:')
try:
    with open('data/aegis_v2_mathematical_enhanced_dataset.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            try:
                data = json.loads(line.strip())
                print(f'Sample {i}: {list(data.keys())}')
                if 'text' in data:
                    preview = data["text"][:100] if len(data["text"]) > 100 else data["text"]
                    print(f'  Text preview: {preview}...')
                elif 'instruction' in data:
                    print(f'  Instruction: {data["instruction"][:50]}...')
            except Exception as e:
                print(f'Sample {i}: INVALID JSON - {e}')
except Exception as e:
    print(f'Error: {e}')

