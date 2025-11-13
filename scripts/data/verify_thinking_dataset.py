#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""検証用スクリプト"""

import json
from pathlib import Path

dataset_path = Path(r"D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl")

with open(dataset_path, 'r', encoding='utf-8') as f:
    samples = [json.loads(line) for line in f.readlines()]

print(f"Total samples: {len(samples)}")
print(f"Samples with thinking format: {sum(1 for s in samples if '# 思考ステップ' in s.get('output', ''))}")
print(f"Samples with final answer: {sum(1 for s in samples if '# 最終回答' in s.get('output', ''))}")

# 最初の3サンプルを表示
print("\n--- First 3 samples ---")
for i, s in enumerate(samples[:3]):
    print(f"\nSample {i+1}:")
    print(f"Instruction: {s.get('instruction', '')[:100]}...")
    print(f"Output length: {len(s.get('output', ''))}")
    if '# 思考ステップ' in s.get('output', ''):
        print("✓ Contains thinking format")
    if '# 最終回答' in s.get('output', ''):
        print("✓ Contains final answer")














