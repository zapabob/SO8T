#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""クレンジングデバッグスクリプト"""

import json
from pathlib import Path

# サンプルを読み込んで確認
input_file = Path("data/collected/synthetic_data.jsonl")
samples = []
with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        sample = json.loads(line.strip())
        samples.append(sample)

print("First 10 samples:")
for i, s in enumerate(samples):
    instruction = s.get("instruction", "")
    output = s.get("output", "")
    text = f"{instruction}\n{output}" if instruction and output else (instruction or output)
    print(f"\nSample {i+1}:")
    print(f"  instruction: {instruction[:100]}")
    print(f"  output: {output[:100]}")
    print(f"  combined text length: {len(text)}")
    print(f"  domain: {s.get('domain', 'N/A')}")









