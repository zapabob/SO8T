#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""データセット形式確認スクリプト"""

from pathlib import Path
import json

dataset_file = Path("data/so8t_thinking_phi35_weighted_train.jsonl")

print(f"[INFO] Checking dataset format: {dataset_file}")
print(f"[INFO] File size: {dataset_file.stat().st_size:,} bytes")
print("")

if dataset_file.exists() and dataset_file.stat().st_size > 0:
    print("[SAMPLE 1]")
    print("="*60)
    with open(dataset_file, 'r', encoding='utf-8') as f:
        line = f.readline()
        if line.strip():
            sample = json.loads(line)
            print(f"Keys: {list(sample.keys())}")
            print("")
            for key, value in sample.items():
                if isinstance(value, str):
                    print(f"{key}: {value[:200]}...")
                else:
                    print(f"{key}: {value}")
    
    print("")
    print("[SAMPLE 2]")
    print("="*60)
    with open(dataset_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) > 1:
            sample2 = json.loads(lines[1])
            print(f"Keys: {list(sample2.keys())}")
            print("")
            for key, value in sample2.items():
                if isinstance(value, str):
                    print(f"{key}: {value[:200]}...")
                else:
                    print(f"{key}: {value}")
else:
    print("[ERROR] File not found or empty")


