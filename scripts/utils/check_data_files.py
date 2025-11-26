#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""データファイル確認スクリプト"""

from pathlib import Path
import json

print("="*80)
print("Data Files Status Check")
print("="*80)
print()

f1 = Path('D:/webdataset/aegis_v2.0/deep_research_thinking_dataset.jsonl')
f2 = Path('D:/webdataset/aegis_v2.0/deep_research_thinking_dataset_cleansed.jsonl')

print(f"[FILE 1] {f1.name}")
print(f"  Exists: {f1.exists()}")
if f1.exists():
    size = f1.stat().st_size
    print(f"  Size: {size:,} bytes")
    if size > 0:
        print("  First 3 lines:")
        try:
            with open(f1, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:3], 1):
                    print(f"    {i}: {line[:100]}...")
            print(f"  Total lines: {len(lines)}")
        except Exception as e:
            print(f"  Error reading file: {e}")
    else:
        print("  WARNING: File is empty (0 bytes)")
else:
    print("  WARNING: File does not exist")
print()

print(f"[FILE 2] {f2.name}")
print(f"  Exists: {f2.exists()}")
if f2.exists():
    size = f2.stat().st_size
    print(f"  Size: {size:,} bytes")
    if size > 0:
        print("  First 3 lines:")
        try:
            with open(f2, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:3], 1):
                    print(f"    {i}: {line[:100]}...")
            print(f"  Total lines: {len(lines)}")
        except Exception as e:
            print(f"  Error reading file: {e}")
    else:
        print("  WARNING: File is empty (0 bytes)")
else:
    print("  WARNING: File does not exist")
print()

print("="*80)



