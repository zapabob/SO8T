#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deep Researchログ確認スクリプト"""

from pathlib import Path

log_file = Path('logs/create_deep_research_thinking_dataset.log')

print(f"[INFO] Log file exists: {log_file.exists()}")
if log_file.exists():
    size = log_file.stat().st_size
    print(f"[INFO] Log file size: {size:,} bytes")
    print()
    print("Last 50 lines:")
    print("="*60)
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[-50:]:
            print(line.rstrip())
else:
    print("[INFO] Log file not found")


