#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""学習ログ確認スクリプト"""

from pathlib import Path

log_file = Path("logs/train_so8t_quadruple_ppo.log")

print(f"[INFO] Log file exists: {log_file.exists()}")
if log_file.exists():
    print(f"[INFO] Log file size: {log_file.stat().st_size} bytes")
    print("")
    print("[Last 30 lines:]")
    print("="*60)
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[-30:]:
            print(line.rstrip())
else:
    print("[INFO] Log file not found yet")



