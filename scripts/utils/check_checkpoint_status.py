#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""チェックポイント状態確認スクリプト"""

from pathlib import Path
import json

checkpoint_dir = Path("D:/webdataset/checkpoints/aegis_v2_pipeline")
session_file = checkpoint_dir / "session.json"

print("[INFO] Checking checkpoint status...")
print(f"Checkpoint dir exists: {checkpoint_dir.exists()}")
print(f"Session file exists: {session_file.exists()}")
print("")

if session_file.exists():
    print("[OK] Session file found!")
    print("="*60)
    with open(session_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Session ID: {data.get('session_id', 'N/A')}")
    print(f"Stage: {data.get('stage', 'N/A')}")
    print(f"Started at: {data.get('started_at', 'N/A')}")
    print(f"Progress keys: {list(data.get('progress', {}).keys())}")
    print(f"Output files: {list(data.get('output_files', {}).keys())}")
    if data.get('error'):
        print(f"Previous error: {data.get('error')}")
    print("="*60)
else:
    print("[INFO] No session file found - will start fresh")


