#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""パイプライン進捗確認スクリプト"""

from pathlib import Path
import json
from datetime import datetime

print("="*80)
print("AEGIS v2.0 Pipeline Progress Report")
print("="*80)
print()

# チェックポイント状態
checkpoint_dir = Path("D:/webdataset/checkpoints/aegis_v2_pipeline")
session_file = checkpoint_dir / "session.json"

print("[CHECKPOINT STATUS]")
print("-"*80)
if session_file.exists():
    with open(session_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Session ID: {data.get('session_id', 'N/A')}")
    print(f"Stage: {data.get('stage', 'N/A')}")
    print(f"Started at: {data.get('started_at', 'N/A')}")
    print(f"Progress: {json.dumps(data.get('progress', {}), indent=2, ensure_ascii=False)}")
    print(f"Output files: {list(data.get('output_files', {}).keys())}")
else:
    print("No checkpoint found")
print()

# 出力ファイル状態
output_dir = Path("D:/webdataset/aegis_v2.0")
print("[OUTPUT FILES STATUS]")
print("-"*80)
print(f"Output directory exists: {output_dir.exists()}")
if output_dir.exists():
    files = list(output_dir.glob('*'))
    print(f"Files/directories: {len(files)}")
    for f in files[:10]:
        if f.is_file():
            size = f.stat().st_size
            print(f"  [FILE] {f.name} ({size:,} bytes)")
        else:
            print(f"  [DIR]  {f.name}/")
print()

# 学習ログ状態
log_file = Path("logs/train_so8t_quadruple_ppo.log")
print("[TRAINING LOG STATUS]")
print("-"*80)
print(f"Log file exists: {log_file.exists()}")
if log_file.exists():
    size = log_file.stat().st_size
    print(f"Log file size: {size:,} bytes")
    print()
    print("Last 10 lines:")
    print("-"*80)
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[-10:]:
            print(line.rstrip())
else:
    print("Log file not created yet (training may not have started)")
print()

# パイプラインログ状態
pipeline_log = Path("logs/aegis_v2_pipeline.log")
print("[PIPELINE LOG STATUS]")
print("-"*80)
print(f"Log file exists: {pipeline_log.exists()}")
if pipeline_log.exists():
    size = pipeline_log.stat().st_size
    print(f"Log file size: {size:,} bytes")
    print()
    print("Last 5 lines:")
    print("-"*80)
    with open(pipeline_log, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[-5:]:
            print(line.rstrip())
print()

# プロセス状態
print("[PROCESS STATUS]")
print("-"*80)
print("Check if Python process is running:")
print("  tasklist /FI \"IMAGENAME eq python.exe\" /FO TABLE")
print()

print("="*80)






