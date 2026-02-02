#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""メモリ使用量チェックスクリプト"""

import psutil
import os

def get_memory_info():
    """メモリ情報を取得"""
    # システムメモリ
    mem = psutil.virtual_memory()
    print("="*60)
    print("System Memory Information")
    print("="*60)
    print(f"Total: {mem.total / (1024**3):.2f} GB")
    print(f"Available: {mem.available / (1024**3):.2f} GB")
    print(f"Used: {mem.used / (1024**3):.2f} GB")
    print(f"Percent: {mem.percent:.1f}%")
    print()
    
    # ディスク容量
    disk = psutil.disk_usage('C:')
    print("="*60)
    print("Disk Space Information (C:)")
    print("="*60)
    print(f"Total: {disk.total / (1024**3):.2f} GB")
    print(f"Used: {disk.used / (1024**3):.2f} GB")
    print(f"Free: {disk.free / (1024**3):.2f} GB")
    print(f"Percent: {disk.percent:.1f}%")
    print()
    
    # ページングファイル（Windows）
    if os.name == 'nt':
        import subprocess
        try:
            result = subprocess.run(
                ['wmic', 'pagefileset', 'get', 'AllocatedBaseSize,CurrentUsage'],
                capture_output=True,
                text=True
            )
            print("="*60)
            print("Page File Information")
            print("="*60)
            print(result.stdout)
        except:
            print("Could not retrieve page file information")

if __name__ == "__main__":
    get_memory_info()









