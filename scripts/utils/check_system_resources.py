#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
システムリソース確認スクリプト
System Resource Check Script

GPUメモリとCUDAの状態を確認する
"""

import sys
import os

def check_gpu_memory():
    """GPUメモリを確認"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
            print(f"CUDA available: {device_name}")
            print(f"GPU Memory: {total_memory}GB")
            return True
        else:
            print("CUDA not available")
            return False
    except ImportError:
        print("PyTorch not available")
        return False
    except Exception as e:
        print(f"GPU check failed: {e}")
        return False

def check_disk_space():
    """ディスク容量を確認"""
    try:
        import psutil
        # D:ドライブの容量を確認
        d_drive = psutil.disk_usage('D:')
        total_gb = d_drive.total // (1024**3)
        free_gb = d_drive.free // (1024**3)
        used_gb = d_drive.used // (1024**3)
        percent_used = d_drive.percent

        print(f"D: Drive - Total: {total_gb}GB, Used: {used_gb}GB ({percent_used:.1f}%), Free: {free_gb}GB")

        # 警告: 空き容量が少ない場合
        if free_gb < 50:  # 50GB未満
            print("WARNING: Low disk space on D: drive!")
        elif free_gb < 100:  # 100GB未満
            print("NOTICE: Disk space on D: drive is getting low")

        return True
    except ImportError:
        print("psutil not available for disk check")
        return False
    except Exception as e:
        print(f"Disk check failed: {e}")
        return False

def main():
    """メイン関数"""
    print("=== System Resource Check ===")

    print("\n--- GPU Check ---")
    gpu_ok = check_gpu_memory()

    print("\n--- Disk Check ---")
    disk_ok = check_disk_space()

    print("\n--- Summary ---")
    if gpu_ok and disk_ok:
        print("[OK] All system checks passed")
        return 0
    else:
        print("[NG] Some system checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
