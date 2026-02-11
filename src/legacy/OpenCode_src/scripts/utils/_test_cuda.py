#!/usr/bin/env python3
"""CUDA環境テストスクリプト"""

import torch

print("[CUDA環境テスト]")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    # メモリ確認
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total Memory: {total_memory:.2f} GB")
else:
    print("[WARNING] CUDA is NOT available!")





