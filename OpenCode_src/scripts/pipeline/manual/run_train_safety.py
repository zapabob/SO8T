#!/usr/bin/env python3
"""
訓練実行ラッパースクリプト
PowerShellの文字エンコーディング問題を回避するため
"""

import sys
import os
from pathlib import Path

# スクリプトのディレクトリに移動
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

print(f"Working directory: {os.getcwd()}")

# train_safety.pyをインポートして実行
sys.path.insert(0, str(script_dir))

# コマンドライン引数を設定
sys.argv = [
    "train_safety.py",
    "--config", "configs/train_safety.yaml",
    "--data_dir", "data",
    "--output_dir", "chk"
]

# train_safety.pyの main() を実行
from train_safety import main

if __name__ == "__main__":
    main()

