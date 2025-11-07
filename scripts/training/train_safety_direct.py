#!/usr/bin/env python3
"""
安全重視SO8T訓練スクリプト（直接実行版）
PowerShellの文字エンコーディング問題を回避
"""

import os
import sys
from pathlib import Path

# プロジェクトディレクトリに移動
project_dir = Path(__file__).parent.absolute()
os.chdir(project_dir)
print(f"Working directory: {os.getcwd()}")

# 必要なモジュールをインポート
sys.path.insert(0, str(project_dir))

# 訓練スクリプトを実行
if __name__ == "__main__":
    try:
        from train_safety import main
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
