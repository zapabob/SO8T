#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment Check Script
環境チェックスクリプト
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Pythonバージョン確認"""
    print(f"Python version: {sys.version}")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print("[OK] Python version OK")
        return True
    else:
        print("[NG] Python version too old (need 3.8+)")
        return False

def check_libraries():
    """ライブラリ確認"""
    required_libs = [
        'torch', 'transformers', 'datasets', 'numpy', 'pandas',
        'scipy', 'matplotlib', 'seaborn', 'tqdm', 'psutil'
    ]

    optional_libs = [
        'llama_cpp', 'lm_eval', 'lighteval'
    ]

    print("\nChecking required libraries...")
    missing_required = []

    for lib in required_libs:
        try:
            importlib.import_module(lib)
            print(f"[OK] {lib}")
        except ImportError:
            print(f"[NG] {lib} (MISSING)")
            missing_required.append(lib)

    print("\nChecking optional libraries...")
    for lib in optional_libs:
        try:
            importlib.import_module(lib)
            print(f"[OK] {lib}")
        except ImportError:
            print(f"[WARN] {lib} (not available)")

    return len(missing_required) == 0

def check_directories():
    """ディレクトリ確認"""
    required_dirs = [
        "D:/webdataset",
        "D:/webdataset/models",
        "D:/webdataset/gguf_models",
        "D:/webdataset/checkpoints",
        "D:/webdataset/results",
        "D:/webdataset/datasets",
        "external"
    ]

    print("\nChecking directories...")
    all_exist = True

    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"[OK] {dir_path}")
        else:
            print(f"[NG] {dir_path} (MISSING)")
            all_exist = False

    return all_exist

def check_cuda():
    """CUDA確認"""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"[OK] CUDA available: {device_count} device(s), {device_name}")
            return True
        else:
            print("[WARN] CUDA not available")
            return False
    except ImportError:
        print("[NG] PyTorch not available")
        return False

def main():
    """メイン関数"""
    print("=" * 60)
    print("SO8T Environment Check")
    print("=" * 60)

    checks = []

    # Pythonバージョン
    checks.append(("Python Version", check_python_version()))

    # ライブラリ
    checks.append(("Libraries", check_libraries()))

    # ディレクトリ
    checks.append(("Directories", check_directories()))

    # CUDA
    checks.append(("CUDA", check_cuda()))

    # 結果サマリー
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for check_name, passed in checks:
        status = "[OK] PASS" if passed else "[NG] FAIL"
        print(f"{check_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("[DONE] Environment check PASSED! Ready to run pipeline.")
        print("\nNext steps:")
        print("1. Run: scripts/setup/run_complete_pipeline_with_setup.bat")
        print("2. Or run individual phases as needed")
    else:
        print("[NG] Environment check FAILED! Please fix issues above.")
        print("\nTo fix:")
        print("1. Install missing Python libraries: pip install <library_name>")
        print("2. Create missing directories")
        print("3. Ensure CUDA drivers are installed (optional)")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
