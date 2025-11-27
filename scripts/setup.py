#!/usr/bin/env python3
"""
SO8T Setup Script

Unified setup script for SO8T development environment.
Provides linear setup process with clear dependencies.
"""

import sys
import subprocess
from pathlib import Path
from so8t.utils import setup_environment

def main():
    """Main setup function with linear dependency flow."""

    print("🚀 SO8T Development Environment Setup")
    print("=" * 50)

    # Step 1: Check Python version
    print("📋 Step 1: Checking Python version...")
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]}")

    # Step 2: Install dependencies
    print("\n📦 Step 2: Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

    # Step 3: Setup directories
    print("\n📁 Step 3: Setting up directories...")
    dirs = [
        "D:/webdataset/models",
        "D:/webdataset/checkpoints",
        "D:/webdataset/gguf_models",
        "D:/webdataset/datasets",
        "D:/webdataset/logs"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")

    # Step 4: Verify installation
    print("\n🔍 Step 4: Verifying installation...")
    try:
        import torch
        import transformers
        import so8t
        print("✅ SO8T package imported successfully")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)

    print("\n🎉 Setup complete! Ready for development.")
    print("\nNext steps:")
    print("  1. python scripts/train.py    # Start training")
    print("  2. python scripts/eval.py     # Run evaluation")
    print("  3. python scripts/deploy.py   # Deploy model")

if __name__ == "__main__":
    main()































