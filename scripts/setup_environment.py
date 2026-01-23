#!/usr/bin/env python3
"""
RTX 3060 Optimized Environment Setup
Environment Auto Setup Script
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Command execution helper"""
    print(f"[EXEC] {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"[OK] {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def setup_conda_environment():
    """Setup conda environment"""
    commands = [
        ('conda create -n sunset-rtx3060 python=3.11 -y', 'Create conda environment'),
        ('conda activate sunset-rtx3060', 'Activate environment'),
    ]

    print("[INFO] Setting up conda environment...")
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    return True

def install_pytorch():
    """Install PyTorch CUDA version"""
    cmd = 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
    return run_command(cmd, 'Install PyTorch CUDA version')

def install_ml_packages():
    """Install machine learning packages"""
    packages = [
        'transformers[torch]',
        'accelerate',
        'bitsandbytes',
        'peft',
        'datasets',
        'evaluate',
        'scikit-learn',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'tqdm',
        'wandb',
        'python-dotenv'
    ]

    cmd = f'pip install {" ".join(packages)}'
    return run_command(cmd, 'Install ML packages')

def verify_installation():
    """Verify installation"""
    print("[INFO] Verifying installation...")
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
        print(f"[OK] CUDA: {torch.cuda.is_available()}")

        import transformers
        print(f"[OK] Transformers: {transformers.__version__}")

        import accelerate
        print(f"[OK] Accelerate: {accelerate.__version__}")

        return True
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("[START] RTX 3060 Sunset Pipeline Environment Setup")
    print("=" * 50)

    success = True

    # Optional conda environment setup
    if input("Create conda environment? (y/n): ").lower() == 'y':
        success &= setup_conda_environment()

    # Install PyTorch
    success &= install_pytorch()

    # Install ML packages
    success &= install_ml_packages()

    # Verify installation
    success &= verify_installation()

    if success:
        print("=" * 50)
        print("[SUCCESS] Setup completed!")
        print("Activate environment with: conda activate sunset-rtx3060")
        print("=" * 50)
    else:
        print("=" * 50)
        print("[ERROR] Setup failed")
        print("Check logs and try again")
        print("=" * 50)

if __name__ == "__main__":
    main()
