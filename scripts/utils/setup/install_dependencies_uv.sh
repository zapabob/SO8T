#!/bin/bash
# uv 依存関係インストールスクリプト (Linux/WSL2版)
# Usage: bash scripts/utils/setup/install_dependencies_uv.sh

set -e

echo "[INFO] Installing dependencies with uv..." 

# Pythonインタープリターの検出
echo "[STEP 0] Detecting Python interpreter..."
if command -v python3 &> /dev/null; then
    PYTHON_EXE=$(python3 -c "import sys; print(sys.executable)" 2>/dev/null)
    if [ -n "$PYTHON_EXE" ]; then
        echo "  Found Python: $PYTHON_EXE"
        PYTHON_ARG="--python"
        PYTHON_VALUE="$PYTHON_EXE"
    else
        echo "[ERROR] Could not detect Python interpreter"
        exit 1
    fi
elif command -v python &> /dev/null; then
    PYTHON_EXE=$(python -c "import sys; print(sys.executable)" 2>/dev/null)
    if [ -n "$PYTHON_EXE" ]; then
        echo "  Found Python: $PYTHON_EXE"
        PYTHON_ARG="--python"
        PYTHON_VALUE="$PYTHON_EXE"
    else
        echo "[ERROR] Could not detect Python interpreter"
        exit 1
    fi
else
    echo "[ERROR] Python not found. Please install Python 3.10+"
    exit 1
fi

# uv のインストール確認
if ! command -v uv &> /dev/null; then
    echo "[WARNING] uv is not installed"
    echo "[INFO] Installing uv..."
    
    # curl経由でインストール
    if command -v curl &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    elif command -v pip &> /dev/null || command -v pip3 &> /dev/null; then
        pip install uv
    elif command -v snap &> /dev/null; then
        echo "[INFO] Installing uv via snap..."
        sudo snap install astral-uv
    else
        echo "[ERROR] Could not install uv. Please install manually:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
fi

# PyTorch with CUDA 12.1 support
echo "[STEP 1] Installing PyTorch with CUDA 12.1..."
uv pip install $PYTHON_ARG "$PYTHON_VALUE" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 基本依存関係のインストール
echo "[STEP 2] Installing core dependencies..."
uv pip install $PYTHON_ARG "$PYTHON_VALUE" -e .

# Flash Attention (Linux/WSL2ではインストール可能)
echo "[STEP 3] Installing flash-attention (optional)..."
read -p "  Install flash-attention? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv pip install $PYTHON_ARG "$PYTHON_VALUE" "flash-attn>=2.5.8" --no-build-isolation || {
        echo "[WARNING] Flash Attention installation failed, but it's optional"
        echo "  Standard attention will be used instead"
    }
else
    echo "[INFO] Skipping flash-attention installation"
fi

# 開発依存関係のインストール
echo "[STEP 4] Installing dev dependencies..."
uv pip install $PYTHON_ARG "$PYTHON_VALUE" -e ".[dev]"

echo "[OK] Dependencies installed successfully!"

