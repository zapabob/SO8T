#!/bin/bash
# Flash Attention インストールスクリプト (uv版, Linux/WSL2)
# Usage: bash scripts/utils/setup/install_flash_attn_uv.sh

set -e

echo "[INFO] Flash Attention Installation with uv"
echo "==========================================="

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

# 環境確認
echo "[STEP 1] Checking environment..."
python3 --version || python --version
echo "  Python: $(python3 --version 2>&1 || python --version 2>&1)"

# CUDA確認
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null; then
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    echo "  CUDA Available: $CUDA_AVAILABLE"
else
    echo "  CUDA: Not checked (torch not installed)"
fi

# uv のインストール確認
if ! command -v uv &> /dev/null; then
    echo "[WARNING] uv is not installed"
    echo "[INFO] Installing uv..."
    
    if command -v curl &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    elif command -v pip &> /dev/null || command -v pip3 &> /dev/null; then
        pip install uv || pip3 install uv
    elif command -v snap &> /dev/null; then
        echo "[INFO] Installing uv via snap..."
        sudo snap install astral-uv
    else
        echo "[ERROR] Could not install uv. Please install manually:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
fi

# Flash Attention インストール
echo "[STEP 2] Installing flash-attention with uv..."
if uv pip install $PYTHON_ARG "$PYTHON_VALUE" "flash-attn>=2.5.8" --no-build-isolation; then
    echo "[OK] Flash Attention installed successfully!"
    
    # 動作確認
    echo "[STEP 3] Verifying installation..."
    if python3 -c "from flash_attn import flash_attn_func; print('[OK] Flash Attention imported successfully')" 2>/dev/null || \
       python -c "from flash_attn import flash_attn_func; print('[OK] Flash Attention imported successfully')" 2>/dev/null; then
        echo "[OK] Flash Attention is ready to use!"
    else
        echo "[WARNING] Flash Attention installed but import failed"
    fi
else
    echo "[ERROR] Flash Attention installation failed"
    echo "[INFO] Flash Attention is optional - the system will work without it"
    echo "  Standard attention will be used instead"
    exit 1
fi

