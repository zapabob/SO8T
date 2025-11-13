#!/bin/bash
# Flash Attention WSL2 インストールスクリプト
# Usage: bash scripts/utils/setup/install_flash_attn_wsl2.sh

set -e

echo "[INFO] Flash Attention WSL2 Installation"
echo "=========================================="

# WSL2環境の確認
echo "[STEP 1] Checking WSL2 environment..."
if [ -f /proc/version ]; then
    if grep -qi microsoft /proc/version; then
        echo "  [OK] Running in WSL2 environment"
    else
        echo "  [WARNING] Not running in WSL2 (may still work on Linux)"
    fi
else
    echo "  [INFO] Cannot detect WSL2, assuming Linux environment"
fi

# CUDA確認
echo ""
echo "[STEP 2] Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "  [OK] NVIDIA driver detected"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "  [WARNING] nvidia-smi not found"
    echo "  Please install NVIDIA drivers for WSL2"
    echo "  https://developer.nvidia.com/cuda-downloads"
fi

# Python確認
echo ""
echo "[STEP 3] Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_EXE=$(which python3)
    PYTHON_VERSION=$(python3 --version)
    echo "  [OK] Found Python: $PYTHON_EXE"
    echo "  Version: $PYTHON_VERSION"
else
    echo "  [ERROR] Python3 not found"
    echo "  Please install Python 3.10+"
    exit 1
fi

# pip確認とインストール
echo ""
echo "[STEP 3.5] Checking pip..."
if ! python3 -m pip --version &> /dev/null; then
    echo "  [WARNING] pip not found, installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3-pip
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3-pip
    else
        echo "  [ERROR] Cannot install pip automatically"
        echo "  Please install pip manually: sudo apt-get install python3-pip"
        exit 1
    fi
else
    echo "  [OK] pip is available"
fi

# PyTorch確認
echo ""
echo "[STEP 4] Checking PyTorch..."
if python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    echo "  [OK] PyTorch is installed"
else
    echo "  [WARNING] PyTorch not found or CUDA not available"
    echo "  Installing PyTorch with CUDA support..."
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# uv確認
echo ""
echo "[STEP 5] Checking uv..."
if command -v uv &> /dev/null; then
    echo "  [OK] uv is installed"
    uv --version
else
    echo "  [INFO] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Flash Attention インストール
echo ""
echo "[STEP 6] Installing flash-attention 2.5.8..."
echo "  This may take 10-30 minutes..."
echo ""

# Pythonインタープリターの検出
PYTHON_EXE=$(python3 -c "import sys; print(sys.executable)" 2>/dev/null)
if [ -z "$PYTHON_EXE" ]; then
    PYTHON_EXE=$(which python3)
fi

echo "  Using Python: $PYTHON_EXE"
echo ""

# インストール実行
if uv pip install --python "$PYTHON_EXE" "flash-attn==2.5.8" --no-build-isolation; then
    echo ""
    echo "[OK] Flash Attention installed successfully!"
    
    # 動作確認
    echo ""
    echo "[STEP 7] Verifying installation..."
    if python3 -c "from flash_attn import flash_attn_func; print('[OK] Flash Attention imported successfully')" 2>/dev/null; then
        echo "[OK] Flash Attention is ready to use!"
        echo ""
        echo "[INFO] Installation complete!"
        echo "  You can now use flash-attention in your Python code"
    else
        echo "[WARNING] Flash Attention installed but import failed"
        echo "  Please check the installation"
    fi
else
    echo ""
    echo "[ERROR] Flash Attention installation failed"
    echo "[INFO] Troubleshooting:" -ForegroundColor Yellow
    echo "  1. Check CUDA Toolkit is installed: nvidia-smi"
    echo "  2. Check PyTorch CUDA support: python3 -c 'import torch; print(torch.cuda.is_available())'"
    echo "  3. Try installing with pip directly: pip install flash-attn==2.5.8 --no-build-isolation"
    exit 1
fi

