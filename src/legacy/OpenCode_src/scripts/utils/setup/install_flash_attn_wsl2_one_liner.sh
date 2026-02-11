#!/bin/bash
# Flash Attention WSL2 ワンライナーインストール
# Usage: wsl bash < scripts/utils/setup/install_flash_attn_wsl2_one_liner.sh

set -e

cd /mnt/c/Users/downl/Desktop/SO8T 2>/dev/null || cd ~

echo "[INFO] Flash Attention WSL2 Installation (One-liner)"
echo "=================================================="

# pipインストール（必要に応じて）
if ! python3 -m pip --version &> /dev/null; then
    echo "[STEP 1] Installing pip..."
    sudo apt-get update -qq && sudo apt-get install -y python3-pip -qq
fi

# PyTorchインストール（必要に応じて）
if ! python3 -c "import torch" 2>/dev/null; then
    echo "[STEP 2] Installing PyTorch with CUDA..."
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
fi

# uvインストール（必要に応じて）
if ! command -v uv &> /dev/null; then
    echo "[STEP 3] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh -s
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Flash Attention インストール
echo "[STEP 4] Installing flash-attention 2.5.0..."
echo "  This may take 10-30 minutes..."
PYTHON_EXE=$(python3 -c "import sys; print(sys.executable)")
uv pip install --python "$PYTHON_EXE" "flash-attn==2.5.0" --no-build-isolation

# 動作確認
echo "[STEP 5] Verifying installation..."
if python3 -c "from flash_attn import flash_attn_func; print('[OK] Flash Attention installed successfully')" 2>&1; then
    echo "[OK] Installation complete!"
else
    echo "[ERROR] Installation verification failed"
    exit 1
fi

