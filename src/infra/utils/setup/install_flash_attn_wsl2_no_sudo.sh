#!/bin/bash
# Flash Attention WSL2 インストール（sudo不要版）
# Usage: wsl bash < scripts/utils/setup/install_flash_attn_wsl2_no_sudo.sh

set -e

cd /mnt/c/Users/downl/Desktop/SO8T 2>/dev/null || cd ~

echo "[INFO] Flash Attention WSL2 Installation (No sudo required)"
echo "=========================================================="

# Python確認
echo "[STEP 1] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found"
    echo "  Please install Python3: sudo apt-get install python3"
    exit 1
fi
PYTHON_EXE=$(python3 -c "import sys; print(sys.executable)")
echo "  [OK] Found Python: $PYTHON_EXE"

# pip確認（ユーザー権限でインストール）
echo ""
echo "[STEP 2] Checking pip..."
if ! python3 -m pip --version &> /dev/null; then
    echo "  [WARNING] pip not found, installing with get-pip.py..."
    # get-pip.pyをダウンロードしてユーザー権限でインストール
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python3 /tmp/get-pip.py --user
    export PATH="$HOME/.local/bin:$PATH"
    
    # 再度確認
    if ! python3 -m pip --version &> /dev/null; then
        echo "  [ERROR] pip installation failed"
        echo "  Please install pip manually:"
        echo "    wsl"
        echo "    sudo apt-get update && sudo apt-get install -y python3-pip"
        exit 1
    fi
    echo "  [OK] pip installed successfully"
else
    echo "  [OK] pip is available"
fi

# PyTorch確認（ユーザー権限でインストール）
echo ""
echo "[STEP 3] Checking PyTorch..."
if ! python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    echo "  [WARNING] PyTorch not found, installing with --user flag..."
    python3 -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "  [OK] PyTorch is installed"
fi

# uv確認（ユーザー権限でインストール）
echo ""
echo "[STEP 4] Checking uv..."
if ! command -v uv &> /dev/null; then
    echo "  [INFO] Installing uv..."
    if [ -d "$HOME/.cargo/bin" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    # uvをユーザー権限でインストール
    curl -LsSf https://astral.sh/uv/install.sh | sh -s
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if ! command -v uv &> /dev/null; then
        echo "  [WARNING] uv not in PATH, trying pip install..."
        python3 -m pip install --user uv
        export PATH="$HOME/.local/bin:$PATH"
    fi
else
    echo "  [OK] uv is installed"
fi

# Flash Attention インストール（ユーザー権限）
echo ""
echo "[STEP 5] Installing flash-attention 2.5.8..."
echo "  This may take 10-30 minutes..."
echo "  Using Python: $PYTHON_EXE"
echo ""

# PATHを確実に設定
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# uvでインストール
if uv pip install --python "$PYTHON_EXE" "flash-attn==2.5.0" --no-build-isolation; then
    echo ""
    echo "[OK] Flash Attention installed successfully!"
    
    # 動作確認
    echo ""
    echo "[STEP 6] Verifying installation..."
    if python3 -c "from flash_attn import flash_attn_func; print('[OK] Flash Attention imported successfully')" 2>&1; then
        echo "[OK] Flash Attention is ready to use!"
        echo ""
        echo "[INFO] Installation complete!"
        echo "  Flash Attention 2.5.8 is now available in WSL2"
    else
        echo "[WARNING] Flash Attention installed but import failed"
        echo "  Please check the installation"
    fi
else
    echo ""
    echo "[ERROR] Flash Attention installation failed"
    echo "[INFO] Troubleshooting:" -ForegroundColor Yellow
    echo "  1. Check CUDA: nvidia-smi"
    echo "  2. Check PyTorch CUDA: python3 -c 'import torch; print(torch.cuda.is_available())'"
    echo "  3. Try with pip directly: python3 -m pip install --user flash-attn==2.5.0 --no-build-isolation"
    exit 1
fi

