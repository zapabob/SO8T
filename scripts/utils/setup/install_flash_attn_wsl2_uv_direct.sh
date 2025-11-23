#!/bin/bash
# Flash Attention 2.5.0 Installation with uv in WSL2
# Direct execution script

set -e

echo "=========================================="
echo "Flash Attention 2.5.0 Installation (uv)"
echo "=========================================="
echo ""

# Change to project directory
cd /mnt/c/Users/downl/Desktop/SO8T || exit 1

# Step 1: Check Python
echo "[STEP 1] Checking Python..."
PYTHON_EXE=$(python3 -c "import sys; print(sys.executable)" 2>/dev/null || echo "")
if [ -z "$PYTHON_EXE" ]; then
    echo "  [ERROR] Python3 not found"
    exit 1
fi
echo "  [OK] Found Python: $PYTHON_EXE"
python3 --version

# Step 2: Install pip if needed
echo ""
echo "[STEP 2] Checking pip..."
if ! python3 -m pip --version &>/dev/null; then
    echo "  [WARNING] pip not found, installing with get-pip.py..."
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    echo "  [INFO] Using --break-system-packages flag (required for externally-managed environment)"
    python3 /tmp/get-pip.py --user --break-system-packages
    export PATH="$HOME/.local/bin:$PATH"
    
    if ! python3 -m pip --version &>/dev/null; then
        echo "  [ERROR] pip installation failed"
        exit 1
    fi
    echo "  [OK] pip installed successfully"
else
    echo "  [OK] pip is available"
    export PATH="$HOME/.local/bin:$PATH"
fi
python3 -m pip --version

# Step 3: Install PyTorch if needed
echo ""
echo "[STEP 3] Checking PyTorch..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "  [WARNING] PyTorch not found, installing with CUDA 12.1..."
    echo "  [INFO] Using --break-system-packages flag (required for externally-managed environment)"
    python3 -m pip install --user --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo "  [OK] PyTorch installed"
else
    echo "  [OK] PyTorch already installed"
    python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"
fi

# Step 3.5: Install tqdm and check logging
echo ""
echo "[STEP 3.5] Checking tqdm and logging..."
if ! python3 -c "import tqdm" 2>/dev/null; then
    echo "  [WARNING] tqdm not found, installing..."
    echo "  [INFO] Using --break-system-packages flag (required for externally-managed environment)"
    python3 -m pip install --user --break-system-packages tqdm
    echo "  [OK] tqdm installed"
else
    echo "  [OK] tqdm already installed"
    python3 -c "import tqdm; print(f'  tqdm version: {tqdm.__version__}')"
fi

if python3 -c "import logging" 2>/dev/null; then
    echo "  [OK] logging (standard library) is available"
else
    echo "  [WARNING] logging not found (unexpected)"
fi

# Step 4: Install uv if needed
echo ""
echo "[STEP 4] Checking uv..."
if ! command -v uv &>/dev/null; then
    echo "  [WARNING] uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if ! command -v uv &>/dev/null; then
        echo "  [ERROR] uv installation failed"
        exit 1
    fi
    echo "  [OK] uv installed successfully"
else
    echo "  [OK] uv is available"
    export PATH="$HOME/.cargo/bin:$PATH"
fi
uv --version

# Step 5: Install flash-attention 2.5.0 with uv
echo ""
echo "[STEP 5] Installing flash-attention 2.5.0 with uv..."
echo "  [WARNING] This will take 10-30 minutes..."
echo "  Using Python: $PYTHON_EXE"
echo "  [INFO] Using --break-system-packages flag (required for externally-managed environment)"
uv pip install --python "$PYTHON_EXE" 'flash-attn==2.5.0' --no-build-isolation --break-system-packages

# Step 6: Verify installation
echo ""
echo "[STEP 6] Verifying installation..."
if python3 -c "from flash_attn import flash_attn_func; print('[OK] Flash Attention 2.5.0 installed successfully!')" 2>/dev/null; then
    echo "  [OK] Flash Attention is working correctly"
    python3 -c "import flash_attn; print(f'  Version: {flash_attn.__version__}')" 2>/dev/null || echo "  Version check skipped"
else
    echo "  [ERROR] Flash Attention verification failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="

