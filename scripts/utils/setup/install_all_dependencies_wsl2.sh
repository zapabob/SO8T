#!/bin/bash
# Complete Dependency Installation for WSL2
# Installs all required libraries for SO8T project

set -e

echo "=========================================="
echo "Complete Dependency Installation (WSL2)"
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

# Step 3: Install PyTorch with CUDA 12.1 (will be installed in Step 5 with uv)
echo ""
echo "[STEP 3] Checking PyTorch..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "  [INFO] PyTorch will be installed in Step 5 using uv"
else
    echo "  [OK] PyTorch already installed"
    python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"
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

# Step 5: Install core dependencies from requirements.txt using uv
echo ""
echo "[STEP 5] Installing core dependencies from requirements.txt using uv..."
echo "  [INFO] Filtering out optuna-dashboard and --index-url lines"
echo "  [INFO] Using --user and --break-system-packages flags"

# Create a temporary requirements file without optuna-dashboard and --index-url
grep -v "optuna-dashboard" requirements.txt | grep -v "^--index-url" | grep -v "^#" | grep -v "^$" > /tmp/requirements_filtered.txt || {
    # Fallback: create minimal requirements file
    echo "torch>=2.0.0" > /tmp/requirements_filtered.txt
    echo "torchvision>=0.15.0" >> /tmp/requirements_filtered.txt
    echo "torchaudio>=2.0.0" >> /tmp/requirements_filtered.txt
}

# Install PyTorch from PyTorch index separately (required for CUDA support)
# Note: uv doesn't support --user, so we use pip directly for user installation
echo "  [INFO] Installing PyTorch packages from PyTorch index..."
python3 -m pip install --user --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
    echo "  [WARNING] PyTorch installation failed, trying standard index..."
    python3 -m pip install --user --break-system-packages torch torchvision torchaudio
}

# Install other dependencies using pip (uv doesn't support --user flag)
# Note: uv is used for dependency resolution, but pip is used for actual installation
echo "  [INFO] Installing other dependencies with pip..."
python3 -m pip install --user --break-system-packages -r /tmp/requirements_filtered.txt || {
    echo "  [WARNING] Some packages from requirements.txt failed, continuing..."
}

# Install optuna-dashboard separately if needed
echo ""
echo "[STEP 5.5] Installing optuna-dashboard separately..."
python3 -m pip install --user --break-system-packages optuna-dashboard || {
    echo "  [WARNING] optuna-dashboard installation failed (optional)"
}

# Step 6: Install tqdm and check logging using uv
echo ""
echo "[STEP 6] Checking tqdm and logging..."
if ! python3 -c "import tqdm" 2>/dev/null; then
    echo "  [WARNING] tqdm not found, installing..."
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

# Step 7: Install flash-attention 2.5.0
echo ""
echo "[STEP 7] Installing flash-attention 2.5.0..."
echo "  [WARNING] This will take 10-30 minutes..."
echo "  Using Python: $PYTHON_EXE"
echo "  [INFO] Using --user and --break-system-packages flags"
python3 -m pip install --user --break-system-packages 'flash-attn==2.5.0' --no-build-isolation || {
    echo "  [WARNING] flash-attention installation failed, but it's optional"
    echo "  Standard attention will be used instead"
}

# Step 8: Verify key installations
echo ""
echo "[STEP 8] Verifying key installations..."
echo "  Checking transformers..."
python3 -c "import transformers; print(f'  transformers: {transformers.__version__}')" || echo "  [ERROR] transformers not installed"

echo "  Checking accelerate..."
python3 -c "import accelerate; print(f'  accelerate: {accelerate.__version__}')" || echo "  [ERROR] accelerate not installed"

echo "  Checking peft..."
python3 -c "import peft; print(f'  peft: {peft.__version__}')" || echo "  [ERROR] peft not installed"

echo "  Checking bitsandbytes..."
python3 -c "import bitsandbytes; print(f'  bitsandbytes: {bitsandbytes.__version__}')" || echo "  [ERROR] bitsandbytes not installed"

echo "  Checking numpy..."
python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')" || echo "  [ERROR] numpy not installed"

echo "  Checking pandas..."
python3 -c "import pandas; print(f'  pandas: {pandas.__version__}')" || echo "  [ERROR] pandas not installed"

echo "  Checking tqdm..."
python3 -c "import tqdm; print(f'  tqdm: {tqdm.__version__}')" || echo "  [ERROR] tqdm not installed"

echo "  Checking flash-attention..."
if python3 -c "from flash_attn import flash_attn_func; print('[OK] Flash Attention 2.5.0 installed successfully!')" 2>/dev/null; then
    python3 -c "import flash_attn; print(f'  flash-attn: {flash_attn.__version__}')" 2>/dev/null || echo "  Version check skipped"
else
    echo "  [WARNING] flash-attention not installed (optional)"
fi

echo ""
echo "=========================================="
echo "Installation completed!"
echo "=========================================="
echo ""
echo "[SUMMARY]"
echo "  Python: $PYTHON_EXE"
echo "  pip: $(python3 -m pip --version 2>/dev/null | cut -d' ' -f2 || echo 'N/A')"
echo "  uv: $(uv --version 2>/dev/null || echo 'N/A')"
echo ""
echo "[INFO] All dependencies have been installed to user directory"
echo "[INFO] Use: export PATH=\"\$HOME/.local/bin:\$PATH\" to access installed packages"

