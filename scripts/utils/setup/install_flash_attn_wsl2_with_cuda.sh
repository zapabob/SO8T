#!/bin/bash
# Flash Attention 2.5.0 Installation with CUDA_HOME Setup for WSL2
# Usage: bash scripts/utils/setup/install_flash_attn_wsl2_with_cuda.sh

set -e

echo "=========================================="
echo "Flash Attention 2.5.0 Installation (WSL2)"
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

# Step 2: Check PyTorch and CUDA
echo ""
echo "[STEP 2] Checking PyTorch and CUDA..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "  [ERROR] PyTorch not found. Please install PyTorch first."
    exit 1
fi

python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
" || {
    echo "  [ERROR] Failed to check PyTorch CUDA"
    exit 1
}

# Step 3: Find CUDA installation
echo ""
echo "[STEP 3] Finding CUDA installation..."

# Check common CUDA locations
CUDA_HOME=""
for dir in /usr/local/cuda-12.1 /usr/local/cuda-12.0 /usr/local/cuda /opt/cuda /usr/cuda; do
    if [ -d "$dir" ] && [ -f "$dir/bin/nvcc" ]; then
        CUDA_HOME="$dir"
        echo "  [OK] Found CUDA at: $CUDA_HOME"
        break
    fi
done

# If not found, try to find from PyTorch
if [ -z "$CUDA_HOME" ]; then
    echo "  [INFO] CUDA not found in common locations, checking PyTorch CUDA path..."
    PYTORCH_CUDA=$(python3 -c "import torch; import os; print(os.path.dirname(os.path.dirname(torch.utils.cpp_extension.CUDA_HOME)))" 2>/dev/null || echo "")
    if [ -n "$PYTORCH_CUDA" ] && [ -d "$PYTORCH_CUDA" ]; then
        CUDA_HOME="$PYTORCH_CUDA"
        echo "  [OK] Found CUDA from PyTorch at: $CUDA_HOME"
    fi
fi

# If still not found, check if nvcc is in PATH
if [ -z "$CUDA_HOME" ]; then
    if command -v nvcc &> /dev/null; then
        NVCC_PATH=$(which nvcc)
        CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
        echo "  [OK] Found CUDA from nvcc PATH at: $CUDA_HOME"
    fi
fi

# If still not found, try to use PyTorch's bundled CUDA
if [ -z "$CUDA_HOME" ]; then
    echo "  [WARNING] CUDA installation not found in standard locations"
    echo "  [INFO] Attempting to use PyTorch's bundled CUDA libraries..."
    # PyTorch bundles CUDA libraries, but we need the full toolkit for compilation
    # Try to find CUDA toolkit from conda or pip installation
    PYTORCH_LIB=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))" 2>/dev/null || echo "")
    if [ -n "$PYTORCH_LIB" ]; then
        # Check if there's a CUDA toolkit in the same environment
        POSSIBLE_CUDA=$(dirname "$PYTORCH_LIB")/cuda
        if [ -d "$POSSIBLE_CUDA" ]; then
            CUDA_HOME="$POSSIBLE_CUDA"
            echo "  [INFO] Using PyTorch bundled CUDA at: $CUDA_HOME"
        fi
    fi
fi

# Final check: if CUDA_HOME is still empty, we need to install CUDA toolkit
if [ -z "$CUDA_HOME" ]; then
    echo ""
    echo "  [ERROR] CUDA toolkit not found!"
    echo "  [INFO] Flash-attention requires CUDA toolkit for compilation."
    echo "  [INFO] Please install CUDA toolkit for WSL2:"
    echo "    https://developer.nvidia.com/cuda-downloads"
    echo "  [INFO] Or set CUDA_HOME environment variable manually:"
    echo "    export CUDA_HOME=/usr/local/cuda-12.1"
    echo ""
    echo "  [INFO] For WSL2, you can also install CUDA toolkit via:"
    echo "    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
    echo "    sudo dpkg -i cuda-keyring_1.1-1_all.deb"
    echo "    sudo apt-get update"
    echo "    sudo apt-get -y install cuda-toolkit-12-1"
    exit 1
fi

# Step 4: Set CUDA_HOME and verify
echo ""
echo "[STEP 4] Setting CUDA_HOME environment variable..."
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "  CUDA_HOME: $CUDA_HOME"
echo "  PATH updated: $CUDA_HOME/bin added"

# Verify nvcc is accessible
if [ -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "  [OK] nvcc found at: $CUDA_HOME/bin/nvcc"
    "$CUDA_HOME/bin/nvcc" --version || echo "  [WARNING] nvcc version check failed"
else
    echo "  [WARNING] nvcc not found at $CUDA_HOME/bin/nvcc"
    echo "  [INFO] Continuing anyway, PyTorch may have bundled CUDA compiler"
fi

# Step 5: Install build dependencies
echo ""
echo "[STEP 5] Installing build dependencies..."
python3 -m pip install --user --break-system-packages wheel packaging ninja || {
    echo "  [WARNING] Some build dependencies failed, continuing..."
}

# Step 6: Install flash-attention 2.5.0
echo ""
echo "[STEP 6] Installing flash-attention 2.5.0..."
echo "  [WARNING] This will take 10-30 minutes..."
echo "  CUDA_HOME: $CUDA_HOME"
echo "  Python: $PYTHON_EXE"

# Set environment variables for the installation
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export MAX_JOBS=4  # Limit parallel jobs to avoid memory issues

python3 -m pip install --user --break-system-packages 'flash-attn==2.5.0' --no-build-isolation || {
    echo ""
    echo "  [ERROR] flash-attention installation failed"
    echo "  [INFO] Common issues:"
    echo "    1. CUDA toolkit not fully installed"
    echo "    2. Insufficient memory (try: export MAX_JOBS=2)"
    echo "    3. Missing build tools (gcc, g++, make)"
    echo ""
    echo "  [INFO] To install build tools:"
    echo "    sudo apt-get update"
    echo "    sudo apt-get install -y build-essential"
    echo ""
    exit 1
}

# Step 7: Verify installation
echo ""
echo "[STEP 7] Verifying flash-attention installation..."
if python3 -c "from flash_attn import flash_attn_func; print('[OK] Flash Attention 2.5.0 installed successfully!')" 2>/dev/null; then
    python3 -c "import flash_attn; print(f'  flash-attn version: {flash_attn.__version__}')" 2>/dev/null || echo "  Version check skipped"
    echo ""
    echo "=========================================="
    echo "Installation completed successfully!"
    echo "=========================================="
    echo ""
    echo "[INFO] To use flash-attention in future sessions, add to ~/.bashrc:"
    echo "  export CUDA_HOME=\"$CUDA_HOME\""
    echo "  export PATH=\"\$CUDA_HOME/bin:\$PATH\""
    echo "  export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH\""
else
    echo "  [ERROR] flash-attention import failed"
    echo "  [INFO] Installation may have completed but import is failing"
    echo "  [INFO] Try: python3 -c 'from flash_attn import flash_attn_func'"
    exit 1
fi





































































































