#!/bin/bash
# Fully Automated Flash Attention 2.5.0 Installation for WSL2
# This script automates the entire installation process

set -e

echo "=========================================="
echo "Fully Automated Flash Attention Installation"
echo "=========================================="
echo ""

# Change to project directory
cd /mnt/c/Users/downl/Desktop/SO8T || exit 1

# Step 1: Check if running as root (for CUDA toolkit installation)
NEED_SUDO=false
if [ "$EUID" -ne 0 ]; then
    NEED_SUDO=true
    echo "[INFO] This script requires sudo privileges for CUDA toolkit installation"
    echo "[INFO] You will be prompted for your password"
    echo ""
fi

# Step 2: Check Python
echo "[STEP 1] Checking Python..."
PYTHON_EXE=$(python3 -c "import sys; print(sys.executable)" 2>/dev/null || echo "")
if [ -z "$PYTHON_EXE" ]; then
    echo "  [ERROR] Python3 not found"
    exit 1
fi
echo "  [OK] Found Python: $PYTHON_EXE"
python3 --version

# Step 2.5: Set up user environment if running as root
if [ "$EUID" -eq 0 ]; then
    echo ""
    echo "[INFO] Running as root, setting up user environment..."
    # Try to find the default user
    DEFAULT_USER=$(getent passwd | grep -E "1000|1001" | cut -d: -f1 | head -1)
    if [ -n "$DEFAULT_USER" ]; then
        echo "  [INFO] Found default user: $DEFAULT_USER"
        export HOME="/home/$DEFAULT_USER"
        export USER="$DEFAULT_USER"
        # Add user's local bin to PATH
        export PATH="/home/$DEFAULT_USER/.local/bin:$PATH"
        # Add user's Python site-packages
        export PYTHONPATH="/home/$DEFAULT_USER/.local/lib/python3.12/site-packages:$PYTHONPATH"
    fi
fi

# Step 3: Check PyTorch
echo ""
echo "[STEP 2] Checking PyTorch..."
# Try to import torch with user's Python path
if [ "$EUID" -eq 0 ] && [ -n "$DEFAULT_USER" ]; then
    # Run as the default user
    if ! su - "$DEFAULT_USER" -c "python3 -c 'import torch'" 2>/dev/null; then
        echo "  [ERROR] PyTorch not found. Please install PyTorch first."
        exit 1
    fi
    # Get PyTorch info as the default user
    su - "$DEFAULT_USER" -c "python3 -c \"
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
\"" || {
        echo "  [ERROR] Failed to check PyTorch CUDA"
        exit 1
    }
else
    # Normal user execution
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

# Step 4: Check if CUDA toolkit is already installed
echo ""
echo "[STEP 3] Checking for existing CUDA toolkit..."
CUDA_HOME=""
for dir in /usr/local/cuda-12.1 /usr/local/cuda-12.0 /usr/local/cuda /opt/cuda; do
    if [ -d "$dir" ] && [ -f "$dir/bin/nvcc" ]; then
        CUDA_HOME="$dir"
        echo "  [OK] Found CUDA toolkit at: $CUDA_HOME"
        break
    fi
done

# If not found, check nvcc in PATH
if [ -z "$CUDA_HOME" ]; then
    if command -v nvcc &> /dev/null; then
        NVCC_PATH=$(which nvcc)
        CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
        echo "  [OK] Found CUDA toolkit from nvcc PATH at: $CUDA_HOME"
    fi
fi

# Step 5: Install CUDA toolkit if not found
if [ -z "$CUDA_HOME" ]; then
    echo ""
    echo "[STEP 4] CUDA toolkit not found. Installing..."
    echo "  [WARNING] This will download and install ~2GB of packages"
    echo "  [WARNING] This may take 10-20 minutes..."
    echo ""
    
    # If running as root, no need for sudo
    if [ "$EUID" -eq 0 ]; then
        echo "  [INFO] Running as root, installing CUDA toolkit directly..."
        bash scripts/utils/setup/install_cuda_toolkit_wsl2.sh || {
            echo "  [ERROR] CUDA toolkit installation failed"
            exit 1
        }
    elif [ "$NEED_SUDO" = true ]; then
        echo "  [INFO] Running CUDA toolkit installation with sudo..."
        echo "  [INFO] You will be prompted for your password"
        sudo bash scripts/utils/setup/install_cuda_toolkit_wsl2.sh || {
            echo "  [ERROR] CUDA toolkit installation failed"
            echo "  [INFO] If password is incorrect, you can:"
            echo "    1. Reset WSL2 password: wsl -u root passwd $USER"
            echo "    2. Or run as root: wsl -u root bash scripts/utils/setup/auto_install_flash_attn_wsl2.sh"
            exit 1
        }
    else
        bash scripts/utils/setup/install_cuda_toolkit_wsl2.sh || {
            echo "  [ERROR] CUDA toolkit installation failed"
            exit 1
        }
    fi
    
    # Source the environment
    if [ -f /etc/profile.d/cuda.sh ]; then
        source /etc/profile.d/cuda.sh
        CUDA_HOME="$CUDA_HOME"
        if [ -z "$CUDA_HOME" ]; then
            # Try to detect from the script
            CUDA_HOME=$(grep "export CUDA_HOME=" /etc/profile.d/cuda.sh | cut -d'=' -f2 | tr -d '"' | tr -d "'")
        fi
    fi
    
    # Verify installation
    if [ -z "$CUDA_HOME" ]; then
        for dir in /usr/local/cuda-12.1 /usr/local/cuda-12.0 /usr/local/cuda; do
            if [ -d "$dir" ] && [ -f "$dir/bin/nvcc" ]; then
                CUDA_HOME="$dir"
                break
            fi
        done
    fi
    
    if [ -z "$CUDA_HOME" ] || [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
        echo "  [ERROR] CUDA toolkit installation verification failed"
        exit 1
    fi
    
    echo "  [OK] CUDA toolkit installed at: $CUDA_HOME"
else
    echo "  [OK] CUDA toolkit already installed"
fi

# Step 6: Set up environment variables
echo ""
echo "[STEP 5] Setting up CUDA environment variables..."
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "  CUDA_HOME: $CUDA_HOME"
echo "  PATH updated: $CUDA_HOME/bin added"

# Verify nvcc is accessible
if [ -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "  [OK] nvcc found at: $CUDA_HOME/bin/nvcc"
    "$CUDA_HOME/bin/nvcc" --version | head -1 || echo "  [WARNING] nvcc version check failed"
else
    echo "  [WARNING] nvcc not found at $CUDA_HOME/bin/nvcc"
fi

# Step 7: Check if flash-attention is already installed
echo ""
echo "[STEP 6] Checking if flash-attention is already installed..."
if python3 -c "from flash_attn import flash_attn_func" 2>/dev/null; then
    echo "  [OK] flash-attention is already installed"
    python3 -c "import flash_attn; print(f'  Version: {flash_attn.__version__}')" 2>/dev/null || echo "  Version check skipped"
    echo ""
    echo "=========================================="
    echo "Flash Attention is already installed!"
    echo "=========================================="
    exit 0
fi

# Step 8: Install build dependencies
echo ""
echo "[STEP 7] Installing build dependencies..."
python3 -m pip install --user --break-system-packages wheel packaging ninja || {
    echo "  [WARNING] Some build dependencies failed, continuing..."
}

# Step 9: Install flash-attention 2.5.0
echo ""
echo "[STEP 8] Installing flash-attention 2.5.0..."
echo "  [WARNING] This will take 10-30 minutes..."
echo "  CUDA_HOME: $CUDA_HOME"
echo "  Python: $PYTHON_EXE"

# Set environment variables for the installation
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export MAX_JOBS=4  # Limit parallel jobs to avoid memory issues

# If running as root, install for the default user
if [ "$EUID" -eq 0 ] && [ -n "$DEFAULT_USER" ]; then
    echo "  [INFO] Installing for user: $DEFAULT_USER"
    su - "$DEFAULT_USER" -c "cd /mnt/c/Users/downl/Desktop/SO8T && export CUDA_HOME=\"$CUDA_HOME\" && export PATH=\"\$CUDA_HOME/bin:\$PATH\" && export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH\" && export MAX_JOBS=4 && python3 -m pip install --user --break-system-packages 'flash-attn==2.5.0' --no-build-isolation" || {
        echo "  [WARNING] Installation failed, trying with reduced parallelism..."
        su - "$DEFAULT_USER" -c "cd /mnt/c/Users/downl/Desktop/SO8T && export CUDA_HOME=\"$CUDA_HOME\" && export PATH=\"\$CUDA_HOME/bin:\$PATH\" && export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH\" && export MAX_JOBS=2 && python3 -m pip install --user --break-system-packages 'flash-attn==2.5.0' --no-build-isolation" || {
            echo ""
            echo "  [ERROR] flash-attention installation failed"
            exit 1
        }
    }
else
    python3 -m pip install --user --break-system-packages 'flash-attn==2.5.0' --no-build-isolation || {
    echo ""
    echo "  [WARNING] Installation failed, trying with reduced parallelism..."
    export MAX_JOBS=2
    python3 -m pip install --user --break-system-packages 'flash-attn==2.5.0' --no-build-isolation || {
        echo ""
        echo "  [ERROR] flash-attention installation failed"
        echo "  [INFO] Common issues:"
        echo "    1. CUDA toolkit not fully installed"
        echo "    2. Insufficient memory (try: export MAX_JOBS=1)"
        echo "    3. Missing build tools (gcc, g++, make)"
        echo ""
        echo "  [INFO] To install build tools:"
        echo "    sudo apt-get update"
        echo "    sudo apt-get install -y build-essential"
        echo ""
        exit 1
    }
}

# Step 10: Verify installation
echo ""
echo "[STEP 9] Verifying flash-attention installation..."
# If running as root, verify as the default user
if [ "$EUID" -eq 0 ] && [ -n "$DEFAULT_USER" ]; then
    if su - "$DEFAULT_USER" -c "python3 -c 'from flash_attn import flash_attn_func; print(\"[OK] Flash Attention 2.5.0 installed successfully!\")'" 2>/dev/null; then
        su - "$DEFAULT_USER" -c "python3 -c 'import flash_attn; print(f\"  flash-attn version: {flash_attn.__version__}\")'" 2>/dev/null || echo "  Version check skipped"
    else
        echo "  [ERROR] flash-attention import failed"
        exit 1
    fi
elif python3 -c "from flash_attn import flash_attn_func; print('[OK] Flash Attention 2.5.0 installed successfully!')" 2>/dev/null; then
    python3 -c "import flash_attn; print(f'  flash-attn version: {flash_attn.__version__}')" 2>/dev/null || echo "  Version check skipped"
    echo ""
    echo "=========================================="
    echo "Installation completed successfully!"
    echo "=========================================="
    echo ""
    echo "[INFO] Flash Attention 2.5.0 is now available"
    echo "[INFO] CUDA_HOME: $CUDA_HOME"
    echo ""
    echo "[INFO] To use in future sessions, add to ~/.bashrc:"
    echo "  export CUDA_HOME=\"$CUDA_HOME\""
    echo "  export PATH=\"\$CUDA_HOME/bin:\$PATH\""
    echo "  export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH\""
else
    echo "  [ERROR] flash-attention import failed"
    echo "  [INFO] Installation may have completed but import is failing"
    echo "  [INFO] Try: python3 -c 'from flash_attn import flash_attn_func'"
    exit 1
fi

