#!/bin/bash
# CUDA Toolkit Installation for WSL2
# Installs CUDA toolkit 12.1 for flash-attention compilation

set -e

echo "=========================================="
echo "CUDA Toolkit 12.1 Installation (WSL2)"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "[ERROR] This script must be run with sudo"
    echo "Usage: sudo bash scripts/utils/setup/install_cuda_toolkit_wsl2.sh"
    exit 1
fi

# Step 1: Check WSL2 environment
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

# Step 2: Check NVIDIA driver
echo ""
echo "[STEP 2] Checking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
    echo "  [OK] NVIDIA driver detected"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "  [ERROR] nvidia-smi not found"
    echo "  Please install NVIDIA drivers for WSL2 first"
    echo "  https://www.nvidia.com/Download/index.aspx"
    exit 1
fi

# Step 3: Add CUDA repository
echo ""
echo "[STEP 3] Adding CUDA repository..."
cd /tmp

# Download and install CUDA keyring
if [ ! -f cuda-keyring_1.1-1_all.deb ]; then
    echo "  [INFO] Downloading CUDA keyring..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb || {
        echo "  [ERROR] Failed to download CUDA keyring"
        echo "  [INFO] Trying alternative method..."
        # Alternative: use apt repository directly
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub || {
            echo "  [ERROR] Failed to add CUDA repository key"
            exit 1
        }
    }
    
    if [ -f cuda-keyring_1.1-1_all.deb ]; then
        echo "  [INFO] Installing CUDA keyring..."
        dpkg -i cuda-keyring_1.1-1_all.deb || {
            echo "  [WARNING] Keyring installation failed, trying alternative..."
        }
    fi
fi

# Add repository
echo "  [INFO] Adding CUDA repository..."
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /" || {
    echo "  [WARNING] Repository addition failed, trying manual method..."
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /" > /etc/apt/sources.list.d/cuda-wsl-ubuntu.list || {
        echo "  [ERROR] Failed to add repository"
        exit 1
    }
}

# Update package list
echo "  [INFO] Updating package list..."
apt-get update || {
    echo "  [ERROR] Failed to update package list"
    exit 1
}

# Fix broken dependencies first
echo "  [INFO] Fixing broken dependencies..."
apt-get install -f -y || {
    echo "  [WARNING] Failed to fix broken dependencies, continuing..."
}

# Step 4: Install CUDA toolkit
echo ""
echo "[STEP 4] Installing CUDA toolkit 12.1..."
echo "  [WARNING] This will download and install ~2GB of packages..."
echo "  [INFO] This may take 10-20 minutes..."
echo "  [INFO] Note: nsight-systems will be excluded due to libtinfo5 dependency issue"

# Install libtinfo5 compatibility package if available
echo "  [INFO] Installing libtinfo5 compatibility package..."
apt-get install -y libtinfo5 || {
    echo "  [WARNING] libtinfo5 not available, trying libtinfo6 compatibility..."
    # Try to create a symlink or install from alternative source
    apt-get install -y libtinfo6 || echo "  [WARNING] libtinfo6 installation failed, continuing..."
}

# Try installing CUDA toolkit without nsight-systems
echo "  [INFO] Installing CUDA toolkit components (excluding nsight-systems)..."
apt-get install -y --no-install-recommends \
    cuda-compiler-12-1 \
    cuda-libraries-12-1 \
    cuda-libraries-dev-12-1 \
    cuda-tools-12-1 \
    cuda-nvml-dev-12-1 || {
    echo "  [WARNING] Individual component installation failed"
    echo "  [INFO] Trying cuda-toolkit-12-1 with nsight-systems exclusion..."
    # Exclude nsight-systems from installation
    apt-get install -y cuda-toolkit-12-1 --no-install-recommends || {
        echo "  [WARNING] CUDA toolkit 12.1 installation failed"
        echo "  [INFO] Trying alternative: cuda-toolkit-12-0 (excluding nsight-systems)..."
        apt-get install -y cuda-toolkit-12-0 --no-install-recommends || {
            echo "  [WARNING] CUDA toolkit 12.0 installation also failed"
            echo "  [INFO] Trying minimal installation (compiler only)..."
            # Minimal installation: just the compiler
            apt-get install -y cuda-nvcc-12-1 cuda-cudart-12-1 cuda-cudart-dev-12-1 || {
                echo "  [ERROR] Minimal CUDA installation failed"
                echo "  [INFO] You may need to manually install CUDA toolkit"
                exit 1
            }
        }
    }
}

# Step 5: Set up environment
echo ""
echo "[STEP 5] Setting up CUDA environment..."

# Create profile script
cat > /etc/profile.d/cuda.sh << 'EOF'
# CUDA Environment Variables
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

# Check if cuda-12.1 exists, otherwise use cuda-12.0 or /usr/local/cuda
if [ ! -d /usr/local/cuda-12.1 ]; then
    if [ -d /usr/local/cuda-12.0 ]; then
        sed -i 's|cuda-12.1|cuda-12.0|g' /etc/profile.d/cuda.sh
        CUDA_HOME=/usr/local/cuda-12.0
    elif [ -d /usr/local/cuda ]; then
        sed -i 's|cuda-12.1|cuda|g' /etc/profile.d/cuda.sh
        CUDA_HOME=/usr/local/cuda
    else
        echo "  [WARNING] CUDA installation directory not found"
        CUDA_HOME=""
    fi
else
    CUDA_HOME=/usr/local/cuda-12.1
fi

chmod +x /etc/profile.d/cuda.sh

# Source the script for current session
source /etc/profile.d/cuda.sh

# Step 6: Verify installation
echo ""
echo "[STEP 6] Verifying CUDA installation..."
if [ -n "$CUDA_HOME" ] && [ -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "  [OK] CUDA toolkit installed at: $CUDA_HOME"
    "$CUDA_HOME/bin/nvcc" --version || echo "  [WARNING] nvcc version check failed"
else
    echo "  [WARNING] CUDA toolkit verification failed"
    echo "  [INFO] Checking alternative locations..."
    find /usr/local -name nvcc 2>/dev/null | head -1 || echo "  [ERROR] nvcc not found"
fi

echo ""
echo "=========================================="
echo "CUDA Toolkit Installation Completed!"
echo "=========================================="
echo ""
echo "[INFO] CUDA_HOME: $CUDA_HOME"
echo "[INFO] Environment variables set in /etc/profile.d/cuda.sh"
echo "[INFO] To use in current session, run:"
echo "  source /etc/profile.d/cuda.sh"
echo ""
echo "[INFO] Next step: Install flash-attention"
echo "  bash scripts/utils/setup/install_flash_attn_wsl2_with_cuda.sh"

