#!/bin/bash
# uv Python パス修正スクリプト (Linux/WSL2版)
# Usage: bash scripts/utils/setup/fix_uv_python_path.sh

echo "[INFO] Fixing uv Python path detection..."
echo "========================================="

# Pythonインタープリターの検出
echo "[STEP 1] Detecting Python interpreters..."

PYTHON_PATHS=()

# python3 コマンド経由で検出
if command -v python3 &> /dev/null; then
    PYTHON3_EXE=$(python3 -c "import sys; print(sys.executable)" 2>/dev/null)
    if [ -n "$PYTHON3_EXE" ]; then
        echo "  [OK] Found via python3: $PYTHON3_EXE"
        PYTHON_PATHS+=("$PYTHON3_EXE")
    fi
fi

# python コマンド経由で検出
if command -v python &> /dev/null; then
    PYTHON_EXE=$(python -c "import sys; print(sys.executable)" 2>/dev/null)
    if [ -n "$PYTHON_EXE" ]; then
        echo "  [OK] Found via python: $PYTHON_EXE"
        # 重複チェック
        if [[ ! " ${PYTHON_PATHS[@]} " =~ " ${PYTHON_EXE} " ]]; then
            PYTHON_PATHS+=("$PYTHON_EXE")
        fi
    fi
fi

if [ ${#PYTHON_PATHS[@]} -eq 0 ]; then
    echo "[ERROR] No Python interpreters found!"
    echo "  Please install Python 3.10+ and ensure it's in PATH"
    exit 1
fi

# 最初に見つかったPythonを使用
SELECTED_PYTHON="${PYTHON_PATHS[0]}"
echo ""
echo "[STEP 2] Selected Python interpreter:"
echo "  $SELECTED_PYTHON"

# Python バージョン確認
echo ""
echo "[STEP 3] Verifying Python version..."
$SELECTED_PYTHON --version

# uv のインストール確認
if ! command -v uv &> /dev/null; then
    echo ""
    echo "[WARNING] uv is not installed"
    echo "[INFO] To install uv:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  # or"
    echo "  pip install uv"
    echo "  # or"
    echo "  sudo snap install astral-uv"
    exit 1
fi

# uv でPythonパスをテスト
echo ""
echo "[STEP 4] Testing uv with explicit Python path..."
if uv pip install --python "$SELECTED_PYTHON" --version &> /dev/null; then
    echo "  [OK] uv can use this Python interpreter"
else
    echo "  [WARNING] uv test failed"
fi

echo ""
echo "[INFO] Usage with explicit Python path:"
echo "  uv pip install --python \"$SELECTED_PYTHON\" <package>"
echo ""
echo "[INFO] Or set UV_PYTHON environment variable:"
echo "  export UV_PYTHON=\"$SELECTED_PYTHON\""
echo "  uv pip install <package>"




























