#!/bin/bash
# SO8T HF → GGUF → Quantize Pipeline
# RTX3060用のGGUF変換・量子化スクリプト

set -e

# 引数解析
MODEL_PATH="${1:-D:/webdataset/models/so8t_baked}"
OUTPUT_DIR="${2:-D:/webdataset/gguf_models}"
QUANTIZATION="${3:-Q5_K_M}"  # Q4_K_M or Q5_K_M
LLAMA_CPP_DIR="${4:-external/llama.cpp-master}"

echo "=========================================="
echo "SO8T HF → GGUF → Quantize Pipeline"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Quantization: $QUANTIZATION"
echo "llama.cpp dir: $LLAMA_CPP_DIR"
echo ""

# ディレクトリ作成
mkdir -p "$OUTPUT_DIR"

# モデル名を取得
MODEL_NAME=$(basename "$MODEL_PATH")

# Step 1: HF → GGUF変換 (F16)
echo "[STEP 1] Converting HF model to GGUF (F16)..."
F16_GGUF="$OUTPUT_DIR/${MODEL_NAME}_f16.gguf"

if [ ! -f "$F16_GGUF" ]; then
    python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
        "$MODEL_PATH" \
        --outfile "$F16_GGUF" \
        --outtype f16
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] GGUF conversion failed"
        exit 1
    fi
    echo "[OK] F16 GGUF saved to: $F16_GGUF"
else
    echo "[SKIP] F16 GGUF already exists: $F16_GGUF"
fi

# Step 2: 量子化
echo ""
echo "[STEP 2] Quantizing GGUF model ($QUANTIZATION)..."
QUANTIZED_GGUF="$OUTPUT_DIR/${MODEL_NAME}_${QUANTIZATION}.gguf"

if [ ! -f "$QUANTIZED_GGUF" ]; then
    # quantize実行ファイルのパスを確認
    if [ -f "$LLAMA_CPP_DIR/quantize" ]; then
        QUANTIZE_BIN="$LLAMA_CPP_DIR/quantize"
    elif [ -f "$LLAMA_CPP_DIR/build/bin/quantize" ]; then
        QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/quantize"
    elif [ -f "$LLAMA_CPP_DIR/build/Release/bin/quantize.exe" ]; then
        QUANTIZE_BIN="$LLAMA_CPP_DIR/build/Release/bin/quantize.exe"
    else
        echo "[ERROR] quantize binary not found. Please build llama.cpp first."
        echo "  Build command: cd $LLAMA_CPP_DIR && mkdir -p build && cd build && cmake .. && cmake --build . --config Release"
        exit 1
    fi
    
    "$QUANTIZE_BIN" "$F16_GGUF" "$QUANTIZED_GGUF" "$QUANTIZATION"
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Quantization failed"
        exit 1
    fi
    echo "[OK] Quantized GGUF saved to: $QUANTIZED_GGUF"
else
    echo "[SKIP] Quantized GGUF already exists: $QUANTIZED_GGUF"
fi

# Step 3: ファイルサイズ表示
echo ""
echo "[STEP 3] File sizes:"
if [ -f "$F16_GGUF" ]; then
    F16_SIZE=$(du -h "$F16_GGUF" | cut -f1)
    echo "  F16 GGUF: $F16_SIZE ($F16_GGUF)"
fi
if [ -f "$QUANTIZED_GGUF" ]; then
    QUANT_SIZE=$(du -h "$QUANTIZED_GGUF" | cut -f1)
    echo "  $QUANTIZATION GGUF: $QUANT_SIZE ($QUANTIZED_GGUF)"
fi

echo ""
echo "=========================================="
echo "Conversion and quantization completed!"
echo "=========================================="
echo "Quantized model: $QUANTIZED_GGUF"
echo ""
echo "Next steps:"
echo "  1. Run calibration: python scripts/training/calibrate_aed.py --model $QUANTIZED_GGUF"
echo "  2. Test inference: ./llama.cpp/main -m $QUANTIZED_GGUF -n 1024 -t 8 --temp 0.7"











