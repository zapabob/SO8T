@echo off
REM SO8T HF → GGUF → Quantize Pipeline (Windows)
REM RTX3060用のGGUF変換・量子化スクリプト

setlocal enabledelayedexpansion

REM 引数解析
set "MODEL_PATH=%~1"
if "!MODEL_PATH!"=="" set "MODEL_PATH=D:/webdataset/models/so8t_baked"

set "OUTPUT_DIR=%~2"
if "!OUTPUT_DIR!"=="" set "OUTPUT_DIR=D:/webdataset/gguf_models"

set "QUANTIZATION=%~3"
if "!QUANTIZATION!"=="" set "QUANTIZATION=Q5_K_M"

set "LLAMA_CPP_DIR=%~4"
if "!LLAMA_CPP_DIR!"=="" set "LLAMA_CPP_DIR=external/llama.cpp-master"

echo ==========================================
echo SO8T HF → GGUF → Quantize Pipeline
echo ==========================================
echo Model path: !MODEL_PATH!
echo Output dir: !OUTPUT_DIR!
echo Quantization: !QUANTIZATION!
echo llama.cpp dir: !LLAMA_CPP_DIR!
echo.

REM ディレクトリ作成
if not exist "!OUTPUT_DIR!" mkdir "!OUTPUT_DIR!"

REM モデル名を取得
for %%F in ("!MODEL_PATH!") do set "MODEL_NAME=%%~nxF"

REM Step 1: HF → GGUF変換 (F16)
echo [STEP 1] Converting HF model to GGUF (F16)...
set "F16_GGUF=!OUTPUT_DIR!/!MODEL_NAME!_f16.gguf"

if not exist "!F16_GGUF!" (
    py -3 "!LLAMA_CPP_DIR!/convert_hf_to_gguf.py" ^
        "!MODEL_PATH!" ^
        --outfile "!F16_GGUF!" ^
        --outtype f16
    
    if errorlevel 1 (
        echo [ERROR] GGUF conversion failed
        exit /b 1
    )
    echo [OK] F16 GGUF saved to: !F16_GGUF!
) else (
    echo [SKIP] F16 GGUF already exists: !F16_GGUF!
)

REM Step 2: 量子化
echo.
echo [STEP 2] Quantizing GGUF model (!QUANTIZATION!)...
set "QUANTIZED_GGUF=!OUTPUT_DIR!/!MODEL_NAME!_!QUANTIZATION!.gguf"

if not exist "!QUANTIZED_GGUF!" (
    REM quantize実行ファイルのパスを確認
    set "QUANTIZE_BIN="
    if exist "!LLAMA_CPP_DIR!/quantize.exe" (
        set "QUANTIZE_BIN=!LLAMA_CPP_DIR!/quantize.exe"
    ) else if exist "!LLAMA_CPP_DIR!/build/bin/quantize.exe" (
        set "QUANTIZE_BIN=!LLAMA_CPP_DIR!/build/bin/quantize.exe"
    ) else if exist "!LLAMA_CPP_DIR!/build/Release/bin/quantize.exe" (
        set "QUANTIZE_BIN=!LLAMA_CPP_DIR!/build/Release/bin/quantize.exe"
    )
    
    if "!QUANTIZE_BIN!"=="" (
        echo [ERROR] quantize binary not found. Please build llama.cpp first.
        echo   Build command: cd !LLAMA_CPP_DIR! ^&^& mkdir build ^&^& cd build ^&^& cmake .. ^&^& cmake --build . --config Release
        exit /b 1
    )
    
    "!QUANTIZE_BIN!" "!F16_GGUF!" "!QUANTIZED_GGUF!" "!QUANTIZATION!"
    
    if errorlevel 1 (
        echo [ERROR] Quantization failed
        exit /b 1
    )
    echo [OK] Quantized GGUF saved to: !QUANTIZED_GGUF!
) else (
    echo [SKIP] Quantized GGUF already exists: !QUANTIZED_GGUF!
)

REM Step 3: ファイルサイズ表示
echo.
echo [STEP 3] File sizes:
if exist "!F16_GGUF!" (
    for %%A in ("!F16_GGUF!") do echo   F16 GGUF: %%~zA bytes (!F16_GGUF!)
)
if exist "!QUANTIZED_GGUF!" (
    for %%A in ("!QUANTIZED_GGUF!") do echo   !QUANTIZATION! GGUF: %%~zA bytes (!QUANTIZED_GGUF!)
)

echo.
echo ==========================================
echo Conversion and quantization completed!
echo ==========================================
echo Quantized model: !QUANTIZED_GGUF!
echo.
echo Next steps:
echo   1. Run calibration: py -3 scripts/training/calibrate_aed.py --model !QUANTIZED_GGUF!
echo   2. Test inference: llama.cpp/main.exe -m !QUANTIZED_GGUF! -n 1024 -t 8 --temp 0.7






















































