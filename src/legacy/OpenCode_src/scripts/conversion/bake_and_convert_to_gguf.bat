@echo off
REM Bake SO8T into Transformer and Convert to GGUF
REM SO(8)残差アダプターの効果を焼き込み、回転ゲートを削除してGGUF変換

chcp 65001 >nul
echo [BAKE-GGUF] Starting SO8T Baking and GGUF Conversion Pipeline
echo ===============================================================
echo This script will:
echo 1. Bake SO(8) residual adapter effects into Transformer weights
echo 2. Remove SO(8) rotation gates from the model
echo 3. Make the model appear as a single vector to existing ecosystems
echo 4. Convert the baked model to GGUF format
echo ===============================================================

REM プロジェクトルートに移動
cd /d "%~dp0\..\.."
if errorlevel 1 (
    echo [ERROR] Failed to change to project root directory
    pause
    exit /b 1
)

REM デフォルト設定
set MODEL_PATH=D:/webdataset/checkpoints/training/phi35_advanced_*/final_model
set BAKED_PATH=D:/webdataset/models/baked_for_gguf/phi35_so8t_baked
set GGUF_PATH=D:/webdataset/gguf_models/phi35_so8t_baked
set GGUF_NAME=phi35_so8t_baked

REM 引数処理
if "%~1"=="" (
    echo [INFO] Using default model path: %MODEL_PATH%
) else (
    set MODEL_PATH=%~1
)

if "%~2"=="" (
    set BAKED_PATH=D:/webdataset/models/baked_for_gguf/%GGUF_NAME%
) else (
    set BAKED_PATH=%~2
)

if "%~3"=="" (
    set GGUF_PATH=D:/webdataset/gguf_models/%GGUF_NAME%
) else (
    set GGUF_PATH=%~3
)

if "%~4"=="" (
    set GGUF_NAME=%GGUF_NAME%
) else (
    set GGUF_NAME=%~4
)

echo [BAKE-GGUF] Configuration:
echo [BAKE-GGUF]   Model Path: %MODEL_PATH%
echo [BAKE-GGUF]   Baked Path: %BAKED_PATH%
echo [BAKE-GGUF]   GGUF Path: %GGUF_PATH%
echo [BAKE-GGUF]   GGUF Name: %GGUF_NAME%
echo.

REM モデルパスの存在確認
if not exist "%MODEL_PATH%" (
    echo [ERROR] Model path does not exist: %MODEL_PATH%
    echo [INFO] Please check the model path or run training first
    pause
    exit /b 1
)

REM 出力ディレクトリ作成
if not exist "%BAKED_PATH%" mkdir "%BAKED_PATH%"
if not exist "%GGUF_PATH%" mkdir "%GGUF_PATH%"

REM GPUメモリ確認
echo [BAKE-GGUF] Checking system resources...
python -c "
import torch
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f'CUDA devices: {device_count}')
    total_memory = 0
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        total_memory += memory_gb
        print(f'GPU {i}: {props.name}, Memory: {memory_gb:.1f} GB')
    print(f'Total GPU Memory: {total_memory:.1f} GB')

    if total_memory < 16:
        print('WARNING: Recommended GPU memory is 16GB or more for baking')
    else:
        print('OK: Sufficient GPU memory for baking process')
else:
    print('ERROR: CUDA not available - baking requires GPU')
    exit /b 1
"
echo.

REM STEP 1: SO8T効果の焼き込み
echo [BAKE-GGUF] STEP 1: Baking SO8T effects into Transformer...
echo [BAKE-GGUF] This will integrate SO(8) residual adapter effects into weights
echo [BAKE-GGUF] and remove rotation gates for GGUF compatibility
echo.

python scripts/conversion/bake_so8t_into_transformer.py ^
    --model "%MODEL_PATH%" ^
    --output "%BAKED_PATH%"

if errorlevel 1 (
    echo [ERROR] Baking process failed with error code %errorlevel%
    pause
    exit /b 1
)

echo [SUCCESS] SO8T effects baked successfully!
echo [BAKE-GGUF] Baked model saved to: %BAKED_PATH%
echo.

REM STEP 2: 焼き込み済みモデルの検証
echo [BAKE-GGUF] STEP 2: Verifying baked model...
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print('Loading baked model...')
model = AutoModelForCausalLM.from_pretrained('%BAKED_PATH%', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('%BAKED_PATH%')

print(f'Model loaded successfully!')
print(f'Model type: {type(model)}')
print(f'Config: {model.config}')
print(f'Has SO8ViT adapter: {hasattr(model, \"so8vit_adapter\")}')
print(f'Has SO8 Trinality: {hasattr(model, \"so8_trinality_inference\")}')
print(f'Has Meta Analyzer: {hasattr(model, \"meta_analyzer\")}')
print('Model is clean and ready for GGUF conversion!')
"
echo.

REM STEP 3: GGUF変換
echo [BAKE-GGUF] STEP 3: Converting baked model to GGUF format...
echo [BAKE-GGUF] This will create GGUF files for Ollama and other GGUF-compatible systems
echo.

REM llama.cppのGGUF変換スクリプトを使用
if exist "external\llama.cpp-master\convert_hf_to_gguf.py" (
    echo [BAKE-GGUF] Using llama.cpp GGUF converter...

    REM F16バージョン
    python external/llama.cpp-master/convert_hf_to_gguf.py ^
        "%BAKED_PATH%" ^
        --outfile "%GGUF_PATH%/%GGUF_NAME%_F16.gguf" ^
        --outtype f16

    if errorlevel 1 (
        echo [WARNING] F16 conversion failed, trying Q8_0...
    ) else (
        echo [SUCCESS] F16 GGUF created: %GGUF_PATH%/%GGUF_NAME%_F16.gguf
    )

    REM Q8_0バージョン
    python external/llama.cpp-master/convert_hf_to_gguf.py ^
        "%BAKED_PATH%" ^
        --outfile "%GGUF_PATH%/%GGUF_NAME%_Q8_0.gguf" ^
        --outtype q8_0

    if errorlevel 1 (
        echo [WARNING] Q8_0 conversion failed, trying Q4_K_M...
    ) else (
        echo [SUCCESS] Q8_0 GGUF created: %GGUF_PATH%/%GGUF_NAME%_Q8_0.gguf
    )

    REM Q4_K_Mバージョン
    python external/llama.cpp-master/convert_hf_to_gguf.py ^
        "%BAKED_PATH%" ^
        --outfile "%GGUF_PATH%/%GGUF_NAME%_Q4_K_M.gguf" ^
        --outtype q4_k_m

    if errorlevel 1 (
        echo [ERROR] All GGUF conversions failed
        echo [INFO] Check llama.cpp installation and model compatibility
    ) else (
        echo [SUCCESS] Q4_K_M GGUF created: %GGUF_PATH%/%GGUF_NAME%_Q4_K_M.gguf
    )

) else (
    echo [WARNING] llama.cpp not found at external/llama.cpp-master/
    echo [INFO] Please install llama.cpp first for GGUF conversion
    echo [INFO] You can still use the baked model for other purposes
)

echo.
echo [BAKE-GGUF] STEP 4: Creating Ollama Modelfile...

REM Ollama Modelfile作成
set MODELFIE_PATH=%GGUF_PATH%/%GGUF_NAME%.modelfile
echo FROM %GGUF_PATH%/%GGUF_NAME%_Q8_0.gguf > "%MODELFIE_PATH%"
echo. >> "%MODELFIE_PATH%"
echo TEMPLATE """{{ .System }} >> "%MODELFIE_PATH%"
echo. >> "%MODELFIE_PATH%"
echo {{ .Prompt }}""" >> "%MODELFIE_PATH%"
echo. >> "%MODELFIE_PATH%"
echo PARAMETER temperature 0.7 >> "%MODELFIE_PATH%"
echo PARAMETER top_p 0.9 >> "%MODELFIE_PATH%"
echo PARAMETER top_k 40 >> "%MODELFIE_PATH%"
echo PARAMETER num_ctx 4096 >> "%MODELFIE_PATH%"

echo [SUCCESS] Ollama Modelfile created: %MODELFIE_PATH%

echo.
echo [BAKE-GGUF] STEP 5: Final verification...

REM GGUFファイルの存在確認
if exist "%GGUF_PATH%/%GGUF_NAME%_Q8_0.gguf" (
    for %%F in ("%GGUF_PATH%/%GGUF_NAME%_*.gguf") do (
        echo [SUCCESS] GGUF file: %%~nxF
    )
) else (
    echo [WARNING] No GGUF files found - conversion may have failed
)

echo.
echo ====================================================
echo [SUCCESS] SO8T Baking and GGUF Conversion COMPLETED!
echo ====================================================
echo Baked Model: %BAKED_PATH%
echo GGUF Models: %GGUF_PATH%
echo Ollama Modelfile: %MODELFIE_PATH%
echo.
echo Next steps:
echo 1. Test the baked model: ollama create %GGUF_NAME% -f %MODELFIE_PATH%
echo 2. Run inference: ollama run %GGUF_NAME% "Test prompt"
echo 3. Compare with original SO8T model performance
echo.
echo The model now appears as a standard Transformer to existing ecosystems!
echo SO(8) effects are baked into the weights, rotation gates are removed.
echo ====================================================

REM オーディオ通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

pause
