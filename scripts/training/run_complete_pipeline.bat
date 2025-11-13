@echo off
REM Borea-Phi-3.5 SO8T/thinking Complete Pipeline Script
REM Executes Steps 1-5 sequentially
REM Note: For PowerShell, use run_complete_pipeline.ps1 instead

echo [PIPELINE] Borea-Phi-3.5 SO8T/thinking Complete Pipeline
echo ========================================================
echo.

REM Step 1: Dataset creation (skip if already exists)
echo [STEP 1] Checking dataset...
set DATASET=D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl
if not exist "%DATASET%" (
    echo [STEP 1] Creating /think format dataset...
    powershell -Command "$files = Get-ChildItem 'D:\webdataset\processed\four_class\four_class_*.jsonl' | Select-Object -ExpandProperty FullName; py -3 scripts\data\create_thinking_sft_dataset.py --inputs $files --output '%DATASET%'"
    if errorlevel 1 (
        echo [ERROR] Step 1 failed
        exit /b 1
    )
) else (
    echo [OK] Dataset already exists: %DATASET%
)
echo.

REM Step 2: Training execution (fast or full mode)
echo [STEP 2] Starting training...
set TRAINING_MODE=fast
if "%1"=="full" set TRAINING_MODE=full

if "%TRAINING_MODE%"=="fast" (
    echo [INFO] Using fast training configuration
    set CONFIG=configs\train_borea_phi35_so8t_thinking_fast.yaml
    set OUTPUT_DIR=D:\webdataset\checkpoints\training\borea_phi35_so8t_thinking_fast
) else (
    echo [INFO] Using full training configuration
    set CONFIG=configs\train_borea_phi35_so8t_thinking.yaml
    set OUTPUT_DIR=D:\webdataset\checkpoints\training\borea_phi35_so8t_thinking
)

py -3 scripts\training\train_borea_phi35_so8t_thinking.py ^
    --config %CONFIG% ^
    --dataset "%DATASET%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --auto-resume

if errorlevel 1 (
    echo [ERROR] Step 2 failed
    exit /b 1
)
echo.

REM Step 3: Baking process
echo [STEP 3] Baking SO8T rotations...
set TRAINED_MODEL=%OUTPUT_DIR%\final_model
set BAKED_MODEL=D:\webdataset\borea_phi35_so8t_thinking\baked_model

if not exist "%TRAINED_MODEL%" (
    echo [ERROR] Trained model not found: %TRAINED_MODEL%
    exit /b 1
)

py -3 scripts\training\bake_borea_phi35_so8t.py ^
    --model-path "%TRAINED_MODEL%" ^
    --output-path "%BAKED_MODEL%"

if errorlevel 1 (
    echo [ERROR] Step 3 failed
    exit /b 1
)
echo.

REM Step 4: GGUF conversion
echo [STEP 4] Converting to GGUF format...
set GGUF_OUTPUT_DIR=D:\webdataset\gguf_models\borea_phi35_so8t_thinking

if not exist "%BAKED_MODEL%" (
    echo [ERROR] Baked model not found: %BAKED_MODEL%
    exit /b 1
)

py -3 scripts\conversion\convert_borea_so8t_to_gguf.py ^
    --model-path "%BAKED_MODEL%" ^
    --output-dir "%GGUF_OUTPUT_DIR%" ^
    --model-name borea_phi35_so8t_thinking ^
    --quantization-types f16 q8_0 q4_k_m

if errorlevel 1 (
    echo [ERROR] Step 4 failed
    exit /b 1
)
echo.

REM Step 5: Ollama import and testing
echo [STEP 5] Importing to Ollama and testing...
set MODELFILE=modelfiles\borea_phi35_so8t_thinking.modelfile
set GGUF_FILE=%GGUF_OUTPUT_DIR%\borea_phi35_so8t_thinking_Q8_0.gguf

if not exist "%GGUF_FILE%" (
    echo [ERROR] GGUF file not found: %GGUF_FILE%
    exit /b 1
)

REM Update FROM path in Modelfile
powershell -Command "(Get-Content '%MODELFILE%') -replace 'FROM .*', 'FROM %GGUF_FILE%' | Set-Content '%MODELFILE%'"

echo [INFO] Creating Ollama model...
ollama create borea-phi35-so8t-thinking -f %MODELFILE%

if errorlevel 1 (
    echo [WARNING] Ollama import failed, but continuing...
) else (
    echo [OK] Ollama model created
    echo.
    echo [TEST] Testing /think format inference...
    ollama run borea-phi35-so8t-thinking "Solve this problem. First organize your thinking steps, then provide the final answer. What is 2+2?"
)
echo.

echo [SUCCESS] Complete pipeline finished!
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
