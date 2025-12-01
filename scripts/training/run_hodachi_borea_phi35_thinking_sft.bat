@echo off
chcp 65001 >nul
echo [SO8T] Starting HODACHI-Borea-phi3.5-mini-instinct-jp Thinking SFT Training
echo ========================================================================

set MODEL_NAME=HODACHI-Borea-phi3.5-mini-instinct-jp
set DATASET_PATH=data\sft_thinking\hodachi_borea_phi35_thinking_sft_dataset.jsonl
set OUTPUT_DIR=outputs\hodachi_borea_phi35_thinking_sft_%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%

echo [INFO] Model: %MODEL_NAME%
echo [INFO] Dataset: %DATASET_PATH%
echo [INFO] Output: %OUTPUT_DIR%

REM GPUメモリチェック
echo [INFO] Checking GPU memory...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current GPU: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] GPU check failed, continuing with CPU
)

REM メモリ使用量チェック
echo [INFO] Checking system memory...
python -c "import psutil; mem = psutil.virtual_memory(); print(f'Total: {mem.total/1024/1024/1024:.1f}GB, Available: {mem.available/1024/1024/1024:.1f}GB, Usage: {mem.percent:.1f}%%')" 2>nul

echo [STEP 1] Starting Thinking SFT Training...
echo [INFO] This will take several hours depending on hardware and dataset size
echo [INFO] Model will learn to output thinking process in <think> tags

python scripts/training/train_hodachi_borea_phi35_thinking_sft.py ^
    --model_name "%MODEL_NAME%" ^
    --dataset_path "%DATASET_PATH%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --batch_size 1 ^
    --learning_rate 0.00002 ^
    --num_epochs 3

if %errorlevel% equ 0 (
    echo [SUCCESS] HODACHI-Borea-phi3.5-mini-instinct-jp Thinking SFT completed successfully!
    echo [INFO] Model can now perform /thinking functionality
    echo [INFO] Use the model with prompts that require structured thinking
) else (
    echo [ERROR] Thinking SFT training failed with error code %errorlevel%
    echo [INFO] Check the logs in %OUTPUT_DIR% for details
)

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

echo ========================================================================
echo [SO8T] HODACHI-Borea-phi3.5-mini-instinct-jp Thinking SFT Training finished
echo [INFO] The model is now capable of /thinking functionality
echo [INFO] Output directory: %OUTPUT_DIR%
pause
