@echo off
REM Phi-3.5 SO8T PPO Training with Alpha Gate Annealing
REM 四重推論とアルファゲートアニーリングによる学習

chcp 65001 >nul
echo [PHI35-PPO] Starting Phi-3.5 SO8T PPO Training with Alpha Gate Annealing
echo ========================================================

REM プロジェクトルートに移動
cd /d "%~dp0\..\.."
if errorlevel 1 (
    echo [ERROR] Failed to change to project root directory
    pause
    exit /b 1
)

REM Python環境設定
set PYTHONPATH=%CD%;%CD%\so8t-mmllm\src;%PYTHONPATH%

REM デフォルト設定
set CONFIG_FILE=configs\train_phi35_so8t_annealing.yaml
set TIMESTAMP=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set OUTPUT_DIR=D:/webdataset/checkpoints/training/phi35_so8t_annealing_%TIMESTAMP%

REM 出力ディレクトリ作成
if not exist "D:/webdataset/checkpoints/training" mkdir "D:/webdataset/checkpoints/training"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo [PHI35-PPO] Configuration: %CONFIG_FILE%
echo [PHI35-PPO] Output Directory: %OUTPUT_DIR%
echo [PHI35-PPO] Timestamp: %TIMESTAMP%
echo.

REM GPUメモリ確認
echo [PHI35-PPO] Checking GPU memory...
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.1f} GB')
else:
    print('CUDA not available')
"
echo.

REM Phi-3.5データセット存在確認
if not exist "D:/webdataset/phi35_integrated/phi35_ppo_optimized_integrated.jsonl" (
    echo [WARNING] Phi-3.5 dataset not found. Creating from integrated dataset...
    python scripts/data/convert_integrated_to_phi35.py ^
        --input "D:/webdataset/integrated_dataset.jsonl" ^
        --output "D:/webdataset/phi35_integrated" ^
        --cot-weight 3.0
    echo.
)

REM Phi-3.5 SO8T PPO学習実行
echo [PHI35-PPO] Starting Phi-3.5 SO8T PPO training...
echo [PHI35-PPO] Alpha Gate Annealing: α = Φ^(-2) with sigmoid annealing
echo [PHI35-PPO] Quadruple Thinking: task/safety/logic/ethics/practical/creative/final
echo.

python scripts/training/train_phi35_so8t_ppo_annealing.py ^
    --config "%CONFIG_FILE%" ^
    --output "%OUTPUT_DIR%"

if errorlevel 0 (
    echo [SUCCESS] Phi-3.5 SO8T PPO training completed successfully!
    echo [SUCCESS] Alpha gate annealing results saved to: %OUTPUT_DIR%/alpha_gate_annealing_results.json
) else (
    echo [ERROR] Phi-3.5 SO8T PPO training failed with error code %errorlevel%
)

REM 結果確認
if exist "%OUTPUT_DIR%/alpha_gate_annealing_results.json" (
    echo.
    echo [RESULTS] Alpha Gate Annealing Results:
    type "%OUTPUT_DIR%/alpha_gate_annealing_results.json"
)

echo.
echo [PHI35-PPO] Training session completed at %DATE% %TIME%

REM オーディオ通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

pause
