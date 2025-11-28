@echo off
REM Phi-3.5 SO8T /thinking Model Conversion Pipeline
REM Borea-Phi3.5-instinct-jpをPPO学習で/thinkingモデル化

chcp 65001 >nul
echo [PHI35-PIPELINE] Starting Phi-3.5 SO8T /thinking Model Conversion Pipeline
echo ========================================================================

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
set LOG_DIR=logs

REM ログディレクトリ作成
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [PHI35-PIPELINE] Configuration: %CONFIG_FILE%
echo [PHI35-PIPELINE] Timestamp: %TIMESTAMP%
echo [PHI35-PIPELINE] Pipeline Steps: HF Collection → Integration → Phi-3.5 Conversion → PPO Training → Evaluation
echo.

REM GPUメモリ確認
echo [PHI35-PIPELINE] Checking system resources...
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
else:
    print('CUDA not available - CPU training will be slow')
"
echo.

REM ディスク容量確認
echo [PHI35-PIPELINE] Checking disk space...
powershell -Command "
$D = Get-WmiObject -Class Win32_LogicalDisk -Filter 'DeviceID=\"D:\"'
$freeGB = [math]::Round($D.FreeSpace / 1GB, 2)
$totalGB = [math]::Round($D.Size / 1GB, 2)
Write-Host \"D: Drive - Free: ${freeGB}GB / Total: ${totalGB}GB\"
if ($freeGB -lt 50) {
    Write-Host '[WARNING] Low disk space! Need at least 50GB free space' -ForegroundColor Yellow
} else {
    Write-Host '[OK] Sufficient disk space available' -ForegroundColor Green
}
"
echo.

REM Phi-3.5パイプライン実行
echo [PHI35-PIPELINE] Starting Phi-3.5 SO8T /thinking conversion pipeline...
echo [PHI35-PIPELINE] This process may take several hours to days depending on dataset size
echo.

python scripts/pipeline/phi35_so8t_thinking_pipeline.py ^
    --config "%CONFIG_FILE%"

if errorlevel 0 (
    echo.
    echo [SUCCESS] Phi-3.5 SO8T /thinking Model Conversion Pipeline COMPLETED!
    echo [SUCCESS] Model saved to: D:/webdataset/checkpoints/training/phi35_pipeline_*/
    echo [SUCCESS] GGUF model available at: D:/webdataset/gguf_models/phi35_so8t_thinking/
) else (
    echo.
    echo [ERROR] Pipeline failed with error code %errorlevel%
    echo [INFO] Pipeline state saved. Use --resume flag to continue from last checkpoint
    echo [INFO] Command: scripts/pipeline/run_phi35_pipeline.bat --resume
)

echo.
echo [PHI35-PIPELINE] Pipeline execution completed at %DATE% %TIME%

REM オーディオ通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

pause
