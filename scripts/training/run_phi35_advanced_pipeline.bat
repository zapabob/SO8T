@echo off
REM Advanced Phi-3.5 SO8T Training Pipeline with Bayesian Optimization
REM SO8ViT/Thinking Adapter, Dynamic Thinking, Multimodal Integration, Meta-reasoning

chcp 65001 >nul
echo [PHI35-ADVANCED] Starting Advanced Phi-3.5 SO8T Training Pipeline
echo ======================================================================
echo Features:
echo   - SO8ViT/Thinking Adapter with SO8 rotation gates
echo   - Dynamic Thinking based on query types
echo   - Multimodal Integration (vision + audio)
echo   - Meta-reasoning Analysis
echo   - Bayesian Optimization of α parameter (α ∈ [0,1])
echo   - Orthogonal Error Logging
echo   - Comprehensive Benchmark Evaluation (ABC tests, ELYZA-100)
echo ======================================================================

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
set CONFIG_FILE=configs/train_phi35_so8t_annealing.yaml
set TIMESTAMP=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set OUTPUT_DIR=D:/webdataset/checkpoints/training/phi35_advanced_%TIMESTAMP%

REM ログディレクトリ作成
if not exist "logs" mkdir "logs"

echo [PHI35-ADVANCED] Configuration: %CONFIG_FILE%
echo [PHI35-ADVANCED] Output Directory: %OUTPUT_DIR%
echo [PHI35-ADVANCED] Timestamp: %TIMESTAMP%
echo.

REM GPUメモリ確認
echo [PHI35-ADVANCED] Checking system resources...
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

    # 推奨メモリチェック
    if total_memory < 24:
        print('WARNING: Recommended GPU memory is 24GB or more for advanced features')
    else:
        print('OK: Sufficient GPU memory for advanced Phi-3.5 training')
else:
    print('ERROR: CUDA not available - advanced features require GPU')
    exit /b 1
"
echo.

REM ディスク容量確認
echo [PHI35-ADVANCED] Checking disk space...
powershell -Command "
$D = Get-WmiObject -Class Win32_LogicalDisk -Filter 'DeviceID=\"D:\"'
$freeGB = [math]::Round($D.FreeSpace / 1GB, 2)
$totalGB = [math]::Round($D.Size / 1GB, 2)
Write-Host \"D: Drive - Free: ${freeGB}GB / Total: ${totalGB}GB\"
if ($freeGB -lt 100) {
    Write-Host 'ERROR: Need at least 100GB free space for advanced training' -ForegroundColor Red
    exit 1
} else {
    Write-Host 'OK: Sufficient disk space available' -ForegroundColor Green
}
"
echo.

REM Phi-3.5データセット存在確認
if not exist "D:/webdataset/phi35_integrated/phi35_ppo_optimized_integrated.jsonl" (
    echo [WARNING] Phi-3.5 dataset not found. Creating from integrated dataset...
    python scripts/data/convert_integrated_to_phi35.py ^
        --input "D:/webdataset/integrated_dataset_full.jsonl" ^
        --output "D:/webdataset/phi35_integrated" ^
        --cot-weight 3.0
    echo.
)

REM 出力ディレクトリ作成
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM 高度なPhi-3.5 SO8T学習実行
echo [PHI35-ADVANCED] Starting Advanced Phi-3.5 SO8T Training...
echo [PHI35-ADVANCED] Features: SO8ViT Adapter, Dynamic Thinking, Multimodal, Meta-reasoning, Bayesian Optimization
echo [PHI35-ADVANCED] Alpha Parameter: α ∈ [0,1] with Bayesian optimization
echo [PHI35-ADVANCED] SO8 Rotation Gates: Orthogonal error logging enabled
echo.

python scripts/training/train_phi35_advanced_pipeline.py ^
    --config "%CONFIG_FILE%" ^
    --output "%OUTPUT_DIR%"

if errorlevel 0 (
    echo.
    echo [SUCCESS] Advanced Phi-3.5 SO8T Training COMPLETED!
    echo [SUCCESS] Model saved to: %OUTPUT_DIR%/final_model
    echo [SUCCESS] Bayesian optimization results: %OUTPUT_DIR%/bayesian_optimization_results.json
    echo [SUCCESS] Thinking statistics: %OUTPUT_DIR%/thinking_statistics.json
    echo [SUCCESS] Orthogonal error logs: Check training logs
    echo.

    REM 評価実行
    echo [PHI35-ADVANCED] Running comprehensive evaluation...
    python scripts/training/train_phi35_advanced_pipeline.py ^
        --evaluate-only ^
        --model-path "%OUTPUT_DIR%/final_model" ^
        --output "%OUTPUT_DIR%"

    if errorlevel 0 (
        echo [SUCCESS] Comprehensive evaluation completed!
        echo [SUCCESS] Results saved to: %OUTPUT_DIR%/evaluation/
        echo [SUCCESS] Check comprehensive_evaluation_report.md for detailed results
    ) else (
        echo [WARNING] Evaluation failed, but training was successful
    )

) else (
    echo.
    echo [ERROR] Advanced Phi-3.5 SO8T training failed with error code %errorlevel%
    echo [INFO] Check logs for detailed error information
    echo [INFO] You can resume training with: --resume flag (if implemented)
)

echo.
echo [PHI35-ADVANCED] Pipeline execution completed at %DATE% %TIME%

REM 最終結果表示
if exist "%OUTPUT_DIR%/evaluation/comprehensive_evaluation_report.md" (
    echo.
    echo [RESULTS SUMMARY] ====================================
    type "%OUTPUT_DIR%/evaluation/comprehensive_evaluation_report.md" | findstr /C:"Winner:" /C:"Performance Difference:" /C:"Statistically Significant:"
    echo ====================================================
)

REM オーディオ通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

pause
