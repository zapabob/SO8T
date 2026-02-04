@echo off
REM Complete SO8T Automation Pipeline - Power-on Auto Start
REM Borea-Phi3.5-instinct-jp → SO8T/thinking Multimodal Model

chcp 65001 >nul
echo [SO8T-AUTO] Starting Complete SO8T Automation Pipeline
echo =======================================================
echo This script will automatically:
echo 1. Collect multimodal datasets (text/image/audio/NSFW)
echo 2. Apply four-class labeling and data cleansing
echo 3. Train with PPO + SO8ViT + Multimodal integration
echo 4. Bake SO8T effects into standard Transformer
echo 5. Run comprehensive benchmarks with statistical analysis
echo 6. Upload to HuggingFace
echo 7. Remove scheduled task on completion
echo =======================================================

REM プロジェクトルートに移動
cd /d "%~dp0\..\.."
if errorlevel 1 (
    echo [ERROR] Failed to change to project root directory
    pause
    exit /b 1
)

REM Python環境設定
set PYTHONPATH=%CD%;%CD%\so8t-mmllm\src;%PYTHONPATH%

REM ログディレクトリ作成
if not exist "logs" mkdir "logs"

REM タイムスタンプ
set TIMESTAMP=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set LOG_FILE=logs\complete_pipeline_%TIMESTAMP%.log

echo [SO8T-AUTO] Pipeline started at %DATE% %TIME% > "%LOG_FILE%"
echo [SO8T-AUTO] Log file: %LOG_FILE%
echo.

REM GPUメモリチェック
echo [SO8T-AUTO] Checking GPU resources...
python -c "
import torch
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    total_memory = 0
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        total_memory += memory_gb
        print(f'GPU {i}: {props.name}, Memory: {memory_gb:.1f} GB')
    print(f'Total GPU Memory: {total_memory:.1f} GB')

    if total_memory < 24:
        print('WARNING: Recommended GPU memory is 24GB+ for full pipeline')
        print('Pipeline may take significantly longer or fail')
    else:
        print('OK: Sufficient GPU memory for complete pipeline')
else:
    print('CRITICAL ERROR: CUDA not available')
    echo CRITICAL ERROR: CUDA not available >> "%LOG_FILE%"
    pause
    exit /b 1
"
echo.

REM ディスク容量チェック
echo [SO8T-AUTO] Checking disk space...
powershell -Command "
$D = Get-WmiObject -Class Win32_LogicalDisk -Filter 'DeviceID=\"D:\"'
$freeGB = [math]::Round($D.FreeSpace / 1GB, 2)
Write-Host \"D: Drive - Free: ${freeGB}GB\"
if ($freeGB -lt 200) {
    Write-Host 'CRITICAL ERROR: Need at least 200GB free space for complete pipeline' -ForegroundColor Red
    exit 1
} elseif ($freeGB -lt 500) {
    Write-Host 'WARNING: Less than 500GB free space - monitoring required' -ForegroundColor Yellow
} else {
    Write-Host 'OK: Sufficient disk space available' -ForegroundColor Green
}
"
if errorlevel 1 (
    echo CRITICAL ERROR: Insufficient disk space >> "%LOG_FILE%"
    pause
    exit /b 1
)
echo.

REM 完全自動パイプライン実行
echo [SO8T-AUTO] Starting complete automation pipeline...
echo [SO8T-AUTO] This will take several hours to days depending on hardware
echo [SO8T-AUTO] Progress will be displayed in real-time with tqdm and debug output
echo [SO8T-AUTO] Log file: %LOG_FILE%
echo [SO8T-AUTO] Do not interrupt the process!
echo.

REM tqdmとloggingをリアルタイム表示しながら実行
python scripts/automation/complete_so8t_automation_pipeline.py 2>&1 | tee "%LOG_FILE%"

set PIPELINE_RESULT=%errorlevel%

REM 結果処理
if %PIPELINE_RESULT% equ 0 (
    echo.
    echo ====================================================
    echo [SUCCESS] COMPLETE SO8T AUTOMATION PIPELINE FINISHED!
    echo ====================================================
    echo Borea-Phi3.5-instinct-jp has been successfully transformed into:
    echo - SO8T/thinking multimodal model
    echo - With PPO training and SO8ViT integration
    echo - Comprehensive benchmark evaluation completed
    echo - Uploaded to HuggingFace
    echo - Scheduled task automatically removed
    echo ====================================================
    echo.

    REM 成功通知
    powershell -ExecutionPolicy Bypass -File "scripts/utils/play_audio_notification.ps1"

) else (
    echo.
    echo ====================================================
    echo [ERROR] COMPLETE SO8T AUTOMATION PIPELINE FAILED!
    echo ====================================================
    echo Error code: %PIPELINE_RESULT%
    echo Check log file for details: %LOG_FILE%
    echo.
    echo Possible solutions:
    echo 1. Check GPU memory and disk space
    echo 2. Verify internet connection for dataset downloads
    echo 3. Check HuggingFace token for uploads
    echo 4. Review error logs in logs/ directory
    echo ====================================================
    echo.

    REM エラー通知（異なる音声）
    powershell -ExecutionPolicy Bypass -Command "
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $player = New-Object System.Media.SoundPlayer
        $player.SoundLocation = 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav'
        $player.PlaySync()
        # エラー時は2回再生
        $player.PlaySync()
    } catch {
        [System.Console]::Beep(800, 1000)
        [System.Console]::Beep(600, 1000)
    }
    "

    REM エラーログ表示
    echo [LAST ERROR LOG] ====================================
    powershell -Command "Get-Content '%LOG_FILE%' -Tail 20"
    echo ===================================================
)

echo.
echo [SO8T-AUTO] Pipeline execution completed at %DATE% %TIME%
echo [SO8T-AUTO] Full log available at: %LOG_FILE%

REM 完了ステータス保存
echo COMPLETED_AT=%DATE% %TIME% >> "%LOG_FILE%"
echo EXIT_CODE=%PIPELINE_RESULT% >> "%LOG_FILE%"

if %PIPELINE_RESULT% equ 0 (
    echo STATUS=SUCCESS >> "%LOG_FILE%"
) else (
    echo STATUS=FAILED >> "%LOG_FILE%"
)

pause
