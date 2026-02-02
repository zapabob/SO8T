@echo off
chcp 65001 >nul
REM SO8T学習開始スクリプト（Windows）
REM バックグラウンド実行、プロセス監視、GPU監視、自動再開

echo [START] SO8T Training Launch Script
echo ========================================

REM 環境チェック
where py >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found
    exit /b 1
)

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] nvidia-smi not found, GPU monitoring disabled
    set GPU_AVAILABLE=0
) else (
    set GPU_AVAILABLE=1
    echo [OK] GPU detected
)

REM ディレクトリ確認
if not exist "data\validated\" (
    echo [ERROR] Validated data directory not found
    echo Please run data collection and validation first
    exit /b 1
)

REM ログディレクトリ作成
if not exist "logs\" mkdir logs
set LOG_FILE=logs\training_%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%.log

REM GPU情報表示
if %GPU_AVAILABLE%==1 (
    echo.
    echo [GPU INFO]
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo.
)

REM 学習開始確認
echo.
echo [CONFIG] Training Configuration:
echo - Model: microsoft/phi-4-mini-instruct
echo - Data: data/validated/
echo - Output: outputs/so8t_ja_finetuned/
echo - Checkpoint Interval: 3 minutes
echo - Max Checkpoints: 5
echo - Expected Duration: ~33 hours
echo.
set /p CONFIRM="Start training? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo [CANCELLED] Training cancelled by user
    exit /b 0
)

echo.
echo [START] Starting training...
echo Log file: %LOG_FILE%
echo.

REM 学習実行（バックグラウンド）
start "SO8T Training" /MIN py -3 scripts/training/train_so8t_ja_full.py ^
    --model microsoft/phi-4-mini-instruct ^
    --data_dir data/validated ^
    --output_dir outputs/so8t_ja_finetuned ^
    > "%LOG_FILE%" 2>&1

REM プロセスID記録
for /f "tokens=2" %%a in ('tasklist /FI "WINDOWTITLE eq SO8T Training*" /FO LIST ^| find "PID:"') do set TRAINING_PID=%%a
echo %TRAINING_PID% > logs\training.pid

echo [OK] Training started (PID: %TRAINING_PID%)
echo.
echo [INFO] Monitoring commands:
echo   - View log: type "%LOG_FILE%"
echo   - GPU usage: nvidia-smi
echo   - Stop training: taskkill /PID %TRAINING_PID%
echo.

REM 監視ループ（オプション）
set /p MONITOR="Start monitoring? (Y/N): "
if /i not "%MONITOR%"=="Y" (
    echo [OK] Training running in background
    exit /b 0
)

echo.
echo [MONITOR] Starting training monitor...
echo Press Ctrl+C to stop monitoring (training will continue)
echo ========================================
echo.

:MONITOR_LOOP
REM プロセス確認
tasklist /FI "PID eq %TRAINING_PID%" | find "%TRAINING_PID%" >nul
if errorlevel 1 (
    echo.
    echo [WARNING] Training process not found
    echo [INFO] Check log file: %LOG_FILE%
    goto END_MONITOR
)

REM GPU監視
if %GPU_AVAILABLE%==1 (
    cls
    echo [MONITOR] Training Monitor (PID: %TRAINING_PID%)
    echo ========================================
    echo.
    echo [GPU STATUS]
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
    echo.
    echo [LOG TAIL] Last 10 lines:
    powershell -Command "Get-Content '%LOG_FILE%' -Tail 10 -ErrorAction SilentlyContinue"
    echo.
    echo Press Ctrl+C to stop monitoring
)

REM 30秒待機
timeout /t 30 /nobreak >nul
goto MONITOR_LOOP

:END_MONITOR
echo.
echo [INFO] Training monitor stopped
echo [INFO] Log file: %LOG_FILE%
echo.

REM 音声通知
if exist "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav" (
    echo [AUDIO] Playing completion notification...
    powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }"
)

echo [END] Training script completed
pause
