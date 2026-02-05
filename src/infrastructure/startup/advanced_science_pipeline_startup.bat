@echo off
REM Advanced Science Reasoning Pipeline Power-On Startup Script
REM 電源投入時にタスクスケジューラから自動起動されるスクリプト

REM ============================================================
REM AUTOSTART KILL-SWITCH (default: disabled)
REM Set SO8T_ENABLE_AUTOSTART=1 to allow running on startup.
REM ============================================================
if /I NOT "%SO8T_ENABLE_AUTOSTART%"=="1" (
    echo [INFO] Advanced science pipeline autostart is disabled. Exiting.
    exit /b 0
)

echo ========================================
echo Advanced Science Reasoning Pipeline Auto Startup
echo ========================================
echo Started at: %DATE% %TIME%
echo Computer: %COMPUTERNAME%
echo User: %USERNAME%
echo ========================================

REM 作業ディレクトリ設定
cd /d "C:\Users\downl\Desktop\SO8T"

REM Python環境設定
set PYTHONPATH=%PYTHONPATH%;"C:\Users\downl\Desktop\SO8T"
set SO8T_AUTO_RECOVER=true
set SO8T_AUTO_CLEANUP=true
set ADVANCED_SCIENCE_PIPELINE=true

REM ログファイル設定
set LOG_DIR=logs\startup
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM 日付・時刻のフォーマット（安全な形式に）
set TIMESTAMP=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set TIMESTAMP=%TIMESTAMP::=%
set TIMESTAMP=%TIMESTAMP:/=_%
set LOG_FILE=%LOG_DIR%\advanced_science_pipeline_startup_%TIMESTAMP%.log

echo Starting Advanced Science Reasoning Pipeline... >> "%LOG_FILE%"
echo Log file: %LOG_FILE% >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"

REM Python実行ファイルの検索
set PYTHON_EXE=
if exist "C:\Python312\python.exe" (
    set PYTHON_EXE=C:\Python312\python.exe
) else if exist "C:\Python311\python.exe" (
    set PYTHON_EXE=C:\Python311\python.exe
) else if exist "C:\Python310\python.exe" (
    set PYTHON_EXE=C:\Python310\python.exe
) else if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe" (
    set PYTHON_EXE=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe
) else if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe" (
    set PYTHON_EXE=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe
) else (
    REM py launcherを使用
    set PYTHON_EXE=py -3.12
)

echo Using Python: %PYTHON_EXE%

REM チェックポイント復旧チェック
echo [CHECKPOINT] Checking for recovery checkpoints... >> "%LOG_FILE%"
set CHECKPOINT_DIR=checkpoints\advanced_science_reasoning
if exist "%CHECKPOINT_DIR%" (
    echo [CHECKPOINT] Checkpoint directory found >> "%LOG_FILE%"
    echo [CHECKPOINT] Attempting recovery from latest checkpoint >> "%LOG_FILE%"
    set RECOVER_MODE=--recover
) else (
    echo [CHECKPOINT] No checkpoint directory found, starting fresh >> "%LOG_FILE%"
    set RECOVER_MODE=
)

REM 高度科学推論学習パイプライン実行スクリプトの存在確認
if not exist "scripts\training\train_unsloth_so8t.py" (
    echo [ERROR] Training script not found: scripts\training\train_unsloth_so8t.py >> "%LOG_FILE%"
    goto :error
)

REM Python実行（バックグラウンドで実行）
echo [INFO] Starting Advanced Science Reasoning Pipeline... >> "%LOG_FILE%"
if "%RECOVER_MODE%"=="" (
    start "Advanced_Science_Pipeline" /B cmd /c "%PYTHON_EXE% scripts\training\train_unsloth_so8t.py --phase full >> "%LOG_FILE%" 2>&1"
) else (
    REM チェックポイントから復旧
    start "Advanced_Science_Pipeline" /B cmd /c "%PYTHON_EXE% scripts\training\train_unsloth_so8t.py --phase full --recover >> "%LOG_FILE%" 2>&1"
)

REM プロセス起動確認
timeout /t 5 /nobreak > nul
tasklist /FI "IMAGENAME eq python.exe" 2>nul | findstr "python.exe" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Advanced Science Reasoning Pipeline started successfully >> "%LOG_FILE%"
    echo Process started in background >> "%LOG_FILE%"
) else (
    echo [ERROR] Failed to start Advanced Science Reasoning Pipeline >> "%LOG_FILE%"
    goto :error
)

REM GPU利用可能確認
nvidia-smi --query-gpu=name --format=csv,noheader,nounits > nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] GPU detected and available >> "%LOG_FILE%"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "%LOG_FILE%"
) else (
    echo [WARNING] GPU not detected or not available >> "%LOG_FILE%"
)

echo ======================================== >> "%LOG_FILE%"
echo Advanced Science Reasoning Pipeline Startup completed successfully >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"

REM 音声通知（オプション）
if exist "scripts\utils\play_audio_notification.ps1" (
    powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1" 2>nul
)

goto :end

:error
echo ======================================== >> "%LOG_FILE%"
echo Advanced Science Reasoning Pipeline Startup FAILED >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"

REM エラー音声通知
powershell -ExecutionPolicy Bypass -Command "[System.Console]::Beep(800, 1000)" 2>nul

:end
echo Startup script completed at: %DATE% %TIME%
