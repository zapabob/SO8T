@echo off
REM SO(8)T Power-On Startup Script
REM 電源投入時にタスクスケジューラから自動起動されるスクリプト

REM ============================================================
REM AUTOSTART KILL-SWITCH (default: disabled)
REM Set SO8T_ENABLE_AUTOSTART=1 to allow running on startup.
REM ============================================================
if /I NOT "%SO8T_ENABLE_AUTOSTART%"=="1" (
    echo [INFO] SO8T power-on autostart is disabled. Exiting.
    exit /b 0
)

echo ========================================
echo SO(8)T Power-On Auto Startup
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

REM ログファイル設定
set LOG_DIR=logs\startup
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM 日付・時刻のフォーマット（安全な形式に）
set TIMESTAMP=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set TIMESTAMP=%TIMESTAMP::=%
set TIMESTAMP=%TIMESTAMP:/=_%
set LOG_FILE=%LOG_DIR%\so8t_startup_%TIMESTAMP%.log

echo Starting SO(8)T Automated Pipeline... >> "%LOG_FILE%"
echo Log file: %LOG_FILE% >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"

REM Python実行（バックグラウンドで実行）
echo Starting Python process...

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
    set PYTHON_EXE=py -3
)

echo Using Python: %PYTHON_EXE%

REM Pythonスクリプトの存在確認
if not exist "scripts\automation\automatic_aegis_phi35_thinking_pipeline.py" (
    echo [ERROR] Python script not found: scripts\automation\automatic_aegis_phi35_thinking_pipeline.py >> "%LOG_FILE%"
    goto :error
)

REM Python実行（バックグラウンドで実行）
echo [INFO] Starting SO8T automated pipeline... >> "%LOG_FILE%"
start "SO8T_Automated_Pipeline" /B cmd /c "%PYTHON_EXE% scripts\automation\automatic_aegis_phi35_thinking_pipeline.py --resume >> "%LOG_FILE%" 2>&1"

REM プロセス起動確認
timeout /t 5 /nobreak > nul
tasklist /FI "IMAGENAME eq python.exe" 2>nul | findstr "python.exe" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] SO(8)T pipeline started successfully >> "%LOG_FILE%"
    echo Process started in background >> "%LOG_FILE%"
) else (
    echo [ERROR] Failed to start SO(8)T pipeline >> "%LOG_FILE%"
    echo [DEBUG] Checking if script exists... >> "%LOG_FILE%"
    if exist "scripts\automation\automatic_aegis_phi35_thinking_pipeline.py" (
        echo [DEBUG] Script exists >> "%LOG_FILE%"
    ) else (
        echo [DEBUG] Script NOT found >> "%LOG_FILE%"
    )
    goto :error
)

REM GPU利用可能確認
nvidia-smi --query-gpu=name --format=csv,noheader,nounits > nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] GPU detected and available >> "%LOG_FILE%"
) else (
    echo [WARNING] GPU not detected or not available >> "%LOG_FILE%"
)

echo ======================================== >> "%LOG_FILE%"
echo SO(8)T Power-On Startup completed successfully >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"

REM 音声通知（オプション）
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1" 2>nul

goto :end

:error
echo ======================================== >> "%LOG_FILE%"
echo SO(8)T Power-On Startup FAILED >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"

REM エラー音声通知
powershell -ExecutionPolicy Bypass -Command "[System.Console]::Beep(800, 1000)" 2>nul

:end
echo Startup script completed at: %DATE% %TIME%
