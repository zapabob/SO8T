@echo off
REM SO(8)T Power-On Startup Script
REM 電源投入時にタスクスケジューラから自動起動されるスクリプト

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
set LOG_FILE=%LOG_DIR%\so8t_startup_%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%.log
set LOG_FILE=%LOG_FILE: =0%

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
start "SO8T_Automated_Pipeline" /B %PYTHON_EXE% so8t_automated_pipeline.py --autostart >> "%LOG_FILE%" 2>&1

REM プロセス起動確認
timeout /t 5 /nobreak > nul
tasklist /FI "IMAGENAME eq python.exe" | find "python.exe" > nul
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] SO(8)T pipeline started successfully >> "%LOG_FILE%"
    echo Process started in background >> "%LOG_FILE%"
) else (
    echo [ERROR] Failed to start SO(8)T pipeline >> "%LOG_FILE%"
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
