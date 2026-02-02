@echo off
REM SO(8)T Startup Test Script
REM タスクスケジューラ設定をテストするためのスクリプト

echo ========================================
echo SO(8)T Startup Test
echo ========================================
echo Testing startup configuration...
echo ========================================

REM 作業ディレクトリ設定
cd /d "C:\Users\downl\Desktop\SO8T"

REM タスク存在確認
echo [TEST 1] Checking if SO8T task exists...
schtasks /query /tn "SO8T_Power_On_Startup" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [PASS] SO8T task found in Task Scheduler
    schtasks /query /tn "SO8T_Power_On_Startup" | findstr "Ready Enabled"
    if %ERRORLEVEL% EQU 0 (
        echo [PASS] Task is ready and enabled
    ) else (
        echo [FAIL] Task exists but may be disabled
    )
) else (
    echo [FAIL] SO8T task not found in Task Scheduler
    goto :setup_needed
)

REM スタートアップスクリプト存在確認
echo.
echo [TEST 2] Checking startup script...
if exist "scripts\startup\so8t_power_on_startup.bat" (
    echo [PASS] Startup script exists
) else (
    echo [FAIL] Startup script not found
)

REM Python環境確認
echo.
echo [TEST 3] Checking Python environment...

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
%PYTHON_EXE% --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Python is available
    %PYTHON_EXE% -c "import torch; print(f'[INFO] PyTorch version: {torch.__version__}')" 2>nul
) else (
    echo [FAIL] Python not found or not in PATH
)

REM GPU確認
echo.
echo [TEST 4] Checking GPU availability...
nvidia-smi --query-gpu=name --format=csv,noheader,nounits >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [PASS] GPU detected
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
) else (
    echo [WARNING] GPU not detected or nvidia-smi not available
)

REM ログディレクトリ確認
echo.
echo [TEST 5] Checking log directories...
if exist "logs" (
    echo [PASS] Logs directory exists
    if exist "logs\startup" (
        echo [PASS] Startup logs directory exists
        dir /b "logs\startup" 2>nul | findstr "." >nul
        if %ERRORLEVEL% EQU 0 (
            echo [INFO] Existing startup logs found:
            dir /b "logs\startup"
        ) else (
            echo [INFO] No startup logs found yet
        )
    ) else (
        echo [FAIL] Startup logs directory missing
    )
) else (
    echo [FAIL] Logs directory missing
)

REM テスト完了
echo.
echo ========================================
echo Startup Test Complete!
echo ========================================
goto :end

:setup_needed
echo.
echo ========================================
echo SETUP REQUIRED!
echo ========================================
echo.
echo The SO8T task is not configured in Task Scheduler.
echo.
echo To set up automatic startup:
echo 1. Run setup_task_scheduler.bat as Administrator
echo 2. Re-run this test script
echo.
echo Manual setup alternative:
echo 1. Open Task Scheduler (taskschd.msc)
echo 2. Create new task: SO8T_Power_On_Startup
echo 3. Set trigger: At logon
echo 4. Set action: Run so8t_power_on_startup.bat
echo.
goto :end

:end
echo.
echo Press any key to continue...
pause >nul
