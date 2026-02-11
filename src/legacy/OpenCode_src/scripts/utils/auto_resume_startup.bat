@echo off
REM ========================================
REM SO8T Auto Resume Startup Script
REM 電源オン時に学習とクロールを自動的に再開
REM ========================================

chcp 65001 >nul
setlocal enabledelayedexpansion

REM プロジェクトルートパス
set "PROJECT_ROOT=%~dp0.."
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

REM ログディレクトリ
set "LOGS_DIR=%PROJECT_ROOT%\logs"
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

REM ログファイル
set "LOG_FILE=%LOGS_DIR%\auto_resume_startup.log"
set "TIMESTAMP=%date% %time%"

REM ログ開始
echo [%TIMESTAMP%] ======================================== >> "%LOG_FILE%"
echo [%TIMESTAMP%] SO8T Auto Resume Startup >> "%LOG_FILE%"
echo [%TIMESTAMP%] ======================================== >> "%LOG_FILE%"
echo [%TIMESTAMP%] Project Root: %PROJECT_ROOT% >> "%LOG_FILE%"

REM Python実行可能ファイルの確認
where py >nul 2>&1
if errorlevel 1 (
    echo [%TIMESTAMP%] [ERROR] Python not found in PATH >> "%LOG_FILE%"
    echo [%TIMESTAMP%] [ERROR] Please install Python or add it to PATH >> "%LOG_FILE%"
    exit /b 1
)

REM 自動再開スクリプトのパス
set "AUTO_RESUME_SCRIPT=%PROJECT_ROOT%\scripts\auto_resume.py"

if not exist "%AUTO_RESUME_SCRIPT%" (
    echo [%TIMESTAMP%] [ERROR] Auto resume script not found: %AUTO_RESUME_SCRIPT% >> "%LOG_FILE%"
    exit /b 1
)

REM 自動再開スクリプトを実行
echo [%TIMESTAMP%] [INFO] Executing auto resume script... >> "%LOG_FILE%"
echo [%TIMESTAMP%] [INFO] Command: py -3 "%AUTO_RESUME_SCRIPT%" >> "%LOG_FILE%"

REM Pythonスクリプトを実行（バックグラウンド）
start /MIN "SO8T-AutoResume" py -3 "%AUTO_RESUME_SCRIPT%" >> "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo [%TIMESTAMP%] [ERROR] Failed to start auto resume script >> "%LOG_FILE%"
    exit /b 1
) else (
    echo [%TIMESTAMP%] [OK] Auto resume script started successfully >> "%LOG_FILE%"
)

REM 少し待機してから終了
timeout /t 3 /nobreak >nul

REM 音声通知
if exist "%PROJECT_ROOT%\.cursor\marisa_owattaze.wav" (
    powershell -Command "if (Test-Path '%PROJECT_ROOT%\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('%PROJECT_ROOT%\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }" >> "%LOG_FILE%" 2>&1
)

echo [%TIMESTAMP%] [OK] Auto resume startup completed >> "%LOG_FILE%"
echo [%TIMESTAMP%] ======================================== >> "%LOG_FILE%"

endlocal
exit /b 0

