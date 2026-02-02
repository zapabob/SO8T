@echo off
REM SO8T A/Bテスト自動起動スクリプト
REM 電源投入時に自動実行されるようにタスクスケジューラーに登録してください

cd /d "%~dp0\..\.."

REM H:\from_D\webdataset が利用可能か確認
if not exist "H:\from_D\webdataset" (
    echo [ERROR] H:\from_D\webdataset not found >> auto_start.log
    exit /b 1
)

REM ログディレクトリ作成
if not exist "H:\from_D\webdataset\logs" mkdir "H:\from_D\webdataset\logs"

echo [AUTO] Starting SO8T A/B Test at %DATE% %TIME% >> "H:\from_D\webdataset\logs\auto_start.log"

REM Python環境確認
python --version >> "H:\from_D\webdataset\logs\auto_start.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found >> "H:\from_D\webdataset\logs\auto_start.log"
    exit /b 1
)

REM A/Bテスト実行
python scripts/benchmark/aegis_ab_test_benchmark.py >> "H:\from_D\webdataset\logs\auto_start.log" 2>&1

if errorlevel 0 (
    echo [SUCCESS] A/B Test completed at %DATE% %TIME% >> "H:\from_D\webdataset\logs\auto_start.log"
) else (
    echo [ERROR] A/B Test failed at %DATE% %TIME% >> "H:\from_D\webdataset\logs\auto_start.log"
)

REM 完了通知（オプション）
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
