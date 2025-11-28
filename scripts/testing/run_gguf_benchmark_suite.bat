@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%\..\..\"
set PROJECT_ROOT=%CD%

echo ========================================================================
echo GGUFモデル統合ベンチマークスイート
echo ========================================================================

set TIMESTAMP=%date:~-4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

echo [TIMESTAMP] %TIMESTAMP%
echo [PROJECT_ROOT] %PROJECT_ROOT%
echo.

REM Check if config file exists
if not exist "%PROJECT_ROOT%\configs\gguf_benchmark_models.json" (
    echo [ERROR] Config file not found: configs\gguf_benchmark_models.json
    exit /b 1
)

echo [STEP 1/1] Running GGUF benchmark suite...
py -3 "%PROJECT_ROOT%\scripts\evaluation\gguf_benchmark_suite.py" ^
    --config "%PROJECT_ROOT%\configs\gguf_benchmark_models.json" ^
    --output-root D:\webdataset\benchmark_results\gguf_benchmark

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] GGUF benchmark suite failed.
    powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] GGUF benchmark suite completed.
echo Results saved to: D:\webdataset\benchmark_results\gguf_benchmark
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

endlocal
























