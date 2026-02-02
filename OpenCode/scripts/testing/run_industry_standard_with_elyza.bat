@echo off
chcp 65001 >nul

set SCRIPT_DIR=%~dp0
REM SCRIPT_DIR is scripts\testing\, so go up two levels to project root
cd /d "%SCRIPT_DIR%\..\..\"
set PROJECT_ROOT=%CD%

echo ========================================================================
echo Industry Standard Benchmark + ELYZA-100
echo ========================================================================

set TIMESTAMP=%date:~-4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

echo [TIMESTAMP] %TIMESTAMP%
echo [PROJECT_ROOT] %PROJECT_ROOT%
echo [VERIFY] Checking script path...
if exist "%PROJECT_ROOT%\scripts\evaluation\industry_standard_benchmark.py" (
    echo [OK] Script found
) else (
    echo [ERROR] Script not found: %PROJECT_ROOT%\scripts\evaluation\industry_standard_benchmark.py
    exit /b 1
)
echo.

REM Step 1: Run Industry Standard Benchmark (with ELYZA-100)
echo [STEP 1/1] Running Industry Standard Benchmark + ELYZA-100...
py -3 "%PROJECT_ROOT%\scripts\evaluation\industry_standard_benchmark.py" ^
    --run-root D:\webdataset\benchmark_results\industry_standard ^
    --elyza-models model-a:q8_0 aegis-phi3.5-fixed-0.8:latest ^
    --label %TIMESTAMP%

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Industry Standard Benchmark + ELYZA-100 failed.
    powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] All benchmarks completed.
echo Results saved to: D:\webdataset\benchmark_results\industry_standard\industry_%TIMESTAMP%
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

