@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo [ABC PIPELINE] A/B/C Complete Model Comparison Pipeline
echo ============================================================
echo Models:
echo   A: microsoft-phi3.5mini-instinct
echo   B: AXCEPT-Borea-phi3.5mini-jp
echo   C: zapabobouj-AEGIS-phi3.5-jp_v4.0 (pipeline output)
echo ============================================================

set "PYTHON_EXE=py -3"
set "SCRIPT_DIR=%~dp0..\.."
set "LOG_DIR=%SCRIPT_DIR%\logs"
set "CHECKPOINT_DIR=D:\webdataset\checkpoints\abc_pipeline"

mkdir "%LOG_DIR%" 2>nul
mkdir "%CHECKPOINT_DIR%" 2>nul

echo [START] %date% %time%
echo.

if "%1"=="resume" (
    echo [RESUME] Resuming from last checkpoint...
    %PYTHON_EXE% "%SCRIPT_DIR%\src\evaluation\abc_pipeline.py" --resume --interval 300
) else (
    echo [RUN] Starting fresh pipeline execution...
    %PYTHON_EXE% "%SCRIPT_DIR%\src\evaluation\abc_pipeline.py" --interval 300
)

echo.
echo [COMPLETE] %date% %time%
pause
