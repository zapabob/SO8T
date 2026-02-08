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
echo.
echo Options:
echo   --skip-data-collection   Skip data collection (data already exists)
echo   --skip-data-processing   Skip data processing (data already processed)
echo   --skip-data-cleansing    Skip data cleansing (use cleansed data)
echo   --resume                 Resume from checkpoint
echo.

set "PYTHON_EXE=py -3"
set "SCRIPT_DIR=%~dp0..\.."
set "LOG_DIR=%SCRIPT_DIR%\logs"
set "CHECKPOINT_DIR=D:\webdataset\checkpoints\abc_pipeline"
set "DATA_DIR=D:\webdataset\data"

mkdir "%LOG_DIR%" 2>nul
mkdir "%CHECKPOINT_DIR%" 2>nul
mkdir "%DATA_DIR%\datasets" 2>nul
mkdir "%DATA_DIR%\datasets\cleansed" 2>nul
mkdir "%DATA_DIR%\datasets\vssi_tagged" 2>nul

echo [START] %date% %time%
echo.

if "%1"=="--resume" (
    echo [RESUME] Resuming from last checkpoint...
    %PYTHON_EXE% "%SCRIPT_DIR%\src\evaluation\abc_pipeline.py" --resume --interval 300
) else if "%1"=="--skip-data-collection" (
    echo [SKIP] Data collection skipped (--skip-data-collection)
    echo [RUN] Starting from data processing phase...
    %PYTHON_EXE% "%SCRIPT_DIR%\src\evaluation\abc_pipeline.py" --skip-data-collection --skip-data-processing --interval 300
) else if "%1"=="--skip-data-cleansing" (
    echo [SKIP] Data collection/processing/cleansing skipped
    echo [RUN] Starting from model benchmarking phase...
    %PYTHON_EXE% "%SCRIPT_DIR%\src\evaluation\abc_pipeline.py" --skip-data-collection --skip-data-processing --skip-data-cleansing --interval 300
) else (
    echo [RUN] Starting fresh pipeline execution (full data collection)...
    %PYTHON_EXE% "%SCRIPT_DIR%\src\evaluation\abc_pipeline.py" --interval 300
)

echo.
echo [COMPLETE] %date% %time%
pause
