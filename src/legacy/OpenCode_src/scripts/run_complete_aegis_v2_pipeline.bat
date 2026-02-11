@echo off
REM AEGIS-v2.0 完全自動パイプライン実行スクリプト
REM Complete automated pipeline for AEGIS-v2.0 training

echo ========================================
echo   AEGIS-v2.0 Complete Automated Pipeline
echo ========================================
echo.

REM UTF-8対応
chcp 65001 >nul

REM 現在のディレクトリをプロジェクトルートに設定
cd /d "%~dp0.."

echo [STEP 1] Checking Python environment...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)
echo.

echo [STEP 2] Installing/updating dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo WARNING: Some dependencies might be missing, continuing...
)
echo.

echo [STEP 3] Creating auto-resume script...
python scripts/training/aegis_v2_training_pipeline.py --create_resume_script
if errorlevel 1 (
    echo ERROR: Failed to create auto-resume script!
    pause
    exit /b 1
)
echo.

echo [STEP 4] Starting complete AEGIS-v2.0 pipeline...
echo This will run dataset collection, model setup, and training with auto-checkpointing.
echo The system will automatically save checkpoints every 3 minutes and maintain 5 rolling checkpoints.
echo.
echo Press Ctrl+C to interrupt training (checkpoint will be saved automatically).
echo.

python scripts/training/aegis_v2_training_pipeline.py
if errorlevel 1 (
    echo.
    echo ERROR: AEGIS-v2.0 pipeline failed!
    echo Check the logs for details.
    echo A checkpoint should have been saved automatically.
) else (
    echo.
    echo SUCCESS: AEGIS-v2.0 pipeline completed successfully!
    echo.
    echo Results saved to: D:\webdataset\models\aegis_v2_phi35_thinking\
    echo Checkpoints: D:\webdataset\models\aegis_v2_phi35_thinking\checkpoints\
)

echo.
echo Press any key to exit...
pause >nul
