@echo off
chcp 65001 >nul
echo [PIPELINE SETUP] Starting SO8T Pipeline Environment Setup
echo ========================================================

echo [STEP 1] Setting up Python environment and libraries...
python scripts/setup/setup_pipeline_environment.py

if %errorlevel% neq 0 (
    echo [ERROR] Pipeline environment setup failed!
    goto :error
)

echo [OK] Pipeline environment setup completed!

echo [STEP 2] Preparing ABC test execution...
echo [INFO] All dependencies, datasets, and preprocessing completed
echo [INFO] Ready to run ABC test evaluation

echo [NEXT STEPS]
echo ===========
echo 1. Run ABC test: scripts/testing/run_complete_abc_test.bat
echo 2. Check results in: D:/webdataset/results/
echo 3. View HF submission files in: D:/webdataset/results/hf_submission/
echo ===========

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

echo [SUCCESS] Pipeline environment setup and ABC test preparation completed!
goto :end

:error
echo [ERROR] Pipeline setup failed with error code %errorlevel%
echo [AUDIO] Playing error notification...
powershell -ExecutionPolicy Bypass -Command "[System.Console]::Beep(800, 1000)"
exit /b 1

:end
echo [COMPLETED] Pipeline Setup Complete
