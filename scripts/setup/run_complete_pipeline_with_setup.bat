@echo off
chcp 65001 >nul
echo [COMPLETE PIPELINE] Starting SO8T Complete Pipeline with Environment Setup
echo ======================================================================

echo [PHASE 1] Environment Setup and Dependencies
echo =============================================
echo [SETUP] Installing Python libraries, downloading datasets, data cleansing...

call scripts/setup/run_pipeline_setup.bat

if %errorlevel% neq 0 (
    echo [ERROR] Environment setup failed!
    goto :error
)

echo [OK] Environment setup completed successfully!

echo [PHASE 2] ABC Test Execution
echo ============================
echo [ABC TEST] Running comprehensive ABC test evaluation...
echo [ABC TEST] Models: A(Borea-Phi3.5 GGUF), B(AEGIS-Phi3.5-Enhanced), C(AEGIS-Phi3.5-Golden-Sigmoid)

call scripts/testing/run_complete_abc_test.bat

if %errorlevel% neq 0 (
    echo [ERROR] ABC test execution failed!
    goto :error
)

echo [OK] ABC test completed successfully!

echo [PHASE 3] Results Summary
echo ========================
echo [RESULTS] All pipeline phases completed successfully!
echo [RESULTS] Check comprehensive results in D:/webdataset/results/
echo.
echo [FINAL OUTPUTS]
echo ==============
echo 1. ABC Test Results: D:/webdataset/results/abc_test_results/
echo    - Raw benchmark scores for all models
echo    - Statistical comparisons and significance tests
echo    - Winner determination with confidence metrics
echo.
echo 2. HF Submission Files: D:/webdataset/results/hf_submission/
echo    - Error bar plots (PNG)
echo    - Summary statistics (CSV/LaTeX)
echo    - Correlation analysis heatmaps
echo    - README and detailed documentation
echo.
echo 3. Datasets: D:/webdataset/datasets/
echo    - ELYZA-100 and other benchmark datasets
echo    - Preprocessed training data
echo.
echo 4. Models: D:/webdataset/models/ and D:/webdataset/gguf_models/
echo    - Trained model checkpoints
echo    - GGUF quantized versions
echo ==============

echo [AUDIO] Playing final completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

echo [SUCCESS] Complete SO8T pipeline execution finished!
echo [SUCCESS] All phases (Setup + ABC Test + Analysis) completed successfully!
goto :end

:error
echo [ERROR] Complete pipeline failed with error code %errorlevel%
echo [AUDIO] Playing error notification...
powershell -ExecutionPolicy Bypass -Command "[System.Console]::Beep(800, 1000)"
exit /b 1

:end
echo [COMPLETED] Complete SO8T Pipeline Execution Finished
pause
