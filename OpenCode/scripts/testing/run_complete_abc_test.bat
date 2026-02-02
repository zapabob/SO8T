@echo off
chcp 65001 >nul
echo [ABC TEST] Starting Complete ABC Test with Statistical Analysis
echo ======================================================

echo [STEP 1] Converting Borea-Phi3.5-instruct-jp to GGUF format...
echo [ABC TEST] Creating Model A (GGUF version)...
python scripts/conversion/convert_borea_phi35_to_gguf.py --create_config --verify

if %errorlevel% neq 0 (
    echo [ERROR] GGUF conversion failed!
    goto :error
)

echo [OK] Model A (GGUF) conversion completed!

echo [STEP 2] Running comprehensive ABC benchmark evaluation...
echo [ABC TEST] Evaluating Model A, Model B, and Model C...
python scripts/evaluation/comprehensive_llm_benchmark.py --abc_test

if %errorlevel% neq 0 (
    echo [ERROR] ABC benchmark evaluation failed!
    goto :error
)

echo [OK] ABC benchmark evaluation completed!

echo [STEP 3] Generating HF submission statistics with error bars...
echo [ABC TEST] Creating statistical analysis plots and tables...
python scripts/evaluation/hf_submission_statistics.py --results_file "D:/webdataset/results/abc_test_results/abc_test_results.json"

if %errorlevel% neq 0 (
    echo [ERROR] HF submission statistics generation failed!
    goto :error
)

echo [OK] HF submission statistics generated!

echo [STEP 4] ABC Test Results Summary
echo ==================================
echo [ABC TEST] Winner determination and statistical significance analysis completed!
echo [ABC TEST] Check results in D:/webdataset/results/abc_test_results/
echo [ABC TEST] HF submission files in D:/webdataset/results/hf_submission/

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

echo [SUCCESS] Complete ABC test with statistical analysis finished!
goto :end

:error
echo [ERROR] ABC test failed with error code %errorlevel%
echo [AUDIO] Playing error notification...
powershell -ExecutionPolicy Bypass -Command "[System.Console]::Beep(800, 1000)"
exit /b 1

:end
echo [COMPLETED] ABC Test Workflow Complete
pause
