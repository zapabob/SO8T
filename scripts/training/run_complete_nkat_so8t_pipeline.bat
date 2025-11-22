@echo off
chcp 65001 >nul
echo [NKAT-SO8T] Complete RTX 3060 Optimized Pipeline
echo Target: 12-hour completion from 70-hour baseline
echo Theoretical validation: Alpha Gate phase transition
echo ===================================================

set MODEL_PATH=models/Borea-Phi-3.5-mini-Instruct-Jp
set OUTPUT_DIR=D:/webdataset/checkpoints/nkat_so8t_rtx3060
set TRAIN_DATA=data/splits/train_four_class.jsonl
set VALIDATION_DIR=%OUTPUT_DIR%
set REPORT_DIR=_docs

echo [PHASE 1] Creating optimized training script...
py -3 scripts/training/train_nkat_so8t_adapter_optimized.py --create-script
if errorlevel 1 (
    echo [ERROR] Failed to create training script
    goto :error
)

echo [PHASE 2] Starting NKAT-SO8T training (RTX 3060 optimized)...
echo Expected completion time: ~12 hours
echo Monitoring: Check logs in %OUTPUT_DIR%
echo.

call scripts/training/run_nkat_so8t_rtx3060.bat
if errorlevel 1 (
    echo [ERROR] Training failed
    goto :error
)

echo [PHASE 3] Validating Alpha Gate phase transition...
py -3 scripts/validation/validate_nkat_so8t_phase_transition.py ^
    --checkpoint-dir "%VALIDATION_DIR%" ^
    --output-report "%REPORT_DIR%/nkat_so8t_phase_transition_report.md" ^
    --plot-phase-transition "%REPORT_DIR%/nkat_so8t_phase_transition.png"
if errorlevel 1 (
    echo [WARNING] Validation failed, but training completed
)

echo [PHASE 4] Converting to GGUF format for deployment...
py -3 scripts/conversion/convert_borea_so8t_to_gguf.py ^
    --model-path "%OUTPUT_DIR%/checkpoint-step-final" ^
    --output-path "D:/webdataset/gguf_models/nkat_so8t_borea_phi35" ^
    --quantization Q8_0 Q4_K_M
if errorlevel 1 (
    echo [WARNING] GGUF conversion failed
)

echo [PHASE 5] Running Japanese performance tests...
call scripts/testing/japanese_llm_performance_test.bat ^
    nkat-so8t-borea-phi35:latest ^
    "D:/webdataset/gguf_models/nkat_so8t_borea_phi35/nkat_so8t_borea_phi35_Q8_0.gguf"
if errorlevel 1 (
    echo [WARNING] Performance tests failed
)

echo [SUCCESS] NKAT-SO8T pipeline completed!
echo Results:
echo   - Checkpoints: %OUTPUT_DIR%
echo   - Validation report: %REPORT_DIR%/nkat_so8t_phase_transition_report.md
echo   - Phase transition plot: %REPORT_DIR%/nkat_so8t_phase_transition.png
echo   - GGUF models: D:/webdataset/gguf_models/nkat_so8t_borea_phi35/
echo   - Performance tests: _docs/japanese_llm_performance_test_*.md

goto :success

:error
echo [FAILED] Pipeline failed
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
exit /b 1

:success
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
exit /b 0


