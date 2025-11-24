@echo off
chcp 65001 >nul
echo [VISUALIZATION] Starting benchmark visualization...
echo ====================================================

echo [STEP 1] Running visualization script...
py -3 scripts\analysis\visualize_all_benchmark_results.py

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Visualization failed!
    echo [AUDIO] Playing completion notification...
    powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
    exit /b 1
)

echo [STEP 2] Visualization completed successfully!
echo [INFO] Results saved to: _docs\benchmark_results\visualizations\

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

echo [OK] All visualization tasks completed!

