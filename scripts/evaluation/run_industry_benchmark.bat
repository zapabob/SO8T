@echo off
chcp 65001 >nul
echo [INDUSTRY BENCHMARK] Starting Integrated Industry Standard Benchmark
echo ============================================================

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..
cd /d %PROJECT_ROOT%

echo [STEP 1] Running integrated benchmark suite...
py -3 scripts\evaluation\integrated_industry_benchmark.py --output-dir D:\webdataset\benchmark_results\industry_standard

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Benchmark execution failed
    exit /b %ERRORLEVEL%
)

echo [STEP 2] Finding latest results file...
for /f "delims=" %%i in ('dir /b /o-d D:\webdataset\benchmark_results\industry_standard\integrated_benchmark_results_*.json 2^>nul') do (
    set LATEST_RESULTS=D:\webdataset\benchmark_results\industry_standard\%%i
    goto :found_results
)

:found_results
if not defined LATEST_RESULTS (
    echo [ERROR] No results file found
    exit /b 1
)

echo [STEP 3] Creating visualizations...
py -3 scripts\evaluation\visualize_industry_benchmark.py --results "%LATEST_RESULTS%" --output-dir D:\webdataset\benchmark_results\industry_standard\figures

if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Visualization failed, continuing...
)

echo [STEP 4] Updating README...
py -3 scripts\evaluation\update_readme_benchmarks.py --readme README.md --results "%LATEST_RESULTS%" --figures-dir D:\webdataset\benchmark_results\industry_standard\figures

if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] README update failed, continuing...
)

echo [STEP 5] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

echo [COMPLETE] Industry benchmark suite finished!
echo Results: %LATEST_RESULTS%
echo Figures: D:\webdataset\benchmark_results\industry_standard\figures\

