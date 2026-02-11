@echo off
chcp 65001 >nul
echo ========================================================================
echo Industry Standard + AGI ABC Test
echo ========================================================================
echo.

REM タイムスタンプ取得
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set datetime=%%i
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%

echo [TIMESTAMP] %TIMESTAMP%
echo.

REM 結果ディレクトリ設定
set RESULTS_DIR=D:\webdataset\benchmark_results\industry_standard_agi\run_%TIMESTAMP%

echo [STEP 1/4] Running Industry Standard + AGI ABC Test...
echo Results will be saved to: %RESULTS_DIR%
echo.

py -3 scripts\evaluation\industry_standard_agi_abc_test.py ^
    --output-root D:\webdataset\benchmark_results\industry_standard_agi ^
    --models modela aegis_adjusted

if errorlevel 1 (
    echo [ERROR] ABC test failed
    goto :error_exit
)

echo.
echo [STEP 2/4] Performing statistical analysis...
echo.

REM 最新の実行ディレクトリを検索
for /f "delims=" %%d in ('dir /b /ad /o-d "D:\webdataset\benchmark_results\industry_standard_agi\run_*" 2^>nul') do (
    set LATEST_RUN=%%d
    goto :found_latest
)
:found_latest

set LATEST_RUN_DIR=D:\webdataset\benchmark_results\industry_standard_agi\%LATEST_RUN%

py -3 scripts\analysis\analyze_industry_standard_agi_results.py ^
    --results-dir "%LATEST_RUN_DIR%" ^
    --output-dir "%LATEST_RUN_DIR%\analysis"

if errorlevel 1 (
    echo [ERROR] Statistical analysis failed
    goto :error_exit
)

echo.
echo [STEP 3/4] Generating markdown report...
echo.

py -3 scripts\evaluation\generate_agi_abc_report.py ^
    --results-dir "%LATEST_RUN_DIR%" ^
    --analysis-dir "%LATEST_RUN_DIR%\analysis"

if errorlevel 1 (
    echo [ERROR] Report generation failed
    goto :error_exit
)

echo.
echo [STEP 4/4] Summary
echo ========================================================================
echo Results directory: %LATEST_RUN_DIR%
echo Analysis directory: %LATEST_RUN_DIR%\analysis
echo Graphs directory: %LATEST_RUN_DIR%\analysis\graphs
echo Report: _docs\benchmark_results\industry_standard_agi\report_*.md
echo ========================================================================
echo.

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

echo.
echo [COMPLETE] Industry Standard + AGI ABC Test finished successfully!
exit /b 0

:error_exit
echo.
echo [ERROR] Test execution failed. Check logs above for details.
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
exit /b 1




