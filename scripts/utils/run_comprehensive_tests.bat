@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T包括的テストスイート実行
echo ================================================================================
echo.

REM 環境設定
set PYTHONPATH=%CD%
set CUDA_VISIBLE_DEVICES=0

REM ログディレクトリの作成
if not exist "_docs\test_logs" mkdir "_docs\test_logs"

REM タイムスタンプの取得
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "timestamp=%YYYY%-%MM%-%DD%_%HH%-%Min%-%Sec%"

echo [INFO] テスト開始時刻: %timestamp%
echo.

REM テスト結果ファイルの設定
set "test_results=_docs\test_logs\comprehensive_test_results_%timestamp%.json"
set "test_log=_docs\test_logs\comprehensive_test_log_%timestamp%.log"

REM テスト実行関数
:run_test
set "test_name=%~1"
set "test_file=%~2"
set "test_description=%~3"

echo [TEST] %test_name% 開始...
echo [DESC] %test_description%
echo.

REM テスト実行
python -m pytest %test_file% -v --tb=short --json-report --json-report-file=%test_results% > %test_log% 2>&1

if %errorlevel% equ 0 (
    echo [OK] %test_name% 成功
    echo %test_name%:SUCCESS >> _docs\test_logs\test_summary_%timestamp%.txt
) else (
    echo [NG] %test_name% 失敗
    echo %test_name%:FAILED >> _docs\test_logs\test_summary_%timestamp%.txt
)

echo.
goto :eof

REM メインテスト実行
echo [PHASE 1] SO(8)演算ユニットテスト
call :run_test "SO8_Operations" "tests\test_so8_operations_comprehensive.py" "SO(8)群構造の数学的性質検証"

echo [PHASE 2] PyTorch比較テスト
call :run_test "PyTorch_Comparison" "tests\test_pytorch_comparison.py" "PyTorchモデルとの精度比較"

echo [PHASE 3] 量子化テスト
call :run_test "Quantization" "tests\test_so8t_quantization.py" "SO8Tモデル量子化機能検証"

echo [PHASE 4] 既存テストスイート
call :run_test "Existing_Tests" "tests\" "既存のテストスイート実行"

REM テスト結果の集計
echo [SUMMARY] テスト結果集計中...
echo.

REM 成功/失敗のカウント
set "success_count=0"
set "failed_count=0"

for /f "tokens=2 delims=:" %%a in ('findstr ":SUCCESS" _docs\test_logs\test_summary_%timestamp%.txt 2^>nul') do set /a success_count+=1
for /f "tokens=2 delims=:" %%a in ('findstr ":FAILED" _docs\test_logs\test_summary_%timestamp%.txt 2^>nul') do set /a failed_count+=1

set /a total_count=success_count+failed_count

echo ================================================================================
echo テスト結果サマリー
echo ================================================================================
echo 総テスト数: %total_count%
echo 成功: %success_count%
echo 失敗: %failed_count%
if %total_count% gtr 0 (
    set /a success_rate=success_count*100/total_count
    echo 成功率: %success_rate%%%
) else (
    echo 成功率: 0%%
)
echo.

REM 結果ファイルの作成
echo { > _docs\test_logs\final_results_%timestamp%.json
echo   "timestamp": "%timestamp%", >> _docs\test_logs\final_results_%timestamp%.json
echo   "total_tests": %total_count%, >> _docs\test_logs\final_results_%timestamp%.json
echo   "successful_tests": %success_count%, >> _docs\test_logs\final_results_%timestamp%.json
echo   "failed_tests": %failed_count%, >> _docs\test_logs\final_results_%timestamp%.json
echo   "success_rate": %success_rate%%, >> _docs\test_logs\final_results_%timestamp%.json
echo   "test_log_file": "%test_log%", >> _docs\test_logs\final_results_%timestamp%.json
echo   "test_results_file": "%test_results%" >> _docs\test_logs\final_results_%timestamp%.json
echo } >> _docs\test_logs\final_results_%timestamp%.json

REM 最終結果の表示
if %failed_count% equ 0 (
    echo [SUCCESS] 全てのテストが成功しました！
    set "final_status=SUCCESS"
) else (
    echo [WARNING] %failed_count%個のテストが失敗しました
    set "final_status=WARNING"
)

echo.
echo ================================================================================
echo テスト完了
echo ================================================================================
echo 結果ファイル: _docs\test_logs\final_results_%timestamp%.json
echo ログファイル: %test_log%
echo 最終ステータス: %final_status%
echo.

REM 音声通知の再生
echo [AUDIO] 完了通知を再生中...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] 音声通知再生完了' -ForegroundColor Green } else { Write-Host '[WARNING] 音声ファイルが見つかりません' -ForegroundColor Yellow }"

REM 終了コードの設定
if %failed_count% equ 0 (
    exit /b 0
) else (
    exit /b 1
)
