@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T自動テスト実行システム
echo ================================================================================
echo.

REM 設定
set "PROJECT_ROOT=%~dp0.."
set "LOG_DIR=%PROJECT_ROOT%\_docs\test_logs"
set "CONFIG_FILE=%PROJECT_ROOT%\configs\test_config.json"

REM ログディレクトリの作成
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM タイムスタンプ
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,4%-%dt:~4,2%-%dt:~6,2%_%dt:~8,2%-%dt:~10,2%-%dt:~12,2%"

echo [INFO] 自動テスト開始: %timestamp%
echo [INFO] プロジェクトルート: %PROJECT_ROOT%
echo [INFO] ログディレクトリ: %LOG_DIR%
echo.

REM 設定ファイルの確認
if not exist "%CONFIG_FILE%" (
    echo [WARNING] 設定ファイルが見つかりません: %CONFIG_FILE%
    echo [INFO] デフォルト設定を使用します
    goto :use_default_config
)

echo [INFO] 設定ファイルを読み込み中: %CONFIG_FILE%
goto :load_config

:use_default_config
echo [INFO] デフォルト設定を使用
set "TEST_MODE=comprehensive"
set "ENABLE_QUANTIZATION=true"
set "ENABLE_PYTORCH_COMPARISON=true"
set "ENABLE_SO8_TESTS=true"
set "GENERATE_REPORT=true"
set "SEND_NOTIFICATION=true"
goto :start_tests

:load_config
REM 設定ファイルの読み込み（簡易版）
for /f "usebackq tokens=1,2 delims=:" %%a in ("%CONFIG_FILE%") do (
    set "%%a=%%b"
)
goto :start_tests

:start_tests
echo [PHASE 1] 環境チェック
call :check_environment
if %errorlevel% neq 0 (
    echo [ERROR] 環境チェックに失敗しました
    goto :error_exit
)

echo [PHASE 2] 依存関係チェック
call :check_dependencies
if %errorlevel% neq 0 (
    echo [ERROR] 依存関係チェックに失敗しました
    goto :error_exit
)

echo [PHASE 3] テスト実行
if "%TEST_MODE%"=="comprehensive" (
    call :run_comprehensive_tests
) else if "%TEST_MODE%"=="unit" (
    call :run_unit_tests
) else if "%TEST_MODE%"=="integration" (
    call :run_integration_tests
) else (
    echo [ERROR] 不明なテストモード: %TEST_MODE%
    goto :error_exit
)

if %errorlevel% neq 0 (
    echo [ERROR] テスト実行に失敗しました
    goto :error_exit
)

echo [PHASE 4] レポート生成
if "%GENERATE_REPORT%"=="true" (
    call :generate_report
)

echo [PHASE 5] 通知送信
if "%SEND_NOTIFICATION%"=="true" (
    call :send_notification
)

echo [SUCCESS] 自動テスト完了
goto :success_exit

:check_environment
echo [CHECK] Python環境確認中...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Pythonが見つかりません
    exit /b 1
)

echo [CHECK] PyTorch環境確認中...
python -c "import torch; print('PyTorch version:', torch.__version__)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PyTorchが見つかりません
    exit /b 1
)

echo [CHECK] CUDA環境確認中...
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] CUDA環境の確認に失敗しました
)

echo [OK] 環境チェック完了
exit /b 0

:check_dependencies
echo [CHECK] 依存関係確認中...

REM 必要なパッケージのチェック
python -c "import pytest" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pytestが見つかりません
    exit /b 1
)

python -c "import tqdm" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] tqdmが見つかりません
    exit /b 1
)

python -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] numpyが見つかりません
    exit /b 1
)

echo [OK] 依存関係チェック完了
exit /b 0

:run_comprehensive_tests
echo [TEST] 包括的テスト実行中...
call "%PROJECT_ROOT%\scripts\run_comprehensive_tests.bat"
exit /b %errorlevel%

:run_unit_tests
echo [TEST] ユニットテスト実行中...
python -m pytest tests\test_so8_operations_comprehensive.py -v --tb=short
if %errorlevel% neq 0 exit /b 1

python -m pytest tests\test_pytorch_comparison.py -v --tb=short
if %errorlevel% neq 0 exit /b 1

python -m pytest tests\test_so8t_quantization.py -v --tb=short
if %errorlevel% neq 0 exit /b 1

echo [OK] ユニットテスト完了
exit /b 0

:run_integration_tests
echo [TEST] 統合テスト実行中...
python -m pytest tests\ -k "not test_so8_operations_comprehensive and not test_pytorch_comparison and not test_so8t_quantization" -v --tb=short
exit /b %errorlevel%

:generate_report
echo [REPORT] テストレポート生成中...

REM レポート生成スクリプトの実行
python "%PROJECT_ROOT%\scripts\generate_test_report.py" --timestamp %timestamp% --log-dir "%LOG_DIR%"
if %errorlevel% neq 0 (
    echo [WARNING] レポート生成に失敗しました
)

echo [OK] レポート生成完了
exit /b 0

:send_notification
echo [NOTIFICATION] 通知送信中...

REM 音声通知
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] 音声通知送信完了' -ForegroundColor Green } else { Write-Host '[WARNING] 音声ファイルが見つかりません' -ForegroundColor Yellow }"

REM ログファイルの確認
if exist "%LOG_DIR%\test_summary_%timestamp%.txt" (
    echo [INFO] テスト結果サマリー:
    type "%LOG_DIR%\test_summary_%timestamp%.txt"
)

echo [OK] 通知送信完了
exit /b 0

:success_exit
echo.
echo ================================================================================
echo 自動テスト成功完了
echo ================================================================================
echo 完了時刻: %date% %time%
echo ログディレクトリ: %LOG_DIR%
echo.

REM 成功音声通知
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[SUCCESS] 成功通知再生完了' -ForegroundColor Green }"

exit /b 0

:error_exit
echo.
echo ================================================================================
echo 自動テスト失敗
echo ================================================================================
echo 失敗時刻: %date% %time%
echo ログディレクトリ: %LOG_DIR%
echo.

REM エラー音声通知
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[ERROR] エラー通知再生完了' -ForegroundColor Red }"

exit /b 1
