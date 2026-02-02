@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T Webスクレイピング 小規模テスト実行
echo ================================================================================
echo.

REM 設定
set "PROJECT_ROOT=%~dp0..\.."
set "LOG_DIR=%PROJECT_ROOT%\_docs\test_logs"
set "PYTHON=%PROJECT_ROOT%\venv\Scripts\python.exe"

REM Pythonパスの確認
if not exist "%PYTHON%" (
    set "PYTHON=python"
)

REM ログディレクトリの作成
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM タイムスタンプ
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,4%-%dt:~4,2%-%dt:~6,2%_%dt:~8,2%-%dt:~10,2%-%dt:~12,2%"

echo [INFO] 小規模テスト開始: %timestamp%
echo [INFO] プロジェクトルート: %PROJECT_ROOT%
echo [INFO] ログディレクトリ: %LOG_DIR%
echo.

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

echo [PHASE 1] 環境チェック
echo [CHECK] Python環境確認...
%PYTHON% --version
if %errorlevel% neq 0 (
    echo [ERROR] Pythonが見つかりません
    goto :error_exit
)

echo [CHECK] 依存ライブラリ確認...
%PYTHON% -c "import requests; import bs4; import yaml; print('[OK] 依存ライブラリ確認完了')"
if %errorlevel% neq 0 (
    echo [WARNING] 一部の依存ライブラリが見つかりません
)

echo.
echo [PHASE 2] 単体テスト実行
echo [TEST] robots.txt遵守テスト、レート制限テスト、専用クローラーテスト...
%PYTHON% scripts\testing\test_production_web_scraping.py --unit > "%LOG_DIR%\small_scale_unit_test_%timestamp%.log" 2>&1
set "UNIT_TEST_RESULT=%errorlevel%"

if %UNIT_TEST_RESULT% equ 0 (
    echo [OK] 単体テスト成功
) else (
    echo [NG] 単体テスト失敗（ログを確認してください）
)

echo.
echo [PHASE 3] 統合テスト実行（実際のクロールは最小限）
echo [TEST] データソース設定確認、統合動作確認...
%PYTHON% scripts\testing\test_production_web_scraping.py --integration > "%LOG_DIR%\small_scale_integration_test_%timestamp%.log" 2>&1
set "INTEGRATION_TEST_RESULT=%errorlevel%"

if %INTEGRATION_TEST_RESULT% equ 0 (
    echo [OK] 統合テスト成功
) else (
    echo [NG] 統合テスト失敗（ログを確認してください）
)

echo.
echo [PHASE 4] テスト結果サマリー
echo ================================================================================
if %UNIT_TEST_RESULT% equ 0 (
    echo [OK] 単体テスト: 成功
) else (
    echo [NG] 単体テスト: 失敗
)

if %INTEGRATION_TEST_RESULT% equ 0 (
    echo [OK] 統合テスト: 成功
) else (
    echo [NG] 統合テスト: 失敗
)

if %UNIT_TEST_RESULT% equ 0 if %INTEGRATION_TEST_RESULT% equ 0 (
    echo.
    echo [SUCCESS] すべてのテストが成功しました
    echo ================================================================================
    goto :success_exit
) else (
    echo.
    echo [FAILED] 一部のテストが失敗しました
    echo [INFO] ログファイルを確認してください: %LOG_DIR%
    echo ================================================================================
    goto :error_exit
)

:success_exit
echo.
echo [INFO] テスト完了時刻: %date% %time%
echo [INFO] ログディレクトリ: %LOG_DIR%
echo.

REM 音声通知
powershell -Command "if (Test-Path '%PROJECT_ROOT%\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('%PROJECT_ROOT%\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] 音声通知送信完了' -ForegroundColor Green } else { Write-Host '[WARNING] 音声ファイルが見つかりません' -ForegroundColor Yellow }"

exit /b 0

:error_exit
echo.
echo [ERROR] テスト実行中にエラーが発生しました
echo [INFO] ログファイルを確認してください: %LOG_DIR%
echo.

REM エラー音声通知
powershell -Command "if (Test-Path '%PROJECT_ROOT%\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('%PROJECT_ROOT%\.cursor\marisa_owattaze.wav').Play(); Write-Host '[ERROR] エラー通知送信完了' -ForegroundColor Red }"

exit /b 1

















