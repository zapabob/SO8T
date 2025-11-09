@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T Webスクレイピング 中規模テスト実行
echo ================================================================================
echo.

REM 設定
set "PROJECT_ROOT=%~dp0..\.."
set "LOG_DIR=%PROJECT_ROOT%\_docs\test_logs"
set "CONFIG_FILE=%PROJECT_ROOT%\configs\test_medium_scale.yaml"
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

echo [INFO] 中規模テスト開始: %timestamp%
echo [INFO] プロジェクトルート: %PROJECT_ROOT%
echo [INFO] 設定ファイル: %CONFIG_FILE%
echo [INFO] ログディレクトリ: %LOG_DIR%
echo.

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

REM 設定ファイルの確認
if not exist "%CONFIG_FILE%" (
    echo [ERROR] 設定ファイルが見つかりません: %CONFIG_FILE%
    echo [INFO] 設定ファイルを作成してください
    goto :error_exit
)

echo [PHASE 1] 設定ファイル確認
echo [CHECK] 設定ファイル読み込み確認...
%PYTHON% -c "import yaml; yaml.safe_load(open(r'%CONFIG_FILE%', 'r', encoding='utf-8')); print('[OK] 設定ファイル読み込み成功')"
if %errorlevel% neq 0 (
    echo [ERROR] 設定ファイルの読み込みに失敗しました
    goto :error_exit
)

echo.
echo [PHASE 2] 統合パイプラインテスト実行（Phase 1のみ）
echo [TEST] 各データソースから10-50ページ/記事を取得...
echo [INFO] このテストは時間がかかる場合があります
echo.

REM パイプライン実行（Phase 1のみ）
%PYTHON% scripts\pipelines\complete_data_pipeline.py --config "%CONFIG_FILE%" > "%LOG_DIR%\medium_scale_pipeline_test_%timestamp%.log" 2>&1
set "PIPELINE_TEST_RESULT=%errorlevel%"

if %PIPELINE_TEST_RESULT% equ 0 (
    echo [OK] 統合パイプラインテスト成功
) else (
    echo [NG] 統合パイプラインテスト失敗（ログを確認してください）
)

echo.
echo [PHASE 3] データ確認
echo [CHECK] 収集データの確認...
REM データファイルの存在確認（簡易版）
if exist "data\processed\web_crawled\*.jsonl" (
    echo [OK] データファイルが生成されました
) else (
    echo [WARNING] データファイルが見つかりません
)

echo.
echo [PHASE 4] テスト結果サマリー
echo ================================================================================
if %PIPELINE_TEST_RESULT% equ 0 (
    echo [OK] 統合パイプラインテスト: 成功
    echo [SUCCESS] 中規模テストが成功しました
    echo ================================================================================
    goto :success_exit
) else (
    echo [NG] 統合パイプラインテスト: 失敗
    echo [FAILED] 中規模テストが失敗しました
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

















