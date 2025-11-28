@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T Webスクレイピング 本番環境実行
echo ================================================================================
echo.

REM 設定
set "PROJECT_ROOT=%~dp0..\.."
set "CONFIG_FILE=%PROJECT_ROOT%\configs\complete_automated_ab_pipeline.yaml"
set "LOG_DIR=%PROJECT_ROOT%\logs"

REM Pythonパスの確認（複数のパスを試行）
set "PYTHON="
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    set "PYTHON=%PROJECT_ROOT%\venv\Scripts\python.exe"
) else if exist "%PROJECT_ROOT%\venv\bin\python.exe" (
    set "PYTHON=%PROJECT_ROOT%\venv\bin\python.exe"
) else (
    REM py ランチャーを試行
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON=py"
    ) else (
        REM システムのPythonを使用
        where python >nul 2>&1
        if %errorlevel% equ 0 (
            set "PYTHON=python"
        ) else (
            echo [ERROR] Pythonが見つかりません
            echo [INFO] venv\Scripts\python.exe または py コマンドが必要です
            goto :error_exit
        )
    )
)

echo [INFO] Python実行ファイル: %PYTHON%

REM ログディレクトリの作成
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM タイムスタンプ
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,4%-%dt:~4,2%-%dt:~6,2%_%dt:~8,2%-%dt:~10,2%-%dt:~12,2%"

echo [INFO] 本番環境実行開始: %timestamp%
echo [INFO] プロジェクトルート: %PROJECT_ROOT%
echo [INFO] 設定ファイル: %CONFIG_FILE%
echo [INFO] ログディレクトリ: %LOG_DIR%
echo.

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

REM 設定ファイルの確認
if not exist "%CONFIG_FILE%" (
    echo [ERROR] 設定ファイルが見つかりません: %CONFIG_FILE%
    goto :error_exit
)

echo [PHASE 1] 環境確認
echo [CHECK] 本番環境準備確認...
cd /d "%PROJECT_ROOT%"
"%PYTHON%" scripts\utils\check_production_environment.py
if %errorlevel% neq 0 (
    echo [WARNING] 環境確認で警告がありますが、続行します
)

echo.
echo [PHASE 2] 設定ファイル検証
echo [CHECK] 設定ファイル検証...
cd /d "%PROJECT_ROOT%"
"%PYTHON%" scripts\utils\validate_config.py --config "%CONFIG_FILE%"
if %errorlevel% neq 0 (
    echo [ERROR] 設定ファイル検証に失敗しました
    goto :error_exit
)

echo.
echo [PHASE 3] 本番環境実行（段階的スケールアップ）
echo [INFO] Phase 1（webスクレイピング）を実行します
echo [INFO] この処理は長時間かかる場合があります
echo [INFO] チェックポイントが自動的に保存されます
echo.

REM パイプライン実行
cd /d "%PROJECT_ROOT%"
"%PYTHON%" scripts\pipelines\complete_data_pipeline.py --config "%CONFIG_FILE%" > "%LOG_DIR%\production_web_scraping_%timestamp%.log" 2>&1
set "PIPELINE_RESULT=%errorlevel%"

if %PIPELINE_RESULT% equ 0 (
    echo [OK] 本番環境実行成功
    goto :success_exit
) else (
    echo [NG] 本番環境実行失敗（ログを確認してください）
    goto :error_exit
)

:success_exit
echo.
echo ================================================================================
echo [SUCCESS] 本番環境実行完了
echo ================================================================================
echo [INFO] 完了時刻: %date% %time%
echo [INFO] ログファイル: %LOG_DIR%\production_web_scraping_%timestamp%.log
echo [INFO] 出力ディレクトリ: D:\webdataset
echo.

REM 音声通知
powershell -Command "if (Test-Path '%PROJECT_ROOT%\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('%PROJECT_ROOT%\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] 音声通知送信完了' -ForegroundColor Green } else { Write-Host '[WARNING] 音声ファイルが見つかりません' -ForegroundColor Yellow }"

exit /b 0

:error_exit
echo.
echo ================================================================================
echo [ERROR] 本番環境実行中にエラーが発生しました
echo ================================================================================
echo [INFO] ログファイルを確認してください: %LOG_DIR%\production_web_scraping_%timestamp%.log
echo [INFO] チェックポイントから復旧できます
echo.

REM エラー音声通知
powershell -Command "if (Test-Path '%PROJECT_ROOT%\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('%PROJECT_ROOT%\.cursor\marisa_owattaze.wav').Play(); Write-Host '[ERROR] エラー通知送信完了' -ForegroundColor Red }"

exit /b 1

