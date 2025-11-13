@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T 統合マスターパイプライン - 本番環境起動
echo ================================================================================
echo.
echo [INFO] DeepResearch Webスクレイピング統合版を起動します
echo [INFO] 10ブラウザ×10タブ（合計100タブ）で並列実行
echo [INFO] CUDA処理統合、Chrome完全偽装、人間模倣動作を有効化
echo.

REM プロジェクトルートに移動
cd /d "%~dp0..\.."

REM Pythonパスの確認
set "PYTHON="
if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
) else (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON=py"
    ) else (
        echo [ERROR] Pythonが見つかりません
        exit /b 1
    )
)

echo [INFO] Python実行ファイル: %PYTHON%
echo [INFO] プロジェクトルート: %CD%
echo.

REM ログディレクトリの作成
if not exist "logs" mkdir "logs"

REM 設定ファイルの確認
if not exist "configs\unified_master_pipeline_config.yaml" (
    echo [ERROR] 設定ファイルが見つかりません: configs\unified_master_pipeline_config.yaml
    exit /b 1
)

if not exist "configs\so8t_chromedev_daemon_config.yaml" (
    echo [ERROR] 設定ファイルが見つかりません: configs\so8t_chromedev_daemon_config.yaml
    exit /b 1
)

echo [INFO] 設定ファイルを確認しました
echo.

REM 環境変数の確認
echo [INFO] 環境変数を確認中...
if not exist ".env" (
    echo [WARNING] .envファイルが見つかりません。example.envをコピーしてください
    echo [WARNING] コマンド: copy example.env .env
)

echo.
echo [INFO] 統合マスターパイプラインを起動します
echo [INFO] Phase 1: SO8T統制ChromeDev並列ブラウザCUDA分散処理スクレイピング
echo [INFO] Phase 2: SO8T全自動データ処理
echo [INFO] Phase 3: SO8T完全統合A/Bテスト
echo [INFO] Phase 4-11: その他のフェーズ
echo.

REM パイプライン実行
"%PYTHON%" -3 scripts\pipelines\unified_master_pipeline.py --config configs\unified_master_pipeline_config.yaml --resume
set "PIPELINE_RESULT=%errorlevel%"

if %PIPELINE_RESULT% equ 0 (
    echo.
    echo [OK] パイプラインが正常に完了しました
    echo [AUDIO] 完了通知を再生します...
    powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
) else (
    echo.
    echo [ERROR] パイプラインがエラーで終了しました（終了コード: %PIPELINE_RESULT%）
    echo [AUDIO] 完了通知を再生します...
    powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
    exit /b %PIPELINE_RESULT%
)

echo.
echo ================================================================================
echo パイプライン実行完了
echo ================================================================================
pause





























