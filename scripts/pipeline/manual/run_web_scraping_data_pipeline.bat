@echo off
chcp 65001 >nul
echo ================================================================================
echo Webスクレイピングデータパイプライン処理
echo ================================================================================
echo.
echo [INFO] データクレンジング、ラベル付け、四値分類を実行します
echo.

REM 設定
set "PROJECT_ROOT=%~dp0..\.."
set "INPUT_DIR=D:\webdataset\processed"
set "OUTPUT_DIR=D:\webdataset\cleaned"

REM Pythonパスの確認
set "PYTHON="
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    set "PYTHON=%PROJECT_ROOT%\venv\Scripts\python.exe"
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
echo [INFO] プロジェクトルート: %PROJECT_ROOT%
echo [INFO] 入力ディレクトリ: %INPUT_DIR%
echo [INFO] 出力ディレクトリ: %OUTPUT_DIR%
echo.

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

echo [INFO] パイプライン処理を開始します
echo [INFO] 1. データクレンジング
echo [INFO] 2. ラベル付け
echo [INFO] 3. 四値分類（SO8T四重推論）
echo.

REM パイプライン処理実行（並列処理対応）
"%PYTHON%" -3 scripts\pipelines\web_scraping_data_pipeline.py --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" --use-so8t --num-workers 4 --batch-size 100

if %errorlevel% equ 0 (
    echo [OK] パイプライン処理が完了しました
) else (
    echo [NG] パイプライン処理に失敗しました
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] Webスクレイピングデータパイプライン処理完了
echo ================================================================================
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

exit /b 0

