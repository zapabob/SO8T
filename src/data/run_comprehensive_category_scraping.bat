@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T 包括的カテゴリWebスクレイピング（日本語・英語・NSFW・Arxiv含む）
echo ================================================================================
echo.
echo [注意] NSFWデータは検知目的のみで、生成目的ではありません
echo.

REM 設定
set "PROJECT_ROOT=%~dp0..\.."
set "OUTPUT_DIR=D:\webdataset\processed"
set "LOG_DIR=%PROJECT_ROOT%\logs"

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
echo [INFO] 出力ディレクトリ: %OUTPUT_DIR%
echo.

REM ログディレクトリの作成
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

echo [INFO] 包括的カテゴリWebスクレイピングを開始します
echo [INFO] 日本語・英語の広範なカテゴリ（NSFW含む）とArxivを含みます
echo [INFO] Cursorブラウザを使用して、人間を模倣した動作でスクレイピングを実行します
echo.

REM スクレイピング実行
"%PYTHON%" -3 scripts\data\comprehensive_category_scraping.py --output "%OUTPUT_DIR%" --use-cursor-browser --max-pages-per-category 50 --include-nsfw --include-arxiv
set "SCRAPE_RESULT=%errorlevel%"

if %SCRAPE_RESULT% equ 0 (
    echo [OK] スクレイピング完了
) else (
    echo [NG] スクレイピング失敗
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] 包括的カテゴリWebスクレイピング完了
echo ================================================================================
echo [INFO] 出力ディレクトリ: %OUTPUT_DIR%
echo [INFO] NSFWデータは検知目的のみで、生成目的ではありません
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

exit /b 0





