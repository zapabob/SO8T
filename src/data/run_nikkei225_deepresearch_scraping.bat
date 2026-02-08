@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T 日経225企業DeepResearch Webスクレイピング（10ブラウザ×10タブ=100タブ並列実行）
echo ================================================================================
echo.
echo [INFO] 10ブラウザ×10タブ（合計100タブ）の並列構成でスクレイピングを実行します
echo [INFO] 日経225企業（防衛・航空宇宙・インフラ企業を含む）の公開データを収集します
echo [INFO] データタイプ: 財務報告、プレスリリース、製品情報、日経企業情報
echo [INFO] デーモンではなく開いたブラウザで、人間を模倣した動作でボット検知を回避します
echo.

REM 設定
set "PROJECT_ROOT=%~dp0..\.."
set "OUTPUT_DIR=D:\webdataset\nikkei225_deepresearch"
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
echo [INFO] ブラウザ数: 10
echo [INFO] タブ数（各ブラウザ）: 10
echo [INFO] 総並列タスク数: 100（10×10）
echo [INFO] 対象企業: 日経225全企業（防衛・航空宇宙・インフラ企業を含む）
echo [INFO] データタイプ: 財務報告、プレスリリース、製品情報、日経企業情報
echo.

REM ログディレクトリの作成
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

echo [INFO] 日経225企業DeepResearch Webスクレイピングを開始します
echo [INFO] 10ブラウザ×10タブの並列構成で実行します
echo [INFO] 人間を模倣した動作でボット検知を回避します
echo.

REM スクレイピング実行
"%PYTHON%" scripts\data\parallel_deep_research_scraping.py --output "%OUTPUT_DIR%" --num-browsers 10 --use-cursor-browser --remote-debugging-port 9222 --delay 1.5 --timeout 30000 --max-memory-gb 8.0 --max-cpu-percent 90.0
set "SCRAPE_RESULT=%errorlevel%"

if %SCRAPE_RESULT% equ 0 (
    echo [OK] スクレイピング完了
) else (
    echo [NG] スクレイピング失敗
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] 日経225企業DeepResearch Webスクレイピング完了
echo ================================================================================
echo [INFO] 出力ディレクトリ: %OUTPUT_DIR%
echo [INFO] カテゴリ別ファイル: nikkei225_financial_reports_*.jsonl, nikkei225_press_releases_*.jsonl, nikkei225_product_info_*.jsonl, nikkei225_nikkei_company_info_*.jsonl
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

exit /b 0


