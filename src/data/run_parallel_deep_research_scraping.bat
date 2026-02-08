@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T 並列DeepResearch Webスクレイピング（10個のブラウザ並列実行）
echo ================================================================================
echo.
echo [注意] NSFWデータは検知目的のみで、生成目的ではありません
echo [INFO] 動的リソース管理を行い、10個のブラウザを並列で起動してスクレイピングを実行します
echo [INFO] 人間を模倣した関連ワード検索やボタン操作を行います
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
echo [INFO] 並列ブラウザ数: 10
echo.

REM ログディレクトリの作成
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

echo [INFO] 並列DeepResearch Webスクレイピングを開始します
echo [INFO] 動的リソース管理を行い、10個のブラウザを並列で起動します
echo [INFO] すべて異なるキーワード検索を人間を模倣した動作で実行します
echo.

REM Gemini APIキーの設定（オプション）
REM 環境変数GEMINI_API_KEYが設定されていない場合は、以下の行のコメントを外して設定してください
REM set "GEMINI_API_KEY=your_gemini_api_key_here"

REM スクレイピング実行
REM Geminiを使用する場合は --use-gemini オプションを追加してください
REM 例: "%PYTHON%" -3 scripts\data\parallel_deep_research_scraping.py --output "%OUTPUT_DIR%" --num-browsers 10 --use-cursor-browser --max-memory-gb 8.0 --max-cpu-percent 80.0 --use-gemini --gemini-model gemini-2.0-flash-exp
"%PYTHON%" -3 scripts\data\parallel_deep_research_scraping.py --output "%OUTPUT_DIR%" --num-browsers 10 --use-cursor-browser --max-memory-gb 8.0 --max-cpu-percent 80.0
set "SCRAPE_RESULT=%errorlevel%"

if %SCRAPE_RESULT% equ 0 (
    echo [OK] スクレイピング完了
) else (
    echo [NG] スクレイピング失敗
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] 並列DeepResearch Webスクレイピング完了
echo ================================================================================
echo [INFO] 出力ディレクトリ: %OUTPUT_DIR%
echo [INFO] NSFWデータは検知目的のみで、生成目的ではありません
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

exit /b 0





