@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T 半自動Webスクレイピング（Cursorブラウザ使用）
echo ================================================================================
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

echo [INFO] Cursorブラウザに接続してスクレイピングを開始します
echo [INFO] ブラウザでページが表示されたら、内容を確認してEnterキーを押してください
echo [INFO] 'skip'と入力すると、そのページをスキップします
echo.

REM デフォルトのURLリスト（引数で上書き可能）
if "%1"=="" (
    set "URLS=https://ja.wikipedia.org/wiki/メインページ https://www.e-gov.go.jp/"
) else (
    set "URLS=%*"
)

echo [SCRAPE] スクレイピング対象URL:
for %%u in (%URLS%) do echo   - %%u
echo.

REM スクレイピング実行
"%PYTHON%" -3 scripts\data\semi_auto_scraping_with_cursor_browser.py --urls %URLS% --output "%OUTPUT_DIR%"
set "SCRAPE_RESULT=%errorlevel%"

if %SCRAPE_RESULT% equ 0 (
    echo [OK] スクレイピング完了
) else (
    echo [NG] スクレイピング失敗
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] 半自動スクレイピング完了
echo ================================================================================

REM 音声通知
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

exit /b 0





