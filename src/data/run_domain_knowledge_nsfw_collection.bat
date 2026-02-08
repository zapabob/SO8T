@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T ドメイン別知識とNSFW検知用データ収集
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

echo [INFO] ドメイン別知識とNSFW検知用データの収集を開始します
echo [INFO] Cursorブラウザを使用してスクレイピングを実行します
echo.

REM データ収集実行
"%PYTHON%" -3 scripts\data\collect_domain_knowledge_and_nsfw.py --output "%OUTPUT_DIR%" --use-cursor-browser --max-samples-per-site 1000
set "COLLECT_RESULT=%errorlevel%"

if %COLLECT_RESULT% equ 0 (
    echo [OK] データ収集完了
) else (
    echo [NG] データ収集失敗
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] ドメイン別知識とNSFW検知用データ収集完了
echo ================================================================================
echo [INFO] 出力ディレクトリ: %OUTPUT_DIR%
echo [INFO] NSFWデータは検知目的のみで、生成目的ではありません
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

exit /b 0





