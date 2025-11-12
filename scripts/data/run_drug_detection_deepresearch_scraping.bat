@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T 違法薬物検知目的DeepResearch Webスクレイピング（10ブラウザ×10タブ=100タブ並列実行）
echo ================================================================================
echo.
echo [重要] この実装は検知目的のみで、生成目的ではありません
echo [重要] 安全判定と拒否挙動の学習を目的とします
echo [INFO] 10ブラウザ×10タブ（合計100タブ）の並列構成でスクレイピングを実行します
echo [INFO] 違法薬物検知、ドメイン別知識、コーディング能力、NSFW検知データを収集します
echo.

REM 設定
set "PROJECT_ROOT=%~dp0..\.."
set "OUTPUT_DIR=D:\webdataset\drug_detection_deepresearch"
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
echo [INFO] 検知目的: 安全判定と拒否挙動の学習（生成目的ではない）
echo.

REM ログディレクトリの作成
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

echo [INFO] 違法薬物検知目的DeepResearch Webスクレイピングを開始します
echo [INFO] 10ブラウザ×10タブの並列構成で実行します
echo [INFO] データソース: PMDA、FDA、e-Gov、WHO、UNODC、EMCDDA、Wikipedia、技術ドキュメントサイト、コーディング教育サイト
echo [INFO] キーワードカテゴリ: 違法薬物、ドメイン別知識、コーディング能力、NSFW検知（検知目的のみ）
echo.

REM スクレイピング実行
"%PYTHON%" scripts\data\drug_detection_deepresearch_scraping.py --output "%OUTPUT_DIR%" --num-browsers 10 --num-tabs 10 --use-cursor-browser --remote-debugging-port 9222 --delay 1.5 --timeout 30000 --max-memory-gb 8.0 --max-cpu-percent 80.0
set "SCRAPE_RESULT=%errorlevel%"

if %SCRAPE_RESULT% equ 0 (
    echo [OK] スクレイピング完了
) else (
    echo [NG] スクレイピング失敗
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] 違法薬物検知目的DeepResearch Webスクレイピング完了
echo ================================================================================
echo [INFO] 出力ディレクトリ: %OUTPUT_DIR%
echo [INFO] 検知目的: 安全判定と拒否挙動の学習（生成目的ではない）
echo [INFO] NSFWデータは検知目的のみで、生成目的ではありません
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

exit /b 0


