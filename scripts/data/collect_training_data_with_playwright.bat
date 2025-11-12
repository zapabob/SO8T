@echo off
REM Playwrightベース学習用データ収集スクリプト（Windows）
REM 日本語ドメイン別知識とコーディング能力向上を狙ったWebスクレイピング

chcp 65001 >nul
setlocal enabledelayedexpansion

echo ==========================================
echo Playwright Training Data Collection
echo 日本語ドメイン別知識 + コーディング能力向上
echo ==========================================
echo.

REM デフォルト設定
set "OUTPUT_DIR=D:/webdataset/training_data_collected"
set "SOURCES=wikipedia_ja,github,stackoverflow,qiita"
set "TARGET_SAMPLES=100000"
set "USE_CURSOR_BROWSER=true"
set "REMOTE_DEBUG_PORT=9222"

REM 引数解析
if not "%~1"=="" set "OUTPUT_DIR=%~1"
if not "%~2"=="" set "SOURCES=%~2"
if not "%~3"=="" set "TARGET_SAMPLES=%~3"

echo [CONFIG] Output directory: !OUTPUT_DIR!
echo [CONFIG] Sources: !SOURCES!
echo [CONFIG] Target samples: !TARGET_SAMPLES!
echo [CONFIG] Use Cursor browser: !USE_CURSOR_BROWSER!
echo.

REM Playwrightインストール確認
echo [CHECK] Checking Playwright installation...
py -3 -c "import playwright" 2>nul
if errorlevel 1 (
    echo [INSTALL] Installing Playwright...
    pip install playwright
    playwright install chromium
    if errorlevel 1 (
        echo [ERROR] Failed to install Playwright
        exit /b 1
    )
)

REM BeautifulSoupインストール確認
echo [CHECK] Checking BeautifulSoup installation...
py -3 -c "import bs4" 2>nul
if errorlevel 1 (
    echo [INSTALL] Installing BeautifulSoup...
    pip install beautifulsoup4 lxml
    if errorlevel 1 (
        echo [ERROR] Failed to install BeautifulSoup
        exit /b 1
    )
)

echo.
echo [START] Starting data collection...
echo.

REM データ収集実行
if "!USE_CURSOR_BROWSER!"=="true" (
    py -3 scripts/data/collect_training_data_with_playwright.py ^
        --output "!OUTPUT_DIR!" ^
        --sources "!SOURCES!" ^
        --target_samples !TARGET_SAMPLES! ^
        --use_cursor_browser ^
        --remote_debugging_port !REMOTE_DEBUG_PORT! ^
        --delay 2.0 ^
        --timeout 30000 ^
        --max_depth 3 ^
        --max_pages_per_source 1000
) else (
    py -3 scripts/data/collect_training_data_with_playwright.py ^
        --output "!OUTPUT_DIR!" ^
        --sources "!SOURCES!" ^
        --target_samples !TARGET_SAMPLES! ^
        --headless ^
        --delay 2.0 ^
        --timeout 30000 ^
        --max_depth 3 ^
        --max_pages_per_source 1000
)

if errorlevel 1 (
    echo.
    echo [ERROR] Data collection failed!
    exit /b 1
)

echo.
echo [OK] Data collection completed successfully!
echo.
echo Output files:
echo   Training data: !OUTPUT_DIR!/training_data_*.jsonl
echo   Statistics: !OUTPUT_DIR!/stats_*.json
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"


