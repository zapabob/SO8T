@echo off
chcp 65001 >nul
setlocal

:: ドメイン別知識サイトPlaywrightスクレイピング（SO8T統合版）実行スクリプト

:: --- 設定 ---
set "OUTPUT_DIR=D:/webdataset/domain_knowledge_collected"
set "DOMAINS=defense,aerospace,transport,general,nsfw_detection,drug_detection"
set "SO8T_MODEL_PATH=models/so8t_thinking"
set "USE_CURSOR_BROWSER=1"
set "REMOTE_DEBUGGING_PORT=9222"
set "DELAY=2.0"
set "TIMEOUT=30000"
set "MAX_PAGES_PER_DOMAIN=100"
set "MAX_DEPTH=3"
set "QUALITY_THRESHOLD=0.7"

:: 引数処理
if not "%1"=="" set "OUTPUT_DIR=%~1"
if not "%2"=="" set "DOMAINS=%~2"
if not "%3"=="" set "SO8T_MODEL_PATH=%~3"

echo ========================================
echo Domain Knowledge Collection with Playwright (SO8T Integrated)
echo ========================================
echo Output Directory: %OUTPUT_DIR%
echo Domains: %DOMAINS%
echo SO8T Model Path: %SO8T_MODEL_PATH%
echo Use Cursor Browser: %USE_CURSOR_BROWSER%
echo Remote Debugging Port: %REMOTE_DEBUGGING_PORT%
echo Delay: %DELAY% seconds
echo Timeout: %TIMEOUT% ms
echo Max Pages per Domain: %MAX_PAGES_PER_DOMAIN%
echo Max Depth: %MAX_DEPTH%
echo Quality Threshold: %QUALITY_THRESHOLD%
echo ========================================
echo.

:: Pythonスクリプト実行
py -3 scripts/data/collect_domain_knowledge_with_playwright.py ^
    --output "%OUTPUT_DIR%" ^
    --domains "%DOMAINS%" ^
    --so8t_model_path "%SO8T_MODEL_PATH%" ^
    --delay %DELAY% ^
    --timeout %TIMEOUT% ^
    --max_pages_per_domain %MAX_PAGES_PER_DOMAIN% ^
    --max_depth %MAX_DEPTH% ^
    --quality_threshold %QUALITY_THRESHOLD%

if %errorlevel% neq 0 (
    echo [ERROR] Script execution failed with error code %errorlevel%
    exit /b %errorlevel%
)

if "%USE_CURSOR_BROWSER%"=="1" (
    echo [INFO] Using Cursor browser (CDP connection on port %REMOTE_DEBUGGING_PORT%)
    py -3 scripts/data/collect_domain_knowledge_with_playwright.py ^
        --output "%OUTPUT_DIR%" ^
        --domains "%DOMAINS%" ^
        --so8t_model_path "%SO8T_MODEL_PATH%" ^
        --use_cursor_browser ^
        --remote_debugging_port %REMOTE_DEBUGGING_PORT% ^
        --delay %DELAY% ^
        --timeout %TIMEOUT% ^
        --max_pages_per_domain %MAX_PAGES_PER_DOMAIN% ^
        --max_depth %MAX_DEPTH% ^
        --quality_threshold %QUALITY_THRESHOLD%
    
    if %errorlevel% neq 0 (
        echo [ERROR] Script execution with Cursor browser failed with error code %errorlevel%
        exit /b %errorlevel%
    )
)

echo.
echo ========================================
echo [OK] Domain knowledge collection completed!
echo ========================================
echo Output: %OUTPUT_DIR%
echo.

:: 音声通知
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

endlocal

