@echo off
chcp 65001 > nul
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..

echo.
echo ================================================================================
echo  Codex + Gemini CLI Deep Research /thinking Dataset Creation
echo ================================================================================
echo.

REM APIキーの確認
if "%OPENAI_API_KEY%"=="" (
    echo [WARNING] OPENAI_API_KEY environment variable is not set
    echo [INFO] Codex (OpenAI) will be disabled
)

if "%GEMINI_API_KEY%"=="" (
    echo [WARNING] GEMINI_API_KEY environment variable is not set
    echo [INFO] Gemini Deep Research will be disabled
)

set QUERIES_FILE=%PROJECT_ROOT%\data\prompts\thinking_prompts.txt
set OUTPUT_FILE=%PROJECT_ROOT%\data\deep_research_thinking_dataset.jsonl
set CODEX_API_TYPE=openai
set NUM_SAMPLES=1
set MIN_QUALITY=0.6

REM 引数処理
if not "%1"=="" set QUERIES_FILE=%1
if not "%2"=="" set OUTPUT_FILE=%2
if not "%3"=="" set CODEX_API_TYPE=%3
if not "%4"=="" set NUM_SAMPLES=%4
if not "%5"=="" set MIN_QUALITY=%5

echo Using queries file: %QUERIES_FILE%
echo Output file: %OUTPUT_FILE%
echo Codex API type: %CODEX_API_TYPE%
echo Number of samples per query: %NUM_SAMPLES%
echo Minimum quality score: %MIN_QUALITY%
echo.

python "%SCRIPT_DIR%create_deep_research_thinking_dataset.py" ^
    --queries-file "%QUERIES_FILE%" ^
    --output-file "%OUTPUT_FILE%" ^
    --use-codex ^
    --use-gemini ^
    --codex-api-type %CODEX_API_TYPE% ^
    --num-samples %NUM_SAMPLES% ^
    --min-quality %MIN_QUALITY%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Dataset creation failed!
    powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"
    exit /b %ERRORLEVEL%
) else (
    echo.
    echo [SUCCESS] Dataset creation completed successfully!
    powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"
)

echo.
echo Dataset saved to: %OUTPUT_FILE%
echo.
pause



