@echo off
chcp 65001 > nul
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..

echo.
echo ================================================================================
echo  Codex /thinking Dataset Creation
echo ================================================================================
echo.

REM APIキーの確認
if "%OPENAI_API_KEY%"=="" (
    echo [WARNING] OPENAI_API_KEY environment variable is not set
    echo [INFO] Please set OPENAI_API_KEY or use --api-key argument
)

set PROMPTS_FILE=%PROJECT_ROOT%\data\prompts\thinking_prompts.txt
set OUTPUT_FILE=%PROJECT_ROOT%\data\codex_thinking_dataset.jsonl
set API_TYPE=openai
set NUM_SAMPLES=1
set MIN_QUALITY=0.5

REM 引数処理
if not "%1"=="" set PROMPTS_FILE=%1
if not "%2"=="" set OUTPUT_FILE=%2
if not "%3"=="" set API_TYPE=%3
if not "%4"=="" set NUM_SAMPLES=%4
if not "%5"=="" set MIN_QUALITY=%5

echo Using prompts file: %PROMPTS_FILE%
echo Output file: %OUTPUT_FILE%
echo API type: %API_TYPE%
echo Number of samples per prompt: %NUM_SAMPLES%
echo Minimum quality score: %MIN_QUALITY%
echo.

python "%SCRIPT_DIR%create_codex_thinking_dataset.py" ^
    --api-type %API_TYPE% ^
    --prompts-file "%PROMPTS_FILE%" ^
    --output-file "%OUTPUT_FILE%" ^
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







