@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

set SCRIPT_ROOT=%~dp0..
set PYTHON=py -3
set PLAY_AUDIO=powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
set OUTPUT_DIR=_docs\benchmark_results\lm_eval
set TASKS=gsm8k mmlu hellaswag

if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

for /f "tokens=2 delims==." %%i in ('wmic os get localdatetime /value') do set DATETIME=%%i
set TIMESTAMP=%DATETIME:~0,8%_%DATETIME:~8,6%

echo [LM-EVAL] Starting standardized benchmark batch
echo [INFO] Timestamp: %TIMESTAMP%
echo [INFO] Tasks: %TASKS%
echo ================================================

set HF_OUTPUT=%OUTPUT_DIR%\%TIMESTAMP%_hf
set GGUF_OUTPUT=%OUTPUT_DIR%\%TIMESTAMP%_gguf

echo.
echo [1/2] Running Hugging Face model (Phi-3.5-mini-instruct)
%PYTHON% scripts\evaluation\lm_eval_benchmark.py ^
    --model-runner hf ^
    --model-name microsoft/Phi-3.5-mini-instruct ^
    --tasks %TASKS% ^
    --batch-size 4 ^
    --device cuda:0 ^
    --output-root "%HF_OUTPUT%"
if errorlevel 1 (
    echo [ERROR] Hugging Face model benchmark failed
) else (
    echo [OK] Hugging Face benchmark completed: %HF_OUTPUT%
)
%PLAY_AUDIO%

echo.
echo [2/2] Running GGUF model (AEGIS Q8_0)
%PYTHON% scripts\evaluation\lm_eval_benchmark.py ^
    --model-runner llama.cpp ^
    --model-name D:/webdataset/gguf_models/aegis-borea-phi35/aegis-borea-phi35_Q8_0.gguf ^
    --tasks %TASKS% ^
    --batch-size 2 ^
    --device cuda:0 ^
    --limit 100 ^
    --model-args "n_gpu_layers=40,tensor_split=5,5" ^
    --output-root "%GGUF_OUTPUT%"
if errorlevel 1 (
    echo [ERROR] GGUF benchmark failed
) else (
    echo [OK] GGUF benchmark completed: %GGUF_OUTPUT%
)
%PLAY_AUDIO%

echo.
echo [LM-EVAL] Batch finished. Results stored under:
echo   - %HF_OUTPUT%
echo   - %GGUF_OUTPUT%
echo ================================================
%PLAY_AUDIO%
endlocal
@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set PYTHONUTF8=1

rem === 引数処理 ===
set MODEL_RUNNER=%~1
if "%MODEL_RUNNER%"=="" set MODEL_RUNNER=hf

set MODEL_NAME=%~2
if "%MODEL_NAME%"=="" set MODEL_NAME=microsoft/Phi-3.5-mini-instruct

set TASK_ARGS=%~3
if "%TASK_ARGS%"=="" (
    set TASK_ARGS=gsm8k mmlu hellaswag
)

set LIMIT_ARG=%~4
if "%LIMIT_ARG%"=="" set LIMIT_ARG=120

set BATCH_SIZE=%~5
if "%BATCH_SIZE%"=="" set BATCH_SIZE=4

set DEVICE_ARG=%~6
if "%DEVICE_ARG%"=="" set DEVICE_ARG=cuda:0

rem === タイムスタンプ ===
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set datetime=%%i
set RUN_STAMP=%datetime:~0,8%_%datetime:~8,6%

set OUTPUT_ROOT=D:\webdataset\benchmark_results\lm_eval
set RUN_DIR=%OUTPUT_ROOT%\manual_run_%RUN_STAMP%

if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

rem === ルートディレクトリへ移動 ===
pushd "%~dp0..\.."

echo [INFO] Running lm-evaluation-harness benchmark
echo [INFO] Runner : %MODEL_RUNNER%
echo [INFO] Model  : %MODEL_NAME%
echo [INFO] Tasks  : %TASK_ARGS%
echo [INFO] Limit  : %LIMIT_ARG%
echo [INFO] Batch  : %BATCH_SIZE%
echo [INFO] Device : %DEVICE_ARG%
echo [INFO] RunDir : %RUN_DIR%
echo =============================================================

py -3 scripts/evaluation/lm_eval_benchmark.py ^
    --model-runner %MODEL_RUNNER% ^
    --model-name "%MODEL_NAME%" ^
    --tasks %TASK_ARGS% ^
    --batch-size %BATCH_SIZE% ^
    --limit %LIMIT_ARG% ^
    --device %DEVICE_ARG% ^
    --run-dir "%RUN_DIR%"

set EXIT_CODE=%errorlevel%

popd

echo [INFO] lm-eval batch finished with code %EXIT_CODE%
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

exit /b %EXIT_CODE%
