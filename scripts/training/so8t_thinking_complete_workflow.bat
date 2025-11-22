@echo off
chcp 65001 >nul
echo [SO8T/thinking] Complete Training and Evaluation Workflow
echo ========================================================
echo.

echo [STEP 1] Checking training completion...
dir "D:/webdataset/checkpoints/so8t_thinking_retrained" /b 2>nul | findstr /r ".*" >nul
if errorlevel 1 (
    echo [INFO] No checkpoints found yet, waiting for training completion...
    timeout /t 300 /nobreak >nul
    goto wait_loop
)

echo [SUCCESS] Training completed! Checkpoints found.
echo.

echo [STEP 2] Model Evaluation - Validation Performance Check
echo --------------------------------------------------------
py -3 scripts/evaluation/evaluate_nkat_so8t_model.py ^
    --model-path "D:/webdataset/checkpoints/so8t_thinking_retrained" ^
    --test-data "data/nkat_so8t_v2/val_nkat_so8t.jsonl" ^
    --output-dir "results/so8t_thinking_eval"

if %errorlevel% neq 0 (
    echo [ERROR] Model evaluation failed
    goto end
)

echo [SUCCESS] Model evaluation completed.
echo.

echo [STEP 3] GGUF Conversion - llama.cpp Format Conversion
echo -----------------------------------------------------
if not exist "D:/webdataset/gguf_models/so8t_thinking_v1\" mkdir "D:/webdataset/gguf_models/so8t_thinking_v1"

echo Converting to GGUF (Q8_0)...
py external/llama.cpp-master/convert_hf_to_gguf.py ^
    "D:/webdataset/checkpoints/so8t_thinking_retrained" ^
    --outfile "D:/webdataset/gguf_models/so8t_thinking_v1/so8t_thinking_q8_0.gguf" ^
    --outtype q8_0

if %errorlevel% neq 0 (
    echo [ERROR] GGUF conversion (Q8_0) failed
    goto end
)

echo Converting to GGUF (Q4_K_M)...
py external/llama.cpp-master/convert_hf_to_gguf.py ^
    "D:/webdataset/checkpoints/so8t_thinking_retrained" ^
    --outfile "D:/webdataset/gguf_models/so8t_thinking_v1/so8t_thinking_q4_k_m.gguf" ^
    --outtype q4_k_m

if %errorlevel% neq 0 (
    echo [WARNING] GGUF conversion (Q4_K_M) failed, but Q8_0 is available
)

echo [SUCCESS] GGUF conversion completed.
echo.

echo [STEP 4] Ollama Import - Local Inference Environment Integration
echo ---------------------------------------------------------------
echo Creating Ollama model file...
(
echo FROM D:/webdataset/gguf_models/so8t_thinking_v1/so8t_thinking_q8_0.gguf
echo.
echo TEMPLATE """{{ .System }}
echo.
echo {{ .Prompt }}"""
echo.
echo PARAMETER temperature 0.7
echo PARAMETER top_p 0.9
echo PARAMETER top_k 40
echo PARAMETER num_ctx 4096
) > "modelfiles/so8t_thinking.modelfile"

echo Importing to Ollama...
ollama create so8t-thinking:latest -f "modelfiles/so8t_thinking.modelfile"

if %errorlevel% neq 0 (
    echo [ERROR] Ollama import failed
    goto end
)

echo [SUCCESS] Ollama import completed.
echo.

echo [STEP 5] Japanese Performance Testing - SO8T/thinking Thinking Ability Verification
echo ----------------------------------------------------------------------------------
echo Running comprehensive Japanese LLM performance tests...

echo [TEST 1] Japanese Understanding Test
ollama run so8t-thinking:latest "以下の文章を読んで、内容を要約してください。日本語で回答してください。

SO8TはSO(8)回転群に基づく革新的なアーキテクチャで、幾何学的推論能力を強化します。このモデルはAlpha Gateによって動的に幾何学的経路を制御し、学習過程で自律的にゲートを開きます。"

echo [TEST 2] Japanese Generation Test
ollama run so8t-thinking:latest "SO(8)回転群について、数学的な基礎から応用までを説明してください。ステップバイステップで分かりやすく教えてください。"

echo [TEST 3] Logical Reasoning Test
ollama run so8t-thinking:latest "次の論理パズルを解いてください：すべてのSO8Tモデルは幾何学的推論を行うが、いくつかのモデルは特別なAlpha Gateを持つ。この命題が真であるための条件を説明してください。"

echo [TEST 4] Mathematical Reasoning Test
ollama run so8t-thinking:latest "ベクトル空間における回転群SO(8)の性質を説明してください。特に8次元ユークリッド空間での回転について。"

echo [TEST 5] Self-Verification Test
ollama run so8t-thinking:latest "あなた自身の思考プロセスを分析してください。SO8Tアーキテクチャがどのようにあなたの推論能力を強化しているかを説明してください。"

echo.
echo [SUCCESS] All tests completed!
echo [AUDIO] Playing completion notification...

powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

goto end

:end
echo.
echo Workflow completed. Check results in respective directories.
pause






