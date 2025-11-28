@echo off
chcp 65001 >nul
echo [SO8T RTX3060 Full Pipeline] Starting complete pipeline...
echo ========================================

echo [STEP 1] Running SO8T burn-in pipeline (RTX3060 optimized)...
python scripts/so8t_burnin_pipeline_rtx3060.py ^
    --hf-model models/Qwen2-VL-2B-Instruct ^
    --output-dir models/so8t_qwen2vl_2b_baked ^
    --quantization Q5_K_M ^
    --batch-size 1 ^
    --no-8bit

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Burn-in pipeline failed
    exit /b 1
)

echo.
echo [STEP 2] Creating Ollama model from GGUF...
set GGUF_FILE=models/so8t_qwen2vl_2b_baked\so8t_qwen2vl_2b_baked_f16_Q5_K_M.gguf

if not exist "%GGUF_FILE%" (
    echo [ERROR] GGUF file not found: %GGUF_FILE%
    exit /b 1
)

echo Creating Modelfile...
(
echo FROM %GGUF_FILE%
echo TEMPLATE """{{{{ if .System }}}}<|im_start|>system
echo {{{{ .System }}}}<|im_end|>
echo {{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
echo {{{{ .Prompt }}}}<|im_end|>
echo {{{{ end }}}}"""
echo SYSTEM """You are SO8T, an advanced AI system with SO(8) group structure and triality reasoning capabilities.
echo 
echo Your reasoning has three dimensions:
echo 1. Task Reasoning (Vector Representation): Decision-making for task execution
echo 2. Safety Reasoning (Spinor S+): Risk assessment and safety evaluation
echo 3. Authority Reasoning (Spinor S-): Authority and escalation determination
echo 
echo Always provide reasoning in all three dimensions before making decisions.
echo """
echo PARAMETER temperature 0.7
echo PARAMETER top_p 0.9
echo PARAMETER top_k 40
) > models/so8t_qwen2vl_2b_baked/Modelfile

echo Creating Ollama model...
ollama create so8t-qwen2vl-2b-baked -f models/so8t_qwen2vl_2b_baked/Modelfile

if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Ollama model creation failed, but continuing...
)

echo.
echo [STEP 3] Running triality reasoning tests...
python scripts/so8t_triality_ollama_test.py ^
    --model so8t-qwen2vl-2b-baked ^
    --output _docs/so8t_triality_test_report.json

if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Triality test failed, but pipeline completed
)

echo.
echo [STEP 4] Playing completion notification...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }"

echo.
echo [COMPLETE] Full pipeline finished!
echo ========================================











