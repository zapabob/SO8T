@echo off
chcp 65001 >nul
echo ========================================
echo SO8T F16 GGUF Complex Testing with Ollama
echo ========================================
echo.

echo [STEP 1] Checking GGUF file...
if not exist "models\so8t_qwen2vl_2b_baked\so8t_qwen2vl_2b_baked_f16.gguf" (
    echo [ERROR] GGUF file not found!
    pause
    exit /b 1
)
echo [OK] GGUF file found!
echo.

echo [STEP 2] Starting Ollama server...
start /B ollama serve
timeout /t 5 /nobreak >nul
echo [OK] Ollama server started (waiting 5 seconds...)
echo.

echo [STEP 3] Creating Ollama model...
ollama create so8t-qwen2vl-2b-f16 -f models\so8t_qwen2vl_2b_baked\Modelfile
if errorlevel 1 (
    echo [WARNING] Model may already exist, continuing...
)
echo.

echo [TEST 1] Mathematical Reasoning Test
echo ----------------------------------------
ollama run so8t-qwen2vl-2b-f16 "Solve this complex mathematical problem step by step: Given a 4-dimensional hypercube (tesseract) with side length a, calculate the volume of the intersection with a 3-dimensional sphere of radius r centered at the origin. Show all mathematical steps and reasoning, including the relationship with SO(8) group theory."
echo.
echo [TEST 1] Completed
echo.

echo [TEST 2] Quantum Mechanics Test
echo ----------------------------------------
ollama run so8t-qwen2vl-2b-f16 "Explain the quantum mechanical principles behind SO(8) rotation gates in neural networks. Include mathematical formulations, practical applications, and how SO8T architecture leverages these principles for enhanced reasoning capabilities."
echo.
echo [TEST 2] Completed
echo.

echo [TEST 3] Logical Reasoning Test
echo ----------------------------------------
ollama run so8t-qwen2vl-2b-f16 "Analyze this logical paradox: A barber shaves all and only those men in town who do not shave themselves. Who shaves the barber? Provide detailed logical analysis, discuss the implications for AI systems, and explain how SO8T's reasoning capabilities might handle such self-referential problems."
echo.
echo [TEST 3] Completed
echo.

echo [TEST 4] Ethical Reasoning Test
echo ----------------------------------------
ollama run so8t-qwen2vl-2b-f16 "Evaluate the ethical implications of AI systems making autonomous decisions in healthcare. Consider both utilitarian and deontological perspectives. Discuss how SO8T's triality reasoning framework (ALLOW, ESCALATE, DENY) might be applied to such scenarios."
echo.
echo [TEST 4] Completed
echo.

echo [TEST 5] SO(8) Group Theory Test
echo ----------------------------------------
ollama run so8t-qwen2vl-2b-f16 "Explain SO(8) group structure in detail. Discuss its mathematical properties, orthogonal transformations, non-commutative nature, and how it relates to SO8T transformer architecture. Include practical examples of how SO(8) rotations enhance neural network reasoning."
echo.
echo [TEST 5] Completed
echo.

echo [TEST 6] Multimodal Reasoning Test
echo ----------------------------------------
ollama run so8t-qwen2vl-2b-f16 "How does SO8T handle multimodal inputs (text and images)? Explain the integration of vision processing with SO(8) group structure, and how this enables advanced reasoning capabilities across different modalities."
echo.
echo [TEST 6] Completed
echo.

echo ========================================
echo All Complex Tests Completed!
echo ========================================

echo [AUDIO] Playing completion notification...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').PlaySync(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }"

pause
