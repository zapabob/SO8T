@echo off
chcp 65001 >nul
echo [OLLAMA] Starting Ollama server for direct English complex testing...

echo [STEP 1] Starting Ollama server...
start /B ollama serve
echo [OK] Ollama server started in background
echo [WAIT] Waiting for server to be ready...
timeout /t 5 /nobreak >nul

echo [STEP 2] Testing complex mathematical reasoning...
echo ========================================
ollama run so8t-qwen2vl-2b:latest "Solve this complex mathematical problem step by step: Given a 4-dimensional hypercube, calculate the volume of the intersection with a 3-dimensional sphere of radius r centered at the origin. Show all mathematical steps and reasoning."
echo ========================================
echo.

echo [STEP 3] Testing scientific concepts...
echo ========================================
ollama run so8t-qwen2vl-2b:latest "Explain the quantum mechanical principles behind SO(8) rotation gates in neural networks. Include mathematical formulations and practical applications."
echo ========================================
echo.

echo [STEP 4] Testing logical reasoning...
echo ========================================
ollama run so8t-qwen2vl-2b:latest "Analyze this logical paradox: A barber shaves all and only those men in town who do not shave themselves. Who shaves the barber? Provide detailed logical analysis."
echo ========================================
echo.

echo [STEP 5] Testing ethical reasoning...
echo ========================================
ollama run so8t-qwen2vl-2b:latest "Evaluate the ethical implications of AI systems making autonomous decisions in healthcare. Consider both utilitarian and deontological perspectives."
echo ========================================
echo.

echo [STEP 6] Testing advanced language understanding...
echo ========================================
ollama run so8t-qwen2vl-2b:latest "Write a comprehensive analysis of the philosophical implications of artificial general intelligence, incorporating perspectives from epistemology, metaphysics, and ethics."
echo ========================================
echo.

echo [STEP 7] Testing problem-solving capabilities...
echo ========================================
ollama run so8t-qwen2vl-2b:latest "Design an algorithm to solve the traveling salesman problem with additional constraints: each city has a time window for visits, and the salesman must return to the starting city within a given time limit. Provide pseudocode and complexity analysis."
echo ========================================
echo.

echo [STEP 8] Testing creative reasoning...
echo ========================================
ollama run so8t-qwen2vl-2b:latest "Create a detailed scenario for how humanity might achieve interstellar travel within the next 100 years, considering current technological limitations and potential breakthroughs in physics, engineering, and biology."
echo ========================================
echo.

echo [AUDIO] Playing completion notification...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }"

echo [OLLAMA] English complex testing completed!
pause
