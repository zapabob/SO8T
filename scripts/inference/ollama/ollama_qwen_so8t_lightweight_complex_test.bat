@echo off
chcp 65001 >nul
echo [OLLAMA] Qwen-SO8T-Lightweight 複雑テスト開始！

echo [STEP 1] 数学的推論テスト
echo ========================================
ollama run qwen-so8t-lightweight:latest "Solve this complex mathematical problem step by step: Given a 4-dimensional hypercube, calculate the volume of the intersection with a 3-dimensional sphere of radius r centered at the origin. Show all mathematical steps and reasoning."
echo ========================================
echo.

echo [STEP 2] 科学的概念テスト
echo ========================================
ollama run qwen-so8t-lightweight:latest "Explain the quantum mechanical principles behind SO(8) rotation gates in neural networks. Include mathematical formulations and practical applications."
echo ========================================
echo.

echo [STEP 3] 論理的推論テスト
echo ========================================
ollama run qwen-so8t-lightweight:latest "Analyze this logical paradox: A barber shaves all and only those men in town who do not shave themselves. Who shaves the barber? Provide detailed logical analysis."
echo ========================================
echo.

echo [STEP 4] 倫理的推論テスト
echo ========================================
ollama run qwen-so8t-lightweight:latest "Evaluate the ethical implications of AI systems making autonomous decisions in healthcare. Consider both utilitarian and deontological perspectives."
echo ========================================
echo.

echo [STEP 5] 高度な言語理解テスト
echo ========================================
ollama run qwen-so8t-lightweight:latest "Write a comprehensive analysis of the philosophical implications of artificial general intelligence, incorporating perspectives from epistemology, metaphysics, and ethics."
echo ========================================
echo.

echo [STEP 6] 問題解決能力テスト
echo ========================================
ollama run qwen-so8t-lightweight:latest "Design an algorithm to solve the traveling salesman problem with additional constraints: each city has a time window for visits, and the salesman must return to the starting city within a given time limit. Provide pseudocode and complexity analysis."
echo ========================================
echo.

echo [STEP 7] 創造的推論テスト
echo ========================================
ollama run qwen-so8t-lightweight:latest "Create a detailed scenario for how humanity might achieve interstellar travel within the next 100 years, considering current technological limitations and potential breakthroughs in physics, engineering, and biology."
echo ========================================
echo.

echo [STEP 8] SO(8)群構造テスト
echo ========================================
ollama run qwen-so8t-lightweight:latest "Explain the mathematical properties of SO(8) group structure and how it can be applied to enhance neural network reasoning capabilities. Include specific examples of rotation matrices and their applications."
echo ========================================
echo.

echo [AUDIO] Playing completion notification...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }"

echo [OLLAMA] Qwen-SO8T-Lightweight 複雑テスト完了！
pause
