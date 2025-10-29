@echo off
chcp 65001 >nul
echo [SO8T OLLAMA COMPLEX TEST] 複雑なテスト実行開始
echo ========================================

echo [TEST 1] 数学的推論テスト - SO(8)群構造
echo ----------------------------------------
ollama run so8t-vl-2b-instruct "Solve this complex mathematical problem step by step: Given a 4-dimensional hypercube, calculate the volume of the intersection with a 3-dimensional sphere of radius r centered at the origin. Show all mathematical steps and reasoning using SO(8) group theory principles."

echo.
echo [TEST 2] 科学的概念テスト - 量子力学
echo ----------------------------------------
ollama run so8t-vl-2b-instruct "Explain the quantum mechanical principles behind SO(8) rotation gates in neural networks. Include mathematical formulations, practical applications, and how they relate to quantum computing and machine learning."

echo.
echo [TEST 3] 論理的推論テスト - パラドックス
echo ----------------------------------------
ollama run so8t-vl-2b-instruct "Analyze this logical paradox: A barber shaves all and only those men in town who do not shave themselves. Who shaves the barber? Provide detailed logical analysis using formal logic and set theory."

echo.
echo [TEST 4] 倫理的推論テスト - AI安全性
echo ----------------------------------------
ollama run so8t-vl-2b-instruct "Evaluate the ethical implications of AI systems making autonomous decisions in healthcare. Consider both utilitarian and deontological perspectives, and discuss how SO(8) group structure might contribute to ethical reasoning in AI systems."

echo.
echo [TEST 5] 複雑な問題解決テスト - アルゴリズム
echo ----------------------------------------
ollama run so8t-vl-2b-instruct "Design an algorithm to solve the traveling salesman problem using SO(8) group rotations for optimization. Explain how the non-commutative properties of SO(8) can be leveraged for better solutions."

echo.
echo [TEST 6] 科学計算テスト - 数値解析
echo ----------------------------------------
ollama run so8t-vl-2b-instruct "Calculate the eigenvalues and eigenvectors of a 8x8 SO(8) rotation matrix. Show the mathematical derivation and explain the physical significance of these eigenvalues in the context of neural network transformations."

echo.
echo [TEST 7] 哲学的問題テスト - 意識とAI
echo ----------------------------------------
ollama run so8t-vl-2b-instruct "Discuss the philosophical implications of consciousness in AI systems. How might SO(8) group structure relate to the emergence of consciousness? Consider both materialist and dualist perspectives."

echo.
echo [TEST 8] 複雑な言語理解テスト - 多言語
echo ----------------------------------------
ollama run so8t-vl-2b-instruct "Translate this complex Japanese text into English while maintaining the mathematical precision: 'SO(8)群の非可換性は、量子力学における不確定性原理と類似の性質を持ち、ニューラルネットワークの表現能力を根本的に拡張する可能性を秘めている。'"

echo.
echo [TEST 9] 創造的問題解決テスト - イノベーション
echo ----------------------------------------
ollama run so8t-vl-2b-instruct "Imagine a new application of SO(8) group theory in artificial intelligence that hasn't been explored yet. Describe the concept, potential benefits, implementation challenges, and expected outcomes."

echo.
echo [TEST 10] 統合テスト - 複合問題
echo ----------------------------------------
ollama run so8t-vl-2b-instruct "Create a comprehensive analysis of how SO(8) group structure could revolutionize machine learning. Include: 1) Mathematical foundations, 2) Practical applications, 3) Ethical considerations, 4) Implementation challenges, 5) Future research directions. Make it detailed and scientifically rigorous."

echo.
echo [AUDIO] テスト完了通知を再生中...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] 音声通知再生完了' -ForegroundColor Green } else { Write-Host '[WARNING] 音声ファイルが見つかりません' -ForegroundColor Yellow }"

echo.
echo [COMPLETE] 複雑なテスト実行完了！
echo ========================================
