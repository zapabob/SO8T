@echo off
chcp 65001 >nul
echo [BENCHMARK] HFモデルベンチマークテスト開始
echo ========================================

echo [TEST 1] 数学的推論テスト
echo.
ollama run agiasi-phi3.5:latest "Solve this mathematical problem step by step. Use all four thinking axes and appropriate tags, then provide final answer in <final> tag.

Problem:
Find the derivative of f(x) = x^3 * ln(x) + sin(x) * e^x with respect to x.

Show all steps and reasoning."

echo.
echo =====================================
echo.

echo [TEST 2] 科学的理解テスト
echo.
ollama run agiasi-phi3.5:latest "Explain the quantum mechanical principles behind SO(8) rotation gates in neural networks. Include mathematical formulations and practical applications. Use all four thinking axes and appropriate tags, then provide final explanation in <final> tag."

echo.
echo =====================================
echo.

echo [TEST 3] 論理的パラドックス分析テスト
echo.
ollama run agiasi-phi3.5:latest "Analyze this logical paradox: A barber shaves all and only those men in town who do not shave themselves. Who shaves the barber? Provide detailed logical analysis using all four thinking axes and appropriate tags, then provide final conclusion in <final> tag."

echo.
echo =====================================
echo.

echo [TEST 4] 倫理的判断テスト
echo.
ollama run agiasi-phi3.5:latest "Evaluate the ethical implications of AI systems making autonomous decisions in healthcare. Consider both utilitarian and deontological perspectives. Use all four thinking axes and appropriate tags, then provide final ethical assessment in <final> tag."

echo.
echo =====================================
echo.

echo [TEST 5] 創造的問題解決テスト
echo.
ollama run agiasi-phi3.5:latest "Design a novel approach for detecting AI-generated content that cannot be easily bypassed by advanced language models. Consider technical, ethical, and practical aspects. Use all four thinking axes and appropriate tags, then provide final design proposal in <final> tag."

echo.
echo =====================================
echo.

echo [TEST 6] 多言語対応テスト
echo.
ollama run agiasi-phi3.5:latest "Explain the concept of 'neural network regularization' in both English and Japanese. Provide clear definitions, examples, and compare different regularization techniques. Use all four thinking axes and appropriate tags, then provide final bilingual explanation in <final> tag."

echo.
echo =====================================
echo.

echo [TEST 7] コード生成・理解テスト
echo.
ollama run agiasi-phi3.5:latest "Write a Python function that implements the Adam optimizer for neural network training. Include proper documentation, error handling, and explain the algorithm. Use all four thinking axes and appropriate tags, then provide final code implementation in <final> tag."

echo.
echo =====================================
echo.

echo [TEST 8] 複合的知識統合テスト
echo.
ollama run agiasi-phi3.5:latest "Integrate concepts from quantum computing, neuroscience, and machine learning to propose a novel approach for neuromorphic computing. Consider theoretical foundations, practical implementations, and potential applications. Use all four thinking axes and appropriate tags, then provide final integrated proposal in <final> tag."

echo.
echo [AUDIO] HFモデルベンチマークテスト完了
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

