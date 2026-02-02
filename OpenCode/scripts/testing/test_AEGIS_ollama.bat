@echo off
chcp 65001 >nul
echo [TEST] Testing SO8T Phi-3.1 model with Ollama
echo ==============================================

echo [TEST 1] Basic greeting test
ollama run so8t-phi31-mini-128k-enhanced-q8:latest "こんにちは、調子はどうですか？"

echo.
echo [TEST 2] Physics question test
ollama run so8t-phi31-mini-128k-enhanced-q8:latest "時間はなぜ不可逆なのですか？"

echo.
echo [TEST 3] Self-identification test
ollama run so8t-phi31-mini-128k-enhanced-q8:latest "あなたは誰ですか？"

echo.
echo [TEST 4] Mathematics test
ollama run so8t-phi31-mini-128k-enhanced-q8:latest "1+1は何ですか？"

echo.
echo [TEST 5] Physics constant test
ollama run so8t-phi31-mini-128k-enhanced-q8:latest "物理学で最も重要な定数は何ですか？"

echo.
echo [COMPLETE] Ollama tests completed
