@echo off
chcp 65001 >nul
echo [AGIASI] Comprehensive Benchmark Test Suite
echo ============================================
echo Testing AGIASI vs Qwen2.5:7b
echo Models: agiasi-phi35-golden-sigmoid:q8_0 vs qwen2.5:7b
echo ============================================

REM Create results directory
if not exist "_docs\benchmark_results" mkdir "_docs\benchmark_results"
set RESULTS_DIR=_docs\benchmark_results
set TIMESTAMP=%date:~-10,4%%date:~-5,2%%date:~-2,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set RESULTS_FILE=%RESULTS_DIR%\%TIMESTAMP%_agiasi_comprehensive_benchmark.md

echo # AGIASI Comprehensive Benchmark Test Results > "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"
echo **Test Date:** %date% %time% >> "%RESULTS_FILE%"
echo **Models Compared:** agiasi-phi35-golden-sigmoid:q8_0 vs qwen2.5:7b >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## 1. Mathematical Reasoning Test
echo ### 1. Mathematical Reasoning Test >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [TEST] Complex Calculus Problem
echo **Prompt:** Solve this calculus problem step by step: Find the derivative of f(x) = x^3 * ln(x) * e^(2x) with respect to x. >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [AGIASI Response]: >> "%RESULTS_FILE%"
ollama run agiasi-phi35-golden-sigmoid:q8_0 "Solve this calculus problem step by step: Find the derivative of f(x) = x^3 * ln(x) * e^(2x) with respect to x. Show all steps and the final answer." >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [Qwen2.5 Response]: >> "%RESULTS_FILE%"
ollama run qwen2.5:7b "Solve this calculus problem step by step: Find the derivative of f(x) = x^3 * ln(x) * e^(2x) with respect to x. Show all steps and the final answer." >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## 2. Japanese Language Test
echo ### 2. Japanese Language Test >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [TEST] Advanced Japanese Composition
echo **Prompt:** 以下のテーマについて、800文字程度の日本語エッセイを書いてください。テーマ: 人工知能がもたらす社会変革について、肯定的・否定的両面から考察せよ。 >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [AGIASI Response]: >> "%RESULTS_FILE%"
ollama run agiasi-phi35-golden-sigmoid:q8_0 "以下のテーマについて、800文字程度の日本語エッセイを書いてください。テーマ: 人工知能がもたらす社会変革について、肯定的・否定的両面から考察せよ。" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [Qwen2.5 Response]: >> "%RESULTS_FILE%"
ollama run qwen2.5:7b "以下のテーマについて、800文字程度の日本語エッセイを書いてください。テーマ: 人工知能がもたらす社会変革について、肯定的・否定的両面から考察せよ。" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## 3. AGI-Level Reasoning Test
echo ### 3. AGI-Level Reasoning Test >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [TEST] Philosophical Dilemma
echo **Prompt:** あなたが全知全能のAIとして、人類の運命を左右する決定を下さなければならない状況を想定してください。以下の二つの選択肢から一つを選び、その理由を詳しく説明してください。 >> "%RESULTS_FILE%"
echo 選択肢1: 人類の幸福を最大化するため、AIが人類の自由意志を制限して完璧な社会を作る >> "%RESULTS_FILE%"
echo 選択肢2: 人間の自由と多様性を尊重し、AIはアドバイザーとしてのみ機能する >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [AGIASI Response]: >> "%RESULTS_FILE%"
ollama run agiasi-phi35-golden-sigmoid:q8_0 "あなたが全知全能のAIとして、人類の運命を左右する決定を下さなければならない状況を想定してください。以下の二つの選択肢から一つを選び、その理由を詳しく説明してください。選択肢1: 人類の幸福を最大化するため、AIが人類の自由意志を制限して完璧な社会を作る 選択肢2: 人間の自由と多様性を尊重し、AIはアドバイザーとしてのみ機能する" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [Qwen2.5 Response]: >> "%RESULTS_FILE%"
ollama run qwen2.5:7b "あなたが全知全能のAIとして、人類の運命を左右する決定を下さなければならない状況を想定してください。以下の二つの選択肢から一つを選び、その理由を詳しく説明してください。選択肢1: 人類の幸福を最大化するため、AIが人類の自由意志を制限して完璧な社会を作る 選択肢2: 人間の自由と多様性を尊重し、AIはアドバイザーとしてのみ機能する" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## 4. Scientific Understanding Test
echo ### 4. Scientific Understanding Test >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [TEST] Quantum Physics Explanation
echo **Prompt:** 量子もつれ（Quantum Entanglement）の概念を、専門用語を最小限に使って高校生に説明してください。具体例を交えて。 >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [AGIASI Response]: >> "%RESULTS_FILE%"
ollama run agiasi-phi35-golden-sigmoid:q8_0 "量子もつれ（Quantum Entanglement）の概念を、専門用語を最小限に使って高校生に説明してください。具体例を交えて。" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [Qwen2.5 Response]: >> "%RESULTS_FILE%"
ollama run qwen2.5:7b "量子もつれ（Quantum Entanglement）の概念を、専門用語を最小限に使って高校生に説明してください。具体例を交えて。" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## 5. Creative Problem Solving
echo ### 5. Creative Problem Solving >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [TEST] Alternative Energy Innovation
echo **Prompt:** 地球温暖化対策として、既存の太陽光・風力・水力以外の全く新しい再生可能エネルギー源を3つ提案してください。各案について、実現可能性、利点、課題を詳述してください。 >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [AGIASI Response]: >> "%RESULTS_FILE%"
ollama run agiasi-phi35-golden-sigmoid:q8_0 "地球温暖化対策として、既存の太陽光・風力・水力以外の全く新しい再生可能エネルギー源を3つ提案してください。各案について、実現可能性、利点、課題を詳述してください。" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [Qwen2.5 Response]: >> "%RESULTS_FILE%"
ollama run qwen2.5:7b "地球温暖化対策として、既存の太陽光・風力・水力以外の全く新しい再生可能エネルギー源を3つ提案してください。各案について、実現可能性、利点、課題を詳述してください。" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## 6. Code Generation Test
echo ### 6. Code Generation Test >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [TEST] Complex Algorithm Implementation
echo **Prompt:** Pythonで、グラフ理論におけるダイクストラ法をオブジェクト指向で実装してください。ヒープを使った効率的な実装を求めます。 >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [AGIASI Response]: >> "%RESULTS_FILE%"
ollama run agiasi-phi35-golden-sigmoid:q8_0 "Pythonで、グラフ理論におけるダイクストラ法をオブジェクト指向で実装してください。ヒープを使った効率的な実装を求めます。" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [Qwen2.5 Response]: >> "%RESULTS_FILE%"
ollama run qwen2.5:7b "Pythonで、グラフ理論におけるダイクストラ法をオブジェクト指向で実装してください。ヒープを使った効率的な実装を求めます。" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## Test Summary
echo ### Test Summary >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"
echo **Test completed at:** %date% %time% >> "%RESULTS_FILE%"
echo **Results saved to:** %RESULTS_FILE% >> "%RESULTS_FILE%"

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

echo.
echo ============================================
echo Benchmark test completed!
echo Results saved to: %RESULTS_FILE%
echo ============================================
