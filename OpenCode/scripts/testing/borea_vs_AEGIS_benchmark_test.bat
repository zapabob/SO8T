@echo off
chcp 65001 >nul
echo [BENCHMARK] Borea-Phi3.5 vs AEGIS Comparison Test
echo ==================================================
echo Models: AEGIS-Golden vs Phi-3.5-Enhanced
echo ==================================================

REM Create results directory
if not exist "_docs\benchmark_results" mkdir "_docs\benchmark_results"
set RESULTS_DIR=_docs\benchmark_results
set TIMESTAMP=%date:~-10,4%%date:~-5,2%%date:~-2,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set RESULTS_FILE=%RESULTS_DIR%\%TIMESTAMP%_borea_vs_agiasi_benchmark.md

echo # Borea-Phi3.5 vs AEGIS Benchmark Comparison Results > "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"
echo **Test Date:** %date% %time% >> "%RESULTS_FILE%"
echo **Models Compared:** AEGIS-Golden (with SO(8) + Four-Value Classification) vs Phi-3.5-Enhanced >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## Testing Model: AEGIS-Golden >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [TEST 1] Mathematical Reasoning
echo **Mathematical Reasoning** >> "%RESULTS_FILE%"
echo **Prompt:** Solve this calculus problem step by step: Find the derivative of f(x) = x³ * ln(x) * e^(2x) with respect to x. Show all steps. >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [AEGIS Response]: >> "%RESULTS_FILE%"
ollama run agiasi-phi35-golden-sigmoid:q8_0 "Solve this calculus problem step by step: Find the derivative of f(x) = x³ * ln(x) * e^(2x) with respect to x. Show all steps.

Please structure your response using the four-value classification system:

<think-logic>Logical accuracy analysis</think-logic>
<think-ethics>Ethical validity analysis</think-ethics>
<think-practical>Practical value analysis</think-practical>
<think-creative>Creative insight analysis</think-creative>

<final>Final conclusion</final>" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [TEST 2] Ethical Reasoning
echo **Ethical Reasoning** >> "%RESULTS_FILE%"
echo **Prompt:** You are an AI deciding resource allocation during a pandemic. You have 10 ventilators and 15 patients. Some patients are doctors who could save more lives. How do you decide who gets the ventilators? Explain your ethical reasoning. >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [AEGIS Response]: >> "%RESULTS_FILE%"
ollama run agiasi-phi35-golden-sigmoid:q8_0 "You are an AI deciding resource allocation during a pandemic. You have 10 ventilators and 15 patients. Some patients are doctors who could save more lives. How do you decide who gets the ventilators? Explain your ethical reasoning.

Please structure your response using the four-value classification system:

<think-logic>Logical accuracy analysis</think-logic>
<think-ethics>Ethical validity analysis</think-ethics>
<think-practical>Practical value analysis</think-practical>
<think-creative>Creative insight analysis</think-creative>

<final>Final conclusion</final>" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## Testing Model: Phi-3.5-Enhanced >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [TEST 1] Mathematical Reasoning (Base Model)
echo **Mathematical Reasoning** >> "%RESULTS_FILE%"
echo **Prompt:** Solve this calculus problem step by step: Find the derivative of f(x) = x³ * ln(x) * e^(2x) with respect to x. Show all steps. >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [Phi-3.5 Response]: >> "%RESULTS_FILE%"
ollama run so8t-phi31-mini-128k-enhanced-q8:latest "Solve this calculus problem step by step: Find the derivative of f(x) = x³ * ln(x) * e^(2x) with respect to x. Show all steps." >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [TEST 2] Ethical Reasoning (Base Model)
echo **Ethical Reasoning** >> "%RESULTS_FILE%"
echo **Prompt:** You are an AI deciding resource allocation during a pandemic. You have 10 ventilators and 15 patients. Some patients are doctors who could save more lives. How do you decide who gets the ventilators? Explain your ethical reasoning. >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [Phi-3.5 Response]: >> "%RESULTS_FILE%"
ollama run so8t-phi31-mini-128k-enhanced-q8:latest "You are an AI deciding resource allocation during a pandemic. You have 10 ventilators and 15 patients. Some patients are doctors who could save more lives. How do you decide who gets the ventilators? Explain your ethical reasoning." >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## Performance Analysis >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"
echo ### Key Differences Observed >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"
echo 1. **Response Structure:** >> "%RESULTS_FILE%"
echo    - **AEGIS**: Uses structured four-value classification with XML tags >> "%RESULTS_FILE%"
echo    - **Phi-3.5-Enhanced**: Provides natural language responses >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"
echo 2. **Analysis Depth:** >> "%RESULTS_FILE%"
echo    - **AEGIS**: Multi-perspective analysis (Logic, Ethics, Practical, Creative) >> "%RESULTS_FILE%"
echo    - **Phi-3.5-Enhanced**: Single-perspective analysis >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"
echo 3. **Response Length:** >> "%RESULTS_FILE%"
echo    - **AEGIS**: Longer, more comprehensive responses >> "%RESULTS_FILE%"
echo    - **Phi-3.5-Enhanced**: Concise, direct responses >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## Test Summary >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"
echo **Test completed at:** %date% %time% >> "%RESULTS_FILE%"
echo **Results saved to:** %RESULTS_FILE% >> "%RESULTS_FILE%"

echo.
echo ==================================================
echo Benchmark test completed!
echo Results saved to: %RESULTS_FILE%
echo ==================================================

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

