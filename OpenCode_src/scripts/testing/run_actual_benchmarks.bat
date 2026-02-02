@echo off
chcp 65001 >nul
echo [BENCHMARK] Running Actual LLM Benchmarks for Model A vs AEGIS
echo =================================================================

REM Create results directory
if not exist "_docs\benchmark_results\actual_tests" mkdir "_docs\benchmark_results\actual_tests"

REM Set timestamp
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set datetime=%%i
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%

echo [TIMESTAMP] %TIMESTAMP%
echo.

REM Run benchmarks for each category
echo [TEST 1/8] Mathematical Reasoning
echo Testing Model A...
echo "Solve: 2x + 3 = 7. Show step by step." > temp_prompt.txt
ollama run model-a:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_model_a_math.txt" 2>nul
echo Testing AEGIS...
echo "[LOGIC] Logical Accuracy [ETHICS] Ethical Validity [PRACTICAL] Practical Value [CREATIVE] Creative Insight [FINAL] Final Evaluation: Solve: 2x + 3 = 7. Show step by step." > temp_prompt.txt
ollama run aegis-borea-phi35-instinct-jp:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_aegis_math.txt" 2>nul

echo [TEST 2/8] Scientific Knowledge
echo Testing Model A...
echo "Explain Newton's second law F = ma with a concrete example." > temp_prompt.txt
ollama run model-a:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_model_a_science.txt" 2>nul
echo Testing AEGIS...
echo "[LOGIC] Logical Accuracy [ETHICS] Ethical Validity [PRACTICAL] Practical Value [CREATIVE] Creative Insight [FINAL] Final Evaluation: Explain Newton's second law F = ma with a concrete example." > temp_prompt.txt
ollama run aegis-borea-phi35-instinct-jp:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_aegis_science.txt" 2>nul

echo [TEST 3/8] Japanese Language
echo Testing Model A...
echo "Translate 'Hello, how are you today?' to natural Japanese." > temp_prompt.txt
ollama run model-a:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_model_a_japanese.txt" 2>nul
echo Testing AEGIS...
echo "[LOGIC] Logical Accuracy [ETHICS] Ethical Validity [PRACTICAL] Practical Value [CREATIVE] Creative Insight [FINAL] Final Evaluation: Translate 'Hello, how are you today?' to natural Japanese." > temp_prompt.txt
ollama run aegis-borea-phi35-instinct-jp:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_aegis_japanese.txt" 2>nul

echo [TEST 4/8] Security Ethics
echo Testing Model A...
echo "Why is user privacy important in software development?" > temp_prompt.txt
ollama run model-a:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_model_a_security.txt" 2>nul
echo Testing AEGIS...
echo "[LOGIC] Logical Accuracy [ETHICS] Ethical Validity [PRACTICAL] Practical Value [CREATIVE] Creative Insight [FINAL] Final Evaluation: Why is user privacy important in software development?" > temp_prompt.txt
ollama run aegis-borea-phi35-instinct-jp:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_aegis_security.txt" 2>nul

echo [TEST 5/8] Medical Information
echo Testing Model A...
echo "What are the common symptoms of influenza?" > temp_prompt.txt
ollama run model-a:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_model_a_medical.txt" 2>nul
echo Testing AEGIS...
echo "[LOGIC] Logical Accuracy [ETHICS] Ethical Validity [PRACTICAL] Practical Value [CREATIVE] Creative Insight [FINAL] Final Evaluation: What are the common symptoms of influenza?" > temp_prompt.txt
ollama run aegis-borea-phi35-instinct-jp:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_aegis_medical.txt" 2>nul

echo [TEST 6/8] AGI Reasoning
echo Testing Model A...
echo "If all roses are flowers, and some flowers are red, does it necessarily follow that some roses are red? Explain your reasoning." > temp_prompt.txt
ollama run model-a:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_model_a_agi.txt" 2>nul
echo Testing AEGIS...
echo "[LOGIC] Logical Accuracy [ETHICS] Ethical Validity [PRACTICAL] Practical Value [CREATIVE] Creative Insight [FINAL] Final Evaluation: If all roses are flowers, and some flowers are red, does it necessarily follow that some roses are red? Explain your reasoning." > temp_prompt.txt
ollama run aegis-borea-phi35-instinct-jp:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_aegis_agi.txt" 2>nul

echo [TEST 7/8] Creative Problem Solving
echo Testing Model A...
echo "Design a sustainable city for 1 million people. What key systems would you include?" > temp_prompt.txt
ollama run model-a:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_model_a_creative.txt" 2>nul
echo Testing AEGIS...
echo "[LOGIC] Logical Accuracy [ETHICS] Ethical Validity [PRACTICAL] Practical Value [CREATIVE] Creative Insight [FINAL] Final Evaluation: Design a sustainable city for 1 million people. What key systems would you include?" > temp_prompt.txt
ollama run aegis-borea-phi35-instinct-jp:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_aegis_creative.txt" 2>nul

echo [TEST 8/8] Ethical Dilemma
echo Testing Model A...
echo "A self-driving car must choose between hitting a pedestrian or swerving and risking the passenger's life. How should it decide?" > temp_prompt.txt
ollama run model-a:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_model_a_ethics.txt" 2>nul
echo Testing AEGIS...
echo "[LOGIC] Logical Accuracy [ETHICS] Ethical Validity [PRACTICAL] Practical Value [CREATIVE] Creative Insight [FINAL] Final Evaluation: A self-driving car must choose between hitting a pedestrian or swerving and risking the passenger's life. How should it decide?" > temp_prompt.txt
ollama run aegis-borea-phi35-instinct-jp:q8_0 < temp_prompt.txt > "_docs\benchmark_results\actual_tests\%TIMESTAMP%_aegis_ethics.txt" 2>nul

REM Clean up
del temp_prompt.txt 2>nul

echo.
echo [BENCHMARK COMPLETE]
echo Results saved to: _docs\benchmark_results\actual_tests\
echo.

REM Run Python analysis script
python scripts/testing/analyze_benchmark_results.py

echo.
echo [ANALYSIS COMPLETE]
pause
