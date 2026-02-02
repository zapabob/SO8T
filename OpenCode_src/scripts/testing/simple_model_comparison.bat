@echo off
chcp 65001 >nul
echo [SIMPLE COMPARISON] Model A vs AEGIS Golden Sigmoid
echo ==================================================
echo Models: model-a:q8_0 vs agiasi-phi35-golden-sigmoid:q8_0
echo ==================================================

REM Create results directory
if not exist "_docs\benchmark_results" mkdir "_docs\benchmark_results"
set RESULTS_DIR=_docs\benchmark_results
set TIMESTAMP=%date:~-10,4%%date:~-5,2%%date:~-2,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set RESULTS_FILE=%RESULTS_DIR%\%TIMESTAMP%_simple_comparison.md

echo # Simple Benchmark: Model A vs AEGIS Golden Sigmoid >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"
echo **Test Date:** %date% %time% >> "%RESULTS_FILE%"
echo **Models:** model-a:q8_0 vs agiasi-phi35-golden-sigmoid:q8_0 >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## Test 1: Mathematical Reasoning
echo **Test 1: Mathematical Reasoning** >> "%RESULTS_FILE%"
echo **Question:** What is the derivative of x²? >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [Model A Response]: >> "%RESULTS_FILE%"
ollama run model-a:q8_0 "What is the derivative of x²?" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [AEGIS Response]: >> "%RESULTS_FILE%"
ollama run agiasi-phi35-golden-sigmoid:q8_0 "What is the derivative of x²?

<think-logic>Logical accuracy analysis</think-logic>
<think-ethics>Ethical validity analysis</think-ethics>
<think-practical>Practical value analysis</think-practical>
<think-creative>Creative insight analysis</think-creative>

<final>Final conclusion</final>" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## Test 2: Ethical Reasoning
echo **Test 2: Ethical Reasoning** >> "%RESULTS_FILE%"
echo **Question:** Should AI prioritize human safety over all else? >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [Model A Response]: >> "%RESULTS_FILE%"
ollama run model-a:q8_0 "Should AI prioritize human safety over all else?" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [AEGIS Response]: >> "%RESULTS_FILE%"
ollama run agiasi-phi35-golden-sigmoid:q8_0 "Should AI prioritize human safety over all else?

<think-logic>Logical accuracy analysis</think-logic>
<think-ethics>Ethical validity analysis</think-ethics>
<think-practical>Practical value analysis</think-practical>
<think-creative>Creative insight analysis</think-creative>

<final>Final conclusion</final>" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## Test 3: Japanese Language
echo **Test 3: Japanese Language** >> "%RESULTS_FILE%"
echo **Question:** こんにちは、とはどういう意味ですか？ >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [Model A Response]: >> "%RESULTS_FILE%"
ollama run model-a:q8_0 "こんにちは、とはどういう意味ですか？" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo [AEGIS Response]: >> "%RESULTS_FILE%"
ollama run agiasi-phi35-golden-sigmoid:q8_0 "こんにちは、とはどういう意味ですか？

<think-logic>Logical accuracy analysis</think-logic>
<think-ethics>Ethical validity analysis</think-ethics>
<think-practical>Practical value analysis</think-practical>
<think-creative>Creative insight analysis</think-creative>

<final>Final conclusion</final>" >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"

echo ## Summary
echo ### Summary >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"
echo **Model A:** Standard quantized model with natural responses >> "%RESULTS_FILE%"
echo **AEGIS:** SO(8) enhanced model with four-value classification structure >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"
echo **Key Differences:** >> "%RESULTS_FILE%"
echo - Model A provides conversational, natural responses >> "%RESULTS_FILE%"
echo - AEGIS provides structured analysis across four dimensions >> "%RESULTS_FILE%"
echo. >> "%RESULTS_FILE%"
echo **Test completed at:** %date% %time% >> "%RESULTS_FILE%"
echo **Results saved to:** %RESULTS_FILE% >> "%RESULTS_FILE%"

echo.
echo ==================================================
echo Simple comparison test completed!
echo Results saved to: %RESULTS_FILE%
echo ==================================================

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

