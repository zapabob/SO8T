@echo off
chcp 65001 >nul
echo [SO8T] Starting burn-in pipeline...
echo.

py -3 scripts/so8t_burnin_pipeline_rtx3060.py --hf-model models/Qwen2-VL-2B-Instruct --output-dir models/so8t_qwen2vl_2b_baked --quantization Q5_K_M --batch-size 1 --no-8bit

echo.
echo [SO8T] Pipeline completed!
echo [AUDIO] Playing completion notification...
py -3 scripts/test_audio.py

pause





