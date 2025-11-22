@echo off
chcp 65001 >nul
echo [SO8T] Starting SO8T/thinking QLoRA Fine-tuning
echo ===============================================

set BASE_MODEL=models/Borea-Phi-3.5-mini-Instruct-Jp
set OUTPUT_DIR=D:/webdataset/checkpoints/so8t_thinking_qlora
set TRAIN_DATA=data/nkat_so8t_v2/train_nkat_so8t.jsonl
set MAX_STEPS=500
set BATCH_SIZE=1
set GRAD_ACCUM=8
set LEARNING_RATE=2e-5
set NUM_SO8T_LAYERS=4
set SO8T_HIDDEN_SIZE=2048

echo [CONFIG] Base model: %BASE_MODEL%
echo [CONFIG] Output: %OUTPUT_DIR%
echo [CONFIG] Training data: %TRAIN_DATA%
echo [CONFIG] Max steps: %MAX_STEPS%
echo [CONFIG] Effective batch size: %BATCH_SIZE% x %GRAD_ACCUM% = %BATCH_SIZE%*%GRAD_ACCUM%
echo [CONFIG] SO8T layers: %NUM_SO8T_LAYERS%
echo [CONFIG] SO8T hidden size: %SO8T_HIDDEN_SIZE%
echo [CONFIG] QLoRA: Enabled (base model frozen)
echo.

echo [TRAINING] Starting SO8T/thinking QLoRA fine-tuning...
py -3 scripts/training/train_so8t_thinking_model.py ^
    --base-model "%BASE_MODEL%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --train-data "%TRAIN_DATA%" ^
    --max-steps %MAX_STEPS% ^
    --batch-size %BATCH_SIZE% ^
    --gradient-accumulation %GRAD_ACCUM% ^
    --learning-rate %LEARNING_RATE% ^
    --num-so8t-layers %NUM_SO8T_LAYERS% ^
    --so8t-hidden-size %SO8T_HIDDEN_SIZE%

echo [SUCCESS] SO8T/thinking QLoRA training completed!
echo [OUTPUT] Model saved to: %OUTPUT_DIR%\final_model
echo.
echo [INFERENCE] To load the model for inference:
echo   from peft import PeftModel
echo   model = PeftModel.from_pretrained('%BASE_MODEL%', '%OUTPUT_DIR%\final_model')
echo.

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

pause





