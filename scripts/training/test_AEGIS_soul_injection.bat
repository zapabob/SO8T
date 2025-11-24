@echo off
REM Simplified AEGIS Training Launch Script
REM Sets environment to bypass Protobuf issues

echo ============================================
echo   AEGIS Soul Injection - Simplified Launch
echo ============================================
echo.

cd /d "%~dp0..\.."

REM Set environment variable to use native Protobuf implementation
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo 🧪 Starting simplified soul injection (10 steps test)...
echo.

py scripts\training\inject_soul_into_borea.py ^
  --base-model "microsoft/Phi-3.5-mini-instruct" ^
  --max-steps 10 ^
  --warmup-steps 2 ^
  --annealing-steps 6 ^
  --batch-size 1 ^
  --max-length 256 ^
  --learning-rate 2e-4 ^
  --save-steps 5 ^
  --logging-steps 1

if %errorlevel% equ 0 (
    echo.
    echo ✅ Test run successful!
    echo 📊 Review checkpoints in: checkpoints_agiasi/
) else (
    echo.
    echo ❌ Training failed. Check error details above.
)

echo.
pause
