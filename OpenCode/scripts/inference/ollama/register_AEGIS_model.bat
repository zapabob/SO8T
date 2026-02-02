@echo off
REM Register AEGIS Borea-Phi3.5-SO8T-Thinking model to Ollama

echo 🤖 Registering AEGIS Borea-Phi3.5-SO8T-Thinking model to Ollama...

cd /d "%~dp0..\..\..\"

REM Check if GGUF file exists
if not exist "D:\webdataset\gguf_models\borea_phi35_so8t_thinking\borea_phi35_so8t_thinking_Q8_0.gguf" (
    echo ❌ Error: GGUF file not found!
    echo Expected path: D:\webdataset\gguf_models\borea_phi35_so8t_thinking\borea_phi35_so8t_thinking_Q8_0.gguf
    echo.
    echo Please run the GGUF conversion pipeline first.
    pause
    exit /b 1
)

REM Register model with Ollama
echo 📝 Creating Ollama model from Modelfile...
ollama create borea-phi35-so8t-thinking -f modelfiles\borea_phi35_so8t_thinking.modelfile

if %errorlevel% equ 0 (
    echo ✅ Model registered successfully!
    echo.
    echo 🚀 You can now use the model:
    echo    ollama run borea-phi35-so8t-thinking "こんにちは"
) else (
    echo ❌ Failed to register model
    exit /b 1
)

echo.
pause
