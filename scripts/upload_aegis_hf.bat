@echo off
REM Upload AEGIS model to HuggingFace Hub (Windows batch script)

setlocal enabledelayedexpansion

REM Configuration
set REPO_NAME=%1
set HF_TOKEN=%2

if "%REPO_NAME%"=="" (
    echo ❌ Usage: %0 ^<repo-name^> [hf-token]
    echo Example: %0 your-username/AEGIS-Phi3.5-Enhanced
    exit /b 1
)

REM Check if huggingface-cli is available via Python
python -c "import huggingface_hub" >nul 2>&1
if errorlevel 1 (
    echo ❌ huggingface_hub not found. Install with: pip install huggingface_hub[cli]
    exit /b 1
)

REM Get token from environment if not provided
if "%HF_TOKEN%"=="" (
    set HF_TOKEN=%HF_TOKEN%
)

if "%HF_TOKEN%"=="" (
    echo ❌ HuggingFace token not found. Set HF_TOKEN environment variable or pass as second argument
    exit /b 1
)

echo 🚀 Starting AEGIS model upload to HuggingFace Hub
echo 📁 Repository: %REPO_NAME%
echo 📂 Upload directory: D:\webdataset\models\aegis-huggingface-upload

REM Set environment variable for token
set HF_TOKEN=%HF_TOKEN%

REM Create repository if it doesn't exist
echo 📝 Creating/checking repository...
python -c "from huggingface_hub import HfApi; api = HfApi(token='%HF_TOKEN%'); api.create_repo('%REPO_NAME%', repo_type='model', exist_ok=True)" || echo Repository already exists

REM Upload files using Python script
echo 📤 Uploading files using Python script...
python scripts\upload_aegis_to_huggingface.py "%REPO_NAME%" --token "%HF_TOKEN%"

echo ✅ Upload completed successfully!
echo 🌐 Model available at: https://huggingface.co/%REPO_NAME%

echo 🎉 AEGIS model upload completed!

pause
