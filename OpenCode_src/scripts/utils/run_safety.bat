@echo off
REM Safety-Aware SO8T Complete Pipeline Runner (Windows Batch)
REM CLIなしで学習推論実証を完全実行するバッチファイル

echo ================================================================================
echo 🚀 Safety-Aware SO8T Complete Pipeline Runner
echo    学習推論実証の完全実行システム
echo ================================================================================
echo.

REM Pythonの存在確認
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM 必要なファイルの確認
if not exist "train_safety.py" (
    echo ❌ train_safety.py not found
    pause
    exit /b 1
)

if not exist "configs\train_safety.yaml" (
    echo ❌ configs\train_safety.yaml not found
    pause
    exit /b 1
)

echo ✅ Required files found!
echo.

REM パイプライン実行
echo 🚀 Starting Safety-Aware SO8T Pipeline...
echo.

python run_safety_complete.py %*

if %errorlevel% equ 0 (
    echo.
    echo 🎉 Pipeline completed successfully!
    echo 📁 Check the output files for detailed results.
) else (
    echo.
    echo ❌ Pipeline failed. Check the error messages above.
)

echo.
pause
