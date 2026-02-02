@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo SO8T Complete System Demo
echo ============================================================
echo.

echo [STEP 1] Checking Python environment...
py -3 --version
if errorlevel 1 (
    echo [ERROR] Python not found
    goto :error
)
echo [OK] Python environment ready
echo.

echo [STEP 2] Checking dependencies...
py -3 -c "import torch; import cv2; import pytesseract; import sqlite3; print('[OK] All dependencies available')"
if errorlevel 1 (
    echo [WARNING] Some dependencies missing
    echo [INFO] Continuing anyway...
)
echo.

echo [STEP 3] Running SO8T Complete System Demo...
echo.
py -3 scripts/demo_complete_so8t_system.py
if errorlevel 1 (
    echo.
    echo [ERROR] Demo failed
    goto :error
)

echo.
echo ============================================================
echo Demo Completed Successfully!
echo ============================================================
echo.
echo [INFO] Check the following databases for logs:
echo   - database/so8t_memory.db (Conversation history)
echo   - database/so8t_compliance.db (Compliance logs)
echo.

goto :success

:error
echo.
echo ============================================================
echo Demo Failed!
echo ============================================================
echo.
echo [AUDIO] Playing error notification...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; $player = New-Object System.Media.SoundPlayer 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav'; $player.PlaySync(); Write-Host '[OK] Audio notification played' -ForegroundColor Red } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }"
exit /b 1

:success
echo [AUDIO] Playing success notification...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; $player = New-Object System.Media.SoundPlayer 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav'; $player.PlaySync(); Write-Host '[OK] Audio notification played' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }"
exit /b 0

