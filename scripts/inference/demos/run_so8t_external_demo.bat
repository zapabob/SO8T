@echo off
chcp 65001 >nul
echo ============================================================
echo  🚀 SO8T External Pipeline Demo 🚀
echo ============================================================
echo.
echo [INFO] Starting SO8T External Pipeline Demo...
echo.

echo [STEP 1] Running external SO8T pipeline demo...
py -3 scripts/demo_so8t_external.py

echo.
echo [STEP 2] Demo completed!
echo.

echo [AUDIO] Playing completion notification...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }"

echo.
echo ============================================================
echo  ✅ SO8T External Pipeline Demo Completed! ✅
echo ============================================================
pause
