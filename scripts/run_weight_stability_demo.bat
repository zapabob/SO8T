@echo off
chcp 65001 >nul
echo ============================================================
echo  🛡️ SO8T Weight Stability Management Demo 🛡️
echo ============================================================
echo.

echo [INFO] Starting SO8T Weight Stability Management Demo...
echo.

echo [STEP 1] Running weight stability demo...
py -3 scripts/weight_stability_demo.py

echo.
echo [STEP 2] Demo completed!
echo.

echo [AUDIO] Playing completion notification...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }"

echo.
echo ============================================================
echo  SO8T Weight Stability Management Demo Finished! 🎉
echo ============================================================




