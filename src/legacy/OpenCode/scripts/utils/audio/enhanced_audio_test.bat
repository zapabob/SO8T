@echo off
chcp 65001 >nul
echo [AUDIO] Enhanced audio test with multiple fallback methods...

echo [METHOD 1] Testing SoundPlayer with PlaySync...
powershell -Command "Write-Host '[AUDIO] Attempting to play audio notification...' -ForegroundColor Green; if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { try { Add-Type -AssemblyName System.Windows.Forms; $player = New-Object System.Media.SoundPlayer 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav'; $player.PlaySync(); Write-Host '[OK] Audio played successfully with SoundPlayer' -ForegroundColor Green } catch { Write-Host '[WARNING] SoundPlayer failed' -ForegroundColor Yellow } } else { Write-Host '[ERROR] Audio file not found' -ForegroundColor Red }"

echo [METHOD 2] Testing system beep as fallback...
powershell -Command "[System.Console]::Beep(1000, 500); Write-Host '[OK] Fallback beep played successfully' -ForegroundColor Green"

echo [METHOD 3] Testing emergency beep...
powershell -Command "[System.Console]::Beep(800, 1000); Write-Host '[OK] Emergency beep played' -ForegroundColor Green"

echo [AUDIO] Enhanced audio test completed
echo [NOTE] If you still cannot hear audio, check:
echo - Volume settings and audio device selection
echo - Windows audio service status
echo - Hardware connections
pause
