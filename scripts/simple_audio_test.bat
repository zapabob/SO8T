@echo off
chcp 65001 >nul
echo [AUDIO] Simple audio test starting...

echo [TEST 1] System beep test
powershell -Command "[System.Console]::Beep(800, 1000)"
echo [OK] Beep test completed

echo [TEST 2] WAV file test
powershell -Command "Add-Type -AssemblyName System.Windows.Forms; $player = New-Object System.Media.SoundPlayer 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav'; $player.PlaySync()"
echo [OK] WAV file test completed

echo [AUDIO] All tests completed
echo [NOTE] If you cannot hear audio, check:
echo - Volume settings
echo - Audio device selection
echo - Speaker/headphone connection
echo - Windows audio service status
pause
