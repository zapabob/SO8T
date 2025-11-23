@echo off
chcp 65001 >nul
echo [AUDIO] Testing audio playback with different method...

echo [STEP 1] Checking audio file...
if exist "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav" (
    echo [OK] Audio file found
) else (
    echo [ERROR] Audio file not found
    pause
    exit /b 1
)

echo [STEP 2] Testing system beep...
powershell -Command "[System.Console]::Beep(1000, 500)"
echo [OK] System beep test completed

echo [STEP 3] Testing WAV file playback...
powershell -Command "Add-Type -AssemblyName System.Windows.Forms; $player = New-Object System.Media.SoundPlayer 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav'; $player.Play(); Start-Sleep -Seconds 3"
echo [OK] WAV file playback attempted

echo [STEP 4] Testing with PlaySync...
powershell -Command "Add-Type -AssemblyName System.Windows.Forms; $player = New-Object System.Media.SoundPlayer 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav'; $player.PlaySync()"
echo [OK] PlaySync test completed

echo [AUDIO] All audio tests completed
pause
