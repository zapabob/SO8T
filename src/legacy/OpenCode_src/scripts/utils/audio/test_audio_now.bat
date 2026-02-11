@echo off
chcp 65001 >nul
echo [AUDIO] Testing audio playback now...

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

echo [STEP 3] Testing WAV file with PlaySync...
powershell -Command "Add-Type -AssemblyName System.Windows.Forms; $player = New-Object System.Media.SoundPlayer 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav'; $player.PlaySync()"
echo [OK] WAV file PlaySync test completed

echo [STEP 4] Testing WAV file with Play...
powershell -Command "Add-Type -AssemblyName System.Windows.Forms; $player = New-Object System.Media.SoundPlayer 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav'; $player.Play(); Start-Sleep -Seconds 3"
echo [OK] WAV file Play test completed

echo [AUDIO] All audio tests completed
echo [NOTE] If you still cannot hear audio, the issue might be:
echo - Volume is too low or muted
echo - Wrong audio device selected
echo - Audio service not running
echo - Hardware connection issue
pause
