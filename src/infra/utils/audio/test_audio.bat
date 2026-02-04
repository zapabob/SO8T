@echo off
chcp 65001 >nul
echo [AUDIO] Testing audio playback...
if exist "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav" (
    echo [OK] Audio file found
    powershell -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play()"
    echo [OK] Audio playback command executed
) else (
    echo [WARNING] Audio file not found
)
echo [AUDIO] Test completed
pause
