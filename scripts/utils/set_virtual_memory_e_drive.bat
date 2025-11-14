@echo off
REM Set UTF-8 code page
chcp 65001 >nul 2>&1
timeout /t 1 >nul 2>&1

echo ========================================
echo E Drive Page File Configuration (320GB)
echo ========================================
echo.
echo [INFO] Setting virtual memory to 320GB on E: drive (adding 100GB)
echo [WARNING] Administrator privileges required
echo [WARNING] Please run as administrator
echo.
echo [INFO] Executing PowerShell script...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0set_virtual_memory_e_drive.ps1"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PowerShell script execution failed
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo [AUDIO] Playing completion notification...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0play_audio_notification.ps1"
pause
