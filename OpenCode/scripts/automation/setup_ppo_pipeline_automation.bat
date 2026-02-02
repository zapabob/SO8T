@echo off
chcp 65001 >nul
echo [SO8T] Setting up PPO Pipeline Power-on Automation
echo =================================================

cd /d "%~dp0\..\.."

echo [STEP 1] Configuring power-on automation...
powershell -ExecutionPolicy Bypass -File "scripts\automation\setup_power_on_automation.ps1"

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Power-on automation configured successfully!
    echo [INFO] PPO Pipeline will now run automatically on system startup
    echo [AUDIO] Playing setup completion notification...
    powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
) else (
    echo [ERROR] Failed to configure power-on automation
    echo [AUDIO] Playing error notification...
    powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
)

echo [DONE] Automation setup completed
pause
