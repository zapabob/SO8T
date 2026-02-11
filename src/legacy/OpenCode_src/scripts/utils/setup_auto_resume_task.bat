@echo off
chcp 65001 >nul
echo [SETUP] Setting up automatic GGUF A/B test resume on startup
echo ========================================================

REM ============================================================
REM AUTOSTART SETUP DISABLED (default)
REM To enable creation of startup tasks, run with:
REM   set SO8T_ENABLE_AUTOSTART=1 && setup_auto_resume_task.bat
REM ============================================================
if /I NOT "%SO8T_ENABLE_AUTOSTART%"=="1" (
    echo [INFO] Auto-resume task setup is disabled by default.
    echo [INFO] Set SO8T_ENABLE_AUTOSTART=1 to intentionally enable.
    exit /b 0
)

set TASK_NAME="SO8T_GGUF_AB_Test_Auto_Resume"
set SCRIPT_PATH="C:\Users\downl\Desktop\SO8T\scripts\utils\auto_resume_gguf_test.bat"

echo [CHECK] Checking administrator privileges...
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Administrator privileges confirmed
) else (
    echo [ERROR] Administrator privileges required
    echo Please run this script as administrator
    pause
    exit /b 1
)

echo [TASK] Creating/Updating Windows Task Scheduler entry...
echo Task Name: %TASK_NAME%
echo Script Path: %SCRIPT_PATH%

schtasks /create /tn %TASK_NAME% /tr "%SCRIPT_PATH%" /sc onlogon /rl highest /f

if %errorLevel% == 0 (
    echo [SUCCESS] Task created successfully
    echo The GGUF A/B test will now automatically resume on system startup
) else (
    echo [ERROR] Failed to create task
    echo Error code: %errorLevel%
)

echo [VERIFY] Verifying task creation...
schtasks /query /tn %TASK_NAME%

echo [AUDIO] Playing setup completion notification...
powershell -ExecutionPolicy Bypass -Command "
Write-Host '[AUDIO] Playing completion notification...' -ForegroundColor Green
$audioFile = 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav'
if (Test-Path $audioFile) {
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $player = New-Object System.Media.SoundPlayer $audioFile
        $player.PlaySync()
        Write-Host '[OK] marisa_owattaze.wav played successfully' -ForegroundColor Green
    } catch {
        Write-Host '[WARNING] Failed to play audio' -ForegroundColor Yellow
    }
} else {
    Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow
}
"

echo [COMPLETE] Setup finished
echo The system will now automatically resume GGUF A/B testing on startup
pause