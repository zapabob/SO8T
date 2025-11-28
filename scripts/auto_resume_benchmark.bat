@echo off
REM Automatic Benchmark Resume Script for Windows Startup
REM This script runs on Windows startup to automatically resume benchmark testing

echo [AUTO RESUME] Starting automatic benchmark resume at %DATE% %TIME%

REM Wait for system to fully boot (30 seconds)
timeout /t 30 /nobreak > nul

REM Change to project directory
cd /d "C:\Users\downl\Desktop\SO8T"

REM Check if Ollama is running, start if not
echo [AUTO RESUME] Checking Ollama service...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if %ERRORLEVEL% NEQ 0 (
    echo [AUTO RESUME] Starting Ollama service...
    start "" "C:\Users\downl\AppData\Local\Programs\Ollama\ollama.exe" serve
    timeout /t 10 /nobreak > nul
) else (
    echo [AUTO RESUME] Ollama is already running
)

REM Check for existing checkpoints
if exist "D:\webdataset\benchmark_results\checkpoints" (
    echo [AUTO RESUME] Found existing checkpoints, resuming benchmark...
    REM Run the benchmark script (it will automatically resume from checkpoint)
    py -3 scripts/comprehensive_ab_benchmark.py
) else (
    echo [AUTO RESUME] No checkpoints found, starting fresh benchmark...
    py -3 scripts/comprehensive_ab_benchmark.py --test-mode
)

REM Play completion sound
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
        Write-Host '[WARNING] Failed to play marisa_owattaze.wav' -ForegroundColor Yellow
    }
}
if (-not $audioPlayed) {
    try {
        [System.Console]::Beep(1000, 500)
        Write-Host '[OK] Fallback beep played successfully' -ForegroundColor Green
    } catch {
        Write-Host '[ERROR] All audio methods failed' -ForegroundColor Red
    }
}
"

echo [AUTO RESUME] Benchmark resume process completed at %DATE% %TIME%
pause































