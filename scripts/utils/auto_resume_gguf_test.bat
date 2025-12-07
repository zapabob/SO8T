@echo off
chcp 65001 >nul
echo [AUTO-RESUME] GGUF A/B Test Auto-Resume Script
echo ==============================================

cd /d "C:\Users\downl\Desktop\SO8T"

echo [CHECK] Checking for existing checkpoints...
if exist "results\ab_test_results\checkpoints\" (
    echo [FOUND] Checkpoints directory exists
    dir "results\ab_test_results\checkpoints\" /b /o:-d

    echo [RESUME] Resuming GGUF A/B test from checkpoint...
    py -3 scripts\evaluation\gguf_ab_test_llama_cpp.py

    echo [AUDIO] Playing completion notification...
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
) else (
    echo [START] No checkpoints found, starting fresh test...
    py -3 scripts\evaluation\gguf_ab_test_llama_cpp.py

    echo [AUDIO] Playing completion notification...
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
)

echo [COMPLETE] Auto-resume script finished
