# Audio System Check Script
# システムの音声設定を確認

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Audio System Check" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check audio devices
Write-Host "[CHECK 1] Audio Devices..." -ForegroundColor Yellow
try {
    Add-Type -TypeDefinition @"
    using System;
    using System.Runtime.InteropServices;
    public class AudioDevice {
        [DllImport("winmm.dll")]
        public static extern int waveOutGetNumDevs();
        public static int GetDeviceCount() {
            return waveOutGetNumDevs();
        }
    }
"@
    $deviceCount = [AudioDevice]::GetDeviceCount()
    Write-Host "[INFO] Audio devices found: $deviceCount" -ForegroundColor Green
    if ($deviceCount -eq 0) {
        Write-Host "[WARNING] No audio devices found!" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Failed to check audio devices: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test beep
Write-Host "[CHECK 2] System Beep Test..." -ForegroundColor Yellow
try {
    Write-Host "[INFO] Playing system beep (1000Hz, 500ms)..." -ForegroundColor Cyan
    [System.Console]::Beep(1000, 500)
    Write-Host "[OK] System beep played" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] System beep failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Check Windows volume
Write-Host "[CHECK 3] Windows Volume Settings..." -ForegroundColor Yellow
try {
    $audio = New-Object -ComObject WMPlayer.OCX
    $volume = $audio.settings.volume
    Write-Host "[INFO] Windows Media Player volume: $volume%" -ForegroundColor Cyan
    if ($volume -lt 50) {
        Write-Host "[WARNING] Volume is low: $volume%" -ForegroundColor Yellow
    }
    $audio.close()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($audio) | Out-Null
} catch {
    Write-Host "[ERROR] Failed to check volume: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Check audio file format
Write-Host "[CHECK 4] Audio File Format..." -ForegroundColor Yellow
$audioFile = "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
if (Test-Path $audioFile) {
    try {
        $bytes = [System.IO.File]::ReadAllBytes($audioFile)
        $header = [System.Text.Encoding]::ASCII.GetString($bytes[0..11])
        Write-Host "[INFO] File header: $header" -ForegroundColor Cyan
        
        if ($header -match "RIFF") {
            Write-Host "[OK] Valid WAV file (RIFF format)" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] File may not be a valid WAV file" -ForegroundColor Yellow
        }
        
        # Check file size
        $fileSize = (Get-Item $audioFile).Length
        Write-Host "[INFO] File size: $fileSize bytes" -ForegroundColor Cyan
        
    } catch {
        Write-Host "[ERROR] Failed to read file: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "[ERROR] Audio file not found: $audioFile" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "System check completed" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

