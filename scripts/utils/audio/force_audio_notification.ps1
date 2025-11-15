# Force Audio Notification Script
# より確実な音声通知を試行

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Force Audio Notification Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Multiple beeps with different frequencies
Write-Host "[TEST 1] Multiple beeps (different frequencies)..." -ForegroundColor Yellow
try {
    Write-Host "[INFO] Playing beep sequence..." -ForegroundColor Cyan
    [System.Console]::Beep(500, 300)   # Low frequency
    Start-Sleep -Milliseconds 200
    [System.Console]::Beep(1000, 300)  # Medium frequency
    Start-Sleep -Milliseconds 200
    [System.Console]::Beep(1500, 300)  # High frequency
    Start-Sleep -Milliseconds 200
    [System.Console]::Beep(2000, 500)  # Very high frequency
    Write-Host "[OK] Beep sequence completed" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Beep failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 2: Windows Volume Control
Write-Host "[TEST 2] Checking Windows volume..." -ForegroundColor Yellow
try {
    # Get current volume
    Add-Type -TypeDefinition @"
    using System;
    using System.Runtime.InteropServices;
    public class VolumeControl {
        [DllImport("user32.dll")]
        public static extern void keybd_event(byte bVk, byte bScan, int dwFlags, int dwExtraInfo);
        public static void VolumeUp() {
            // VK_VOLUME_UP = 0xAF
            keybd_event(0xAF, 0, 0, 0);
            keybd_event(0xAF, 0, 2, 0);
        }
    }
"@
    Write-Host "[INFO] Current volume check..." -ForegroundColor Cyan
    Write-Host "[INFO] If volume is low, try increasing it manually" -ForegroundColor Yellow
} catch {
    Write-Host "[WARNING] Volume control check failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""

# Test 3: Visual notification (popup)
Write-Host "[TEST 3] Visual notification (popup)..." -ForegroundColor Yellow
try {
    Add-Type -AssemblyName System.Windows.Forms
    $result = [System.Windows.Forms.MessageBox]::Show(
        "Audio notification test completed.`nDid you hear the beeps?",
        "Audio Notification Test",
        [System.Windows.Forms.MessageBoxButtons]::YesNo,
        [System.Windows.Forms.MessageBoxIcon]::Question
    )
    if ($result -eq [System.Windows.Forms.DialogResult]::Yes) {
        Write-Host "[OK] User confirmed audio was heard" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] User did not hear audio" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[ERROR] Visual notification failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 4: Check audio device status
Write-Host "[TEST 4] Audio device status..." -ForegroundColor Yellow
try {
    $audioDevices = Get-PnpDevice -Class AudioEndpoint -Status OK
    if ($audioDevices) {
        Write-Host "[INFO] Audio devices found:" -ForegroundColor Cyan
        foreach ($device in $audioDevices) {
            Write-Host "  - $($device.FriendlyName)" -ForegroundColor Gray
        }
    } else {
        Write-Host "[WARNING] No audio devices found!" -ForegroundColor Red
    }
} catch {
    Write-Host "[WARNING] Failed to check audio devices: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""

# Test 5: Try playing a system sound
Write-Host "[TEST 5] System sound (Asterisk)..." -ForegroundColor Yellow
try {
    Add-Type -TypeDefinition @"
    using System;
    using System.Runtime.InteropServices;
    public class SystemSound {
        [DllImport("user32.dll")]
        public static extern bool MessageBeep(uint uType);
        public static void PlayAsterisk() {
            MessageBeep(0x00000040); // MB_ICONASTERISK
        }
        public static void PlayExclamation() {
            MessageBeep(0x00000030); // MB_ICONEXCLAMATION
        }
        public static void PlayQuestion() {
            MessageBeep(0x00000020); // MB_ICONQUESTION
        }
    }
"@
    Write-Host "[INFO] Playing system sounds..." -ForegroundColor Cyan
    [SystemSound]::PlayAsterisk()
    Start-Sleep -Milliseconds 500
    [SystemSound]::PlayExclamation()
    Start-Sleep -Milliseconds 500
    [SystemSound]::PlayQuestion()
    Write-Host "[OK] System sounds played" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] System sound failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Force audio test completed" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[IMPORTANT] If you still don't hear anything:" -ForegroundColor Red
Write-Host "  1. Check Windows volume settings" -ForegroundColor Yellow
Write-Host "  2. Check audio device connection" -ForegroundColor Yellow
Write-Host "  3. Try playing audio from another application" -ForegroundColor Yellow
Write-Host "  4. Check if audio is muted" -ForegroundColor Yellow

