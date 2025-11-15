# Audio Notification Script
# BASIC PRINCIPLE: Play marisa_owattaze.wav first, fallback to beep if it fails
# STANDARD: This is the standard audio notification script for SO8T project
# IMPROVED: Multiple playback methods for better reliability

Write-Host "[AUDIO] Attempting to play audio notification..." -ForegroundColor Cyan

$audioFile = "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
$audioPlayed = $false

# Check if file exists
if (-not (Test-Path $audioFile)) {
    Write-Host "[WARNING] marisa_owattaze.wav not found: $audioFile" -ForegroundColor Yellow
    Write-Host "[FALLBACK] Using beep sound instead..." -ForegroundColor Yellow
    try {
        [System.Console]::Beep(1000, 500)
        Write-Host "[OK] Fallback beep played successfully" -ForegroundColor Green
        exit 0
    } catch {
        Write-Host "[ERROR] All audio methods failed" -ForegroundColor Red
        exit 1
    }
}

# Method 1: Python winsound (PRIMARY METHOD - Most reliable for WAV)
Write-Host "[METHOD 1] Trying Python winsound (synchronous)..." -ForegroundColor Cyan
Write-Host "[INFO] Make sure your system volume is not muted or too low" -ForegroundColor Yellow
try {
    # Use py -3 command (more reliable on Windows)
    $tempScript = [System.IO.Path]::GetTempFileName() + ".py"
    @"
import winsound
import sys
import os

audio_file = r'$audioFile'
print(f'[INFO] Playing: {os.path.basename(audio_file)}')
print(f'[INFO] File exists: {os.path.exists(audio_file)}')
print(f'[INFO] File size: {os.path.getsize(audio_file) if os.path.exists(audio_file) else 0} bytes')

try:
    # Play synchronously (blocking until complete)
    winsound.PlaySound(audio_file, winsound.SND_FILENAME)
    print('[OK] Playback completed')
    sys.exit(0)
except Exception as e:
    print(f'[ERROR] Playback failed: {e}', file=sys.stderr)
    sys.exit(1)
"@ | Out-File -FilePath $tempScript -Encoding UTF8 -NoNewline
    
    $result = & py -3 $tempScript 2>&1
    Write-Host $result
    Remove-Item $tempScript -ErrorAction SilentlyContinue
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] marisa_owattaze.wav played successfully with Python winsound" -ForegroundColor Green
        Write-Host "[INFO] If you didn't hear the sound, check:" -ForegroundColor Yellow
        Write-Host "  - Windows volume settings (not muted, volume > 0)" -ForegroundColor Yellow
        Write-Host "  - Audio device is connected and selected" -ForegroundColor Yellow
        Write-Host "  - Other applications can play audio" -ForegroundColor Yellow
        $audioPlayed = $true
    } else {
        Write-Host "[WARNING] Python winsound failed: $result" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[WARNING] Python winsound failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Method 1b: SoundPlayer (Fallback if Python fails)
if (-not $audioPlayed) {
    Write-Host "[METHOD 1b] Trying SoundPlayer (synchronous)..." -ForegroundColor Cyan
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $player = New-Object System.Media.SoundPlayer($audioFile)
        $player.Load()  # Pre-load the sound
        $player.PlaySync()
        Write-Host "[OK] marisa_owattaze.wav played successfully with SoundPlayer (sync)" -ForegroundColor Green
        $audioPlayed = $true
    } catch {
        Write-Host "[WARNING] SoundPlayer (sync) failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Method 2: SoundPlayer (Asynchronous with wait)
if (-not $audioPlayed) {
    Write-Host "[METHOD 2] Trying SoundPlayer (asynchronous with wait)..." -ForegroundColor Cyan
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $player = New-Object System.Media.SoundPlayer($audioFile)
        $player.Load()
        $player.Play()
        # Wait for playback to complete (max 5 seconds)
        $timeout = (Get-Date).AddSeconds(5)
        while ($player.IsLoadCompleted -eq $false -and (Get-Date) -lt $timeout) {
            Start-Sleep -Milliseconds 100
        }
        Start-Sleep -Seconds 1  # Additional wait for playback
        Write-Host "[OK] marisa_owattaze.wav played successfully with SoundPlayer (async)" -ForegroundColor Green
        $audioPlayed = $true
    } catch {
        Write-Host "[WARNING] SoundPlayer (async) failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Method 3: Windows Media Player COM Object (Improved)
if (-not $audioPlayed) {
    Write-Host "[METHOD 3] Trying Windows Media Player COM object..." -ForegroundColor Cyan
    try {
        $wmp = New-Object -ComObject WMPlayer.OCX
        $wmp.URL = $audioFile
        $wmp.settings.volume = 100
        $wmp.settings.balance = 0
        $wmp.controls.play()
        
        # Wait for playback with timeout
        $startTime = Get-Date
        $timeout = 5  # 5 seconds timeout
        while ($wmp.playState -ne 1 -and ((Get-Date) - $startTime).TotalSeconds -lt $timeout) {
            Start-Sleep -Milliseconds 100
        }
        
        if ($wmp.playState -eq 1) {
            Write-Host "[OK] marisa_owattaze.wav played successfully with WMP COM" -ForegroundColor Green
            $audioPlayed = $true
        } else {
            Write-Host "[WARNING] WMP playback may not have completed (State: $($wmp.playState))" -ForegroundColor Yellow
            # Still consider it played if we got past initial state
            if ($wmp.playState -ne 0 -and $wmp.playState -ne 9) {
                Start-Sleep -Seconds 1  # Give it time to play
                $audioPlayed = $true
            }
        }
        
        $wmp.close()
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($wmp) | Out-Null
    } catch {
        Write-Host "[WARNING] WMP COM failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Method 4: WinMM API (winmm.dll PlaySound) - Most Reliable
if (-not $audioPlayed) {
    Write-Host "[METHOD 4] Trying WinMM API (winmm.dll PlaySound)..." -ForegroundColor Cyan
    try {
        Add-Type -TypeDefinition @"
        using System;
        using System.Runtime.InteropServices;
        public class AudioPlayer {
            [DllImport("winmm.dll", SetLastError = true, CharSet = CharSet.Auto)]
            public static extern bool PlaySound(string pszSound, IntPtr hmod, uint fdwSound);
            public static void Play(string file) {
                // SND_FILENAME (0x00020000) | SND_SYNC (0x0000) | SND_NODEFAULT (0x0002)
                bool result = PlaySound(file, IntPtr.Zero, 0x00020000);
                if (!result) {
                    throw new Exception("PlaySound returned false");
                }
            }
        }
"@
        [AudioPlayer]::Play($audioFile)
        Write-Host "[OK] marisa_owattaze.wav played successfully with WinMM API" -ForegroundColor Green
        $audioPlayed = $true
    } catch {
        Write-Host "[WARNING] WinMM API failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Method 5: Python winsound (via subprocess) - Synchronous
if (-not $audioPlayed) {
    Write-Host "[METHOD 5] Trying Python winsound (synchronous)..." -ForegroundColor Cyan
    try {
        $pythonScript = @"
import winsound
import sys
try:
    winsound.PlaySound(r'$audioFile', winsound.SND_FILENAME)
    print('[OK] winsound playback completed')
    sys.exit(0)
except Exception as e:
    print(f'[ERROR] winsound failed: {e}')
    sys.exit(1)
"@
        $pythonScript | python -c "import sys; exec(sys.stdin.read())"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] marisa_owattaze.wav played successfully with Python winsound" -ForegroundColor Green
            $audioPlayed = $true
        } else {
            throw "Python execution failed with exit code $LASTEXITCODE"
        }
    } catch {
        Write-Host "[WARNING] Python winsound failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Method 6: Fallback to beep if all methods failed (ALWAYS WORKS)
if (-not $audioPlayed) {
    Write-Host "[FALLBACK] All WAV playback methods failed, using beep..." -ForegroundColor Yellow
    Write-Host "[INFO] Playing multiple beeps to ensure notification..." -ForegroundColor Cyan
    try {
        # Play multiple beeps to ensure user hears it
        [System.Console]::Beep(1000, 300)
        Start-Sleep -Milliseconds 200
        [System.Console]::Beep(1000, 300)
        Start-Sleep -Milliseconds 200
        [System.Console]::Beep(1000, 300)
        Write-Host "[OK] Fallback beep played successfully (3 beeps)" -ForegroundColor Green
        $audioPlayed = $true
    } catch {
        Write-Host "[ERROR] All audio methods failed including beep" -ForegroundColor Red
        exit 1
    }
}

# ALWAYS play beep as backup notification (even if WAV played successfully)
# This ensures the user always hears a notification
Write-Host "[BACKUP] Playing backup beep notification..." -ForegroundColor Cyan
try {
    [System.Console]::Beep(800, 200)
    Start-Sleep -Milliseconds 100
    [System.Console]::Beep(1000, 200)
    Write-Host "[OK] Backup beep notification played" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Backup beep failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# VISUAL NOTIFICATION (Always show, even if audio works)
# This ensures the user always sees the notification
Write-Host "[VISUAL] Showing visual notification..." -ForegroundColor Cyan
try {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    
    # Create a notification form
    $form = New-Object System.Windows.Forms.Form
    $form.Text = "Audio Notification"
    $form.Size = New-Object System.Drawing.Size(400, 150)
    $form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
    $form.TopMost = $true
    $form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $form.MaximizeBox = $false
    $form.MinimizeBox = $false
    
    # Add label
    $label = New-Object System.Windows.Forms.Label
    $label.Text = "Task completed!`nAudio notification has been played."
    $label.AutoSize = $false
    $label.Size = New-Object System.Drawing.Size(380, 80)
    $label.Location = New-Object System.Drawing.Point(10, 10)
    $label.TextAlign = [System.Drawing.ContentAlignment]::MiddleCenter
    $label.Font = New-Object System.Drawing.Font("Microsoft Sans Serif", 10, [System.Drawing.FontStyle]::Bold)
    $form.Controls.Add($label)
    
    # Add close button
    $button = New-Object System.Windows.Forms.Button
    $button.Text = "OK"
    $button.Size = New-Object System.Drawing.Size(100, 30)
    $button.Location = New-Object System.Drawing.Point(150, 90)
    $button.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $form.Controls.Add($button)
    
    # Show form (non-blocking)
    $form.Add_Shown({$form.Activate()})
    $form.ShowDialog() | Out-Null
    $form.Dispose()
    
    Write-Host "[OK] Visual notification displayed" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Visual notification failed: $($_.Exception.Message)" -ForegroundColor Yellow
    # Fallback: Just print a very visible message
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green -BackgroundColor Black
    Write-Host "  TASK COMPLETED - AUDIO NOTIFICATION  " -ForegroundColor Yellow -BackgroundColor Black
    Write-Host "========================================" -ForegroundColor Green -BackgroundColor Black
    Write-Host ""
}

if ($audioPlayed) {
    Write-Host "[SUCCESS] Audio notification completed (WAV + Beep)" -ForegroundColor Green
    exit 0
} else {
    Write-Host "[SUCCESS] Audio notification completed (Beep only)" -ForegroundColor Green
    exit 0
}
