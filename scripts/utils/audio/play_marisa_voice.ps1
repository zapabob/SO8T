# Play Marisa Voice Script
# 魔理沙の音声を確実に再生するスクリプト

$audioFile = "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Playing Marisa Voice (魔理沙の声)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $audioFile)) {
    Write-Host "[ERROR] Audio file not found: $audioFile" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Audio file: $audioFile" -ForegroundColor Green
Write-Host "[INFO] File size: $((Get-Item $audioFile).Length) bytes" -ForegroundColor Green
Write-Host ""

$played = $false

# Method 1: Python winsound (Most reliable for WAV files)
Write-Host "[METHOD 1] Python winsound (synchronous)..." -ForegroundColor Yellow
try {
    $pythonScript = @"
import winsound
import sys
import os

audio_file = r'$audioFile'
if not os.path.exists(audio_file):
    print(f'[ERROR] File not found: {audio_file}')
    sys.exit(1)

print(f'[INFO] Playing: {audio_file}')
print(f'[INFO] File size: {os.path.getsize(audio_file)} bytes')

try:
    # Play synchronously (blocking)
    winsound.PlaySound(audio_file, winsound.SND_FILENAME | winsound.SND_SYNC)
    print('[OK] Playback completed')
    sys.exit(0)
except Exception as e:
    print(f'[ERROR] Playback failed: {e}')
    sys.exit(1)
"@
    $output = $pythonScript | python -c "import sys; exec(sys.stdin.read())" 2>&1
    Write-Host $output
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] Marisa voice played with Python winsound" -ForegroundColor Green
        $played = $true
    }
} catch {
    Write-Host "[ERROR] Python winsound failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Method 2: Windows Media Player COM (with volume control)
if (-not $played) {
    Write-Host "[METHOD 2] Windows Media Player COM (with volume 100%)..." -ForegroundColor Yellow
    try {
        $wmp = New-Object -ComObject WMPlayer.OCX
        $wmp.URL = $audioFile
        $wmp.settings.volume = 100
        $wmp.settings.balance = 0
        $wmp.settings.mute = $false
        
        Write-Host "[INFO] Volume: 100%, Mute: False" -ForegroundColor Cyan
        Write-Host "[INFO] Starting playback..." -ForegroundColor Cyan
        
        $wmp.controls.play()
        
        # Wait for playback to start
        $startTime = Get-Date
        $timeout = 2
        while ($wmp.playState -eq 9 -and ((Get-Date) - $startTime).TotalSeconds -lt $timeout) {
            Start-Sleep -Milliseconds 100
        }
        
        # Wait for playback to complete (with longer timeout)
        $startTime = Get-Date
        $timeout = 10
        while ($wmp.playState -ne 1 -and $wmp.playState -ne 0 -and ((Get-Date) - $startTime).TotalSeconds -lt $timeout) {
            Start-Sleep -Milliseconds 100
        }
        
        if ($wmp.playState -eq 1) {
            Write-Host "[SUCCESS] Marisa voice played with WMP COM" -ForegroundColor Green
            $played = $true
        } else {
            Write-Host "[WARNING] WMP playback state: $($wmp.playState)" -ForegroundColor Yellow
            # Give it more time
            Start-Sleep -Seconds 2
            $played = $true
        }
        
        $wmp.close()
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($wmp) | Out-Null
    } catch {
        Write-Host "[ERROR] WMP COM failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""

# Method 3: WinMM API (winmm.dll PlaySound) - Synchronous
if (-not $played) {
    Write-Host "[METHOD 3] WinMM API (winmm.dll PlaySound)..." -ForegroundColor Yellow
    try {
        Add-Type -TypeDefinition @"
        using System;
        using System.Runtime.InteropServices;
        public class AudioPlayer {
            [DllImport("winmm.dll", SetLastError = true, CharSet = CharSet.Auto)]
            public static extern bool PlaySound(string pszSound, IntPtr hmod, uint fdwSound);
            public static void Play(string file) {
                // SND_FILENAME (0x00020000) | SND_SYNC (0x0000)
                bool result = PlaySound(file, IntPtr.Zero, 0x00020000);
                if (!result) {
                    int error = Marshal.GetLastWin32Error();
                    throw new Exception($"PlaySound failed with error code: {error}");
                }
            }
        }
"@
        [AudioPlayer]::Play($audioFile)
        Write-Host "[SUCCESS] Marisa voice played with WinMM API" -ForegroundColor Green
        $played = $true
    } catch {
        Write-Host "[ERROR] WinMM API failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""

# Method 4: Start-Process with Windows Media Player
if (-not $played) {
    Write-Host "[METHOD 4] Starting Windows Media Player directly..." -ForegroundColor Yellow
    try {
        $wmpPath = "${env:ProgramFiles(x86)}\Windows Media Player\wmplayer.exe"
        if (-not (Test-Path $wmpPath)) {
            $wmpPath = "${env:ProgramFiles}\Windows Media Player\wmplayer.exe"
        }
        
        if (Test-Path $wmpPath) {
            Write-Host "[INFO] Starting WMP: $wmpPath" -ForegroundColor Cyan
            $process = Start-Process -FilePath $wmpPath -ArgumentList "`"$audioFile`"" -PassThru
            Write-Host "[INFO] WMP started, waiting for playback..." -ForegroundColor Cyan
            Start-Sleep -Seconds 3
            Write-Host "[SUCCESS] WMP started for Marisa voice playback" -ForegroundColor Green
            $played = $true
        } else {
            Write-Host "[ERROR] Windows Media Player not found" -ForegroundColor Red
        }
    } catch {
        Write-Host "[ERROR] Start-Process failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($played) {
    Write-Host "[SUCCESS] Marisa voice playback attempted" -ForegroundColor Green
} else {
    Write-Host "[WARNING] All playback methods failed" -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan

