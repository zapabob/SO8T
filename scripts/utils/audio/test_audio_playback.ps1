# Audio Playback Test Script
# 複数の方法で音声再生をテスト

$audioFile = "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Audio Playback Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check file
if (-not (Test-Path $audioFile)) {
    Write-Host "[ERROR] Audio file not found: $audioFile" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Audio file found: $audioFile" -ForegroundColor Green
Write-Host "[INFO] File size: $((Get-Item $audioFile).Length) bytes" -ForegroundColor Green
Write-Host ""

# Test 1: Windows Media Player COM (Most Reliable)
Write-Host "[TEST 1] Windows Media Player COM Object..." -ForegroundColor Yellow
try {
    $wmp = New-Object -ComObject WMPlayer.OCX
    $wmp.URL = $audioFile
    $wmp.settings.volume = 100
    $wmp.settings.balance = 0
    Write-Host "[INFO] Volume set to 100%, playing..." -ForegroundColor Cyan
    $wmp.controls.play()
    
    # Wait for playback
    $startTime = Get-Date
    $timeout = 10  # 10 seconds timeout
    while ($wmp.playState -ne 1 -and ((Get-Date) - $startTime).TotalSeconds -lt $timeout) {
        Start-Sleep -Milliseconds 100
        $currentState = $wmp.playState
        Write-Host "[DEBUG] PlayState: $currentState" -ForegroundColor Gray
    }
    
    if ($wmp.playState -eq 1) {
        Write-Host "[OK] WMP playback completed successfully" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] WMP playback timeout or incomplete" -ForegroundColor Yellow
    }
    
    $wmp.close()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($wmp) | Out-Null
    Write-Host "[SUCCESS] Test 1 completed" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] WMP COM failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Start-Sleep -Seconds 1

# Test 2: Python winsound (Synchronous)
Write-Host "[TEST 2] Python winsound (Synchronous)..." -ForegroundColor Yellow
try {
    $pythonScript = @"
import winsound
import sys
print('[INFO] Playing with winsound (sync)...')
winsound.PlaySound(r'$audioFile', winsound.SND_FILENAME)
print('[OK] winsound playback completed')
"@
    $pythonScript | python -c "import sys; exec(sys.stdin.read())"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] Test 2 completed" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Python execution failed" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Python winsound failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Start-Sleep -Seconds 1

# Test 3: Start-Process with Windows Media Player
Write-Host "[TEST 3] Start-Process with Windows Media Player..." -ForegroundColor Yellow
try {
    $wmpPath = "${env:ProgramFiles(x86)}\Windows Media Player\wmplayer.exe"
    if (-not (Test-Path $wmpPath)) {
        $wmpPath = "${env:ProgramFiles}\Windows Media Player\wmplayer.exe"
    }
    
    if (Test-Path $wmpPath) {
        Write-Host "[INFO] Starting Windows Media Player..." -ForegroundColor Cyan
        $process = Start-Process -FilePath $wmpPath -ArgumentList "`"$audioFile`"" -PassThru -WindowStyle Minimized
        Start-Sleep -Seconds 3
        if (-not $process.HasExited) {
            Write-Host "[INFO] WMP started, waiting for playback..." -ForegroundColor Cyan
            Start-Sleep -Seconds 2
            $process.CloseMainWindow()
            Start-Sleep -Seconds 1
            if (-not $process.HasExited) {
                $process.Kill()
            }
        }
        Write-Host "[SUCCESS] Test 3 completed" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Windows Media Player not found" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[ERROR] Start-Process failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All tests completed" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

