# 学習エラー監視スクリプト
# ログファイルを監視してエラーを検知

param(
    [string]$LogFile = "logs\train_borea_phi35_so8t_thinking.log",
    [int]$CheckInterval = 30
)

$ErrorPatterns = @(
    "RuntimeError",
    "element 0 of tensors",
    "does not require grad",
    "does not have a grad_fn",
    "Traceback",
    "Exception"
)

Write-Host "[MONITOR] Starting error monitoring for: $LogFile" -ForegroundColor Green
Write-Host "[MONITOR] Check interval: $CheckInterval seconds" -ForegroundColor Green
Write-Host ""

$lastPosition = 0
if (Test-Path $LogFile) {
    $lastPosition = (Get-Item $LogFile).Length
}

while ($true) {
    Start-Sleep -Seconds $CheckInterval
    
    if (Test-Path $LogFile) {
        $currentSize = (Get-Item $LogFile).Length
        
        if ($currentSize -gt $lastPosition) {
            # 新しいログを読み込む
            $stream = [System.IO.File]::Open($LogFile, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
            $stream.Position = $lastPosition
            $reader = New-Object System.IO.StreamReader($stream)
            
            while ($null -ne ($line = $reader.ReadLine())) {
                foreach ($pattern in $ErrorPatterns) {
                    if ($line -match $pattern) {
                        Write-Host "[ERROR DETECTED] $line" -ForegroundColor Red
                        # エラー周辺のコンテキストを取得
                        $context = Get-Content $LogFile -Tail 20 | Select-String -Pattern $pattern -Context 5
                        if ($context) {
                            Write-Host "[CONTEXT]" -ForegroundColor Yellow
                            Write-Host $context
                        }
                        # エラー通知を再生
                        $audioFile = "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
                        if (Test-Path $audioFile) {
                            try {
                                Add-Type -AssemblyName System.Windows.Forms
                                $player = New-Object System.Media.SoundPlayer $audioFile
                                $player.PlaySync()
                            } catch {
                                [System.Console]::Beep(1000, 500)
                            }
                        } else {
                            [System.Console]::Beep(1000, 500)
                        }
                    }
                }
            }
            
            $reader.Close()
            $stream.Close()
            $lastPosition = $currentSize
        }
    } else {
        Write-Host "[WARNING] Log file not found: $LogFile" -ForegroundColor Yellow
    }
}


