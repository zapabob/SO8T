# 四値分類トレーニング自動復旧スクリプト
# システム再起動時に自動でチェックポイントからトレーニングを再開する

param(
    [string]$ConfigPath = "configs/train_four_class.yaml",
    [string]$LogFile = "logs/auto_resume_four_class.log",
    [int]$CheckIntervalSeconds = 60,
    [switch]$NoAudio
)

# ログ関数
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Write-Host $logMessage -ForegroundColor $(if ($Level -eq "ERROR") { "Red" } elseif ($Level -eq "WARNING") { "Yellow" } else { "Green" })

    # ログファイルに書き込み
    try {
        $logMessage | Out-File -FilePath $LogFile -Append -Encoding UTF8
    } catch {
        Write-Host "ログファイル書き込みエラー: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# トレーニングプロセスが実行中かチェック
function Test-TrainingRunning {
    $pythonProcesses = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -like "*train_four_class_classifier*"
    }
    return $pythonProcesses.Count -gt 0
}

# 最新のチェックポイントを取得
function Get-LatestCheckpoint {
    param([string]$OutputDir = "D:/webdataset/checkpoints/training/borea_phi35_four_class")

    if (!(Test-Path $OutputDir)) {
        return $null
    }

    # checkpoint-* ディレクトリを検索
    $checkpointDirs = Get-ChildItem -Path $OutputDir -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue

    if ($checkpointDirs.Count -eq 0) {
        return $null
    }

    # ステップ番号でソートして最新のものを選択
    $checkpointDirs = $checkpointDirs | Sort-Object {
        $stepNum = $_.Name -replace "checkpoint-", ""
        if ($stepNum -match "^\d+$") { [int]$stepNum } else { 0 }
    }

    $latestCheckpoint = $checkpointDirs[-1]
    Write-Log "Found latest checkpoint: $($latestCheckpoint.FullName)"
    return $latestCheckpoint.FullName
}

# トレーニングを開始
function Start-Training {
    param([string]$CheckpointPath = $null)

    $command = "py -3 scripts/training/train_four_class_classifier.py --config $ConfigPath"

    if ($CheckpointPath) {
        $command += " --resume-from-checkpoint `"$CheckpointPath`""
    }

    Write-Log "Starting training: $command"

    try {
        # バックグラウンドで実行
        Start-Process -FilePath "powershell.exe" -ArgumentList "-Command $command" -NoNewWindow
        Write-Log "Training started successfully"
        return $true
    } catch {
        Write-Log "Failed to start training: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# オーディオ通知を再生
function Play-AudioNotification {
    if ($NoAudio) { return }

    $audioFile = "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
    if (Test-Path $audioFile) {
        try {
            Add-Type -AssemblyName System.Windows.Forms
            $player = New-Object System.Media.SoundPlayer $audioFile
            $player.PlaySync()
            Write-Log "Audio notification played"
        } catch {
            Write-Log "Failed to play audio: $($_.Exception.Message)" "WARNING"
            # フォールバックとしてビープ音
            [System.Console]::Beep(1000, 500)
        }
    } else {
        Write-Log "Audio file not found: $audioFile" "WARNING"
        [System.Console]::Beep(1000, 500)
    }
}

# メイン処理
function Main {
    Write-Log "=== 四値分類トレーニング自動復旧システム起動 ==="
    Write-Log "Config: $ConfigPath"
    Write-Log "Check Interval: ${CheckIntervalSeconds}秒"

    # 初回チェック
    if (Test-TrainingRunning) {
        Write-Log "トレーニングが既に実行中です"
        return
    }

    # 最新チェックポイントを取得
    $latestCheckpoint = Get-LatestCheckpoint

    if ($latestCheckpoint) {
        Write-Log "チェックポイントから再開します: $latestCheckpoint"
    } else {
        Write-Log "新規トレーニングを開始します"
    }

    # トレーニングを開始
    if (Start-Training -CheckpointPath $latestCheckpoint) {
        Play-AudioNotification
        Write-Log "トレーニング開始完了"

        # 定期的に状態を監視
        while ($true) {
            Start-Sleep -Seconds $CheckIntervalSeconds

            if (!(Test-TrainingRunning)) {
                Write-Log "トレーニングプロセスが終了しました" "WARNING"
                Play-AudioNotification

                # 再チェックポイントを取得して再開を試行
                $newCheckpoint = Get-LatestCheckpoint
                if ($newCheckpoint -and $newCheckpoint -ne $latestCheckpoint) {
                    Write-Log "新しいチェックポイントが見つかりました。再開を試行します"
                    if (Start-Training -CheckpointPath $newCheckpoint) {
                        $latestCheckpoint = $newCheckpoint
                        Play-AudioNotification
                    }
                } else {
                    Write-Log "再開可能なチェックポイントが見つかりません"
                    break
                }
            } else {
                Write-Log "トレーニング正常実行中"
            }
        }
    } else {
        Write-Log "トレーニング開始に失敗しました" "ERROR"
    }
}

# スクリプト実行
try {
    Main
} catch {
    Write-Log "予期せぬエラー: $($_.Exception.Message)" "ERROR"
    Play-AudioNotification
} finally {
    Write-Log "=== 四値分類トレーニング自動復旧システム終了 ==="
}



