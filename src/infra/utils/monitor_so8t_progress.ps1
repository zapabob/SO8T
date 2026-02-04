# SO8Tプロジェクト進捗監視スクリプト
# 各コンポーネントの進捗状況をリアルタイムで監視

param(
    [switch]$Continuous,
    [int]$IntervalSeconds = 30,
    [switch]$Quiet
)

# UTF-8エンコーディング設定
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ワークツリー名取得
function Get-WorktreeName {
    try {
        $gitDir = git rev-parse --git-dir 2>$null
        if ($gitDir -and ($gitDir -like "*worktrees*")) {
            $parts = $gitDir -split "\\"
            $worktreeIndex = $parts.IndexOf("worktrees")
            if ($worktreeIndex -ge 0 -and $worktreeIndex -lt ($parts.Length - 1)) {
                return $parts[$worktreeIndex + 1]
            }
        }
        return "main"
    } catch {
        return "main"
    }
}

# 進捗状況取得関数群
function Get-TrainingProgress {
    $logFiles = Get-ChildItem "logs\train_*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending

    if ($logFiles.Count -eq 0) {
        return @{
            Status = "なし"
            LastUpdate = "N/A"
            CurrentEpoch = "N/A"
            TotalEpochs = "N/A"
            Loss = "N/A"
            Progress = 0
        }
    }

    $latestLog = $logFiles[0]
    $content = Get-Content $latestLog.FullName -Tail 50 -Encoding UTF8 -ErrorAction SilentlyContinue

    # 最新のトレーニング情報を解析
    $epochMatch = $content | Select-String "Epoch (\d+)/(\d+)" | Select-Object -Last 1
    $lossMatch = $content | Select-String "loss[:=]\s*([\d.]+)" | Select-Object -Last 1

    $currentEpoch = "N/A"
    $totalEpochs = "N/A"
    $loss = "N/A"
    $progress = 0

    if ($epochMatch) {
        $currentEpoch = $epochMatch.Matches[0].Groups[1].Value
        $totalEpochs = $epochMatch.Matches[0].Groups[2].Value
        if ($totalEpochs -ne "N/A" -and $totalEpochs -ne "0") {
            $progress = [math]::Round(([int]$currentEpoch / [int]$totalEpochs) * 100, 1)
        }
    }

    if ($lossMatch) {
        $loss = $lossMatch.Matches[0].Groups[1].Value
    }

    return @{
        Status = "実行中"
        LastUpdate = $latestLog.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        CurrentEpoch = $currentEpoch
        TotalEpochs = $totalEpochs
        Loss = $loss
        Progress = $progress
        LogFile = $latestLog.Name
    }
}

function Get-DatasetProgress {
    $datasetDir = "D:\webdataset\datasets"
    $cleanedDir = "D:\webdataset\cleaned"

    $datasets = @()
    if (Test-Path $datasetDir) {
        $datasets = Get-ChildItem $datasetDir -Directory -ErrorAction SilentlyContinue
    }

    $cleanedFiles = @()
    if (Test-Path $cleanedDir) {
        $cleanedFiles = Get-ChildItem $cleanedDir -Filter "*.jsonl" -ErrorAction SilentlyContinue
    }

    $inventoryFile = "_docs\2025-11-27_main_dataset_inventory.md"
    $inventoryExists = Test-Path $inventoryFile

    return @{
        DatasetCount = $datasets.Count
        CleanedCount = $cleanedFiles.Count
        InventoryExists = $inventoryExists
        LastInventoryUpdate = if ($inventoryExists) { (Get-Item $inventoryFile).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss") } else { "N/A" }
        TotalSizeGB = if ($datasets.Count -gt 0) {
            try {
                $size = (Get-ChildItem $datasetDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
                [math]::Round($size, 2)
            } catch { 0 }
        } else { 0 }
    }
}

function Get-GGUFProgress {
    $ggufDir = "D:\webdataset\gguf_models"

    $models = @()
    if (Test-Path $ggufDir) {
        $models = Get-ChildItem $ggufDir -Directory -ErrorAction SilentlyContinue
    }

    $totalFiles = 0
    $totalSizeGB = 0

    foreach ($model in $models) {
        $files = Get-ChildItem $model.FullName -File -ErrorAction SilentlyContinue
        $totalFiles += $files.Count
        $size = ($files | Measure-Object -Property Length -Sum).Sum / 1GB
        $totalSizeGB += $size
    }

    return @{
        ModelCount = $models.Count
        TotalFiles = $totalFiles
        TotalSizeGB = [math]::Round($totalSizeGB, 2)
        LastConversion = if ($models.Count -gt 0) {
            $latest = $models | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            $latest.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        } else { "N/A" }
    }
}

function Get-TestProgress {
    $testFiles = Get-ChildItem "_docs" -Filter "*test*.md" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending

    $testCategories = @{
        "日本語テスト" = ($testFiles | Where-Object { $_.Name -like "*japanese*" }).Count
        "複雑問題テスト" = ($testFiles | Where-Object { $_.Name -like "*complex*" }).Count
        "安全性テスト" = ($testFiles | Where-Object { $_.Name -like "*safety*" }).Count
        "Ollamaテスト" = ($testFiles | Where-Object { $_.Name -like "*ollama*" }).Count
        "GGUFテスト" = ($testFiles | Where-Object { $_.Name -like "*gguf*" }).Count
    }

    return @{
        TotalTests = $testFiles.Count
        Categories = $testCategories
        LastTest = if ($testFiles.Count -gt 0) { $testFiles[0].LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss") } else { "N/A" }
        RecentTests = $testFiles | Select-Object -First 5 | ForEach-Object { $_.Name }
    }
}

function Get-ImplementationProgress {
    $logFiles = Get-ChildItem "_docs" -Filter "2025-*.md" -ErrorAction SilentlyContinue

    # 完了した実装をカウント
    $completedLogs = $logFiles | Where-Object { $_.Name -like "*完了*" -or $_.Name -like "*complete*" }

    # 最近の実装ログ
    $recentLogs = $logFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 10

    return @{
        TotalLogs = $logFiles.Count
        CompletedLogs = $completedLogs.Count
        RecentLogs = $recentLogs | ForEach-Object {
            @{
                Name = $_.Name
                Date = $_.LastWriteTime.ToString("yyyy-MM-dd")
                IsCompleted = $_.Name -like "*完了*" -or $_.Name -like "*complete*"
            }
        }
    }
}

function Get-SystemResources {
    # ディスク使用状況
    $dDrive = Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='D:'" -ErrorAction SilentlyContinue
    $dDriveUsage = if ($dDrive) {
        $used = ($dDrive.Size - $dDrive.FreeSpace) / 1GB
        $total = $dDrive.Size / 1GB
        @{
            UsedGB = [math]::Round($used, 2)
            TotalGB = [math]::Round($total, 2)
            FreeGB = [math]::Round(($dDrive.FreeSpace / 1GB), 2)
            UsagePercent = [math]::Round(($used / $total) * 100, 1)
        }
    } else {
        @{ UsedGB = 0; TotalGB = 0; FreeGB = 0; UsagePercent = 0 }
    }

    # CPUとメモリ使用率
    $cpu = Get-WmiObject Win32_Processor -ErrorAction SilentlyContinue | Select-Object -First 1
    $cpuUsage = if ($cpu) { $cpu.LoadPercentage } else { 0 }

    $memory = Get-WmiObject Win32_OperatingSystem -ErrorAction SilentlyContinue
    $memoryUsage = if ($memory) {
        $totalMemory = $memory.TotalVisibleMemorySize
        $freeMemory = $memory.FreePhysicalMemory
        $usedMemory = $totalMemory - $freeMemory
        [math]::Round(($usedMemory / $totalMemory) * 100, 1)
    } else { 0 }

    return @{
        DDrive = $dDriveUsage
        CPUUsage = $cpuUsage
        MemoryUsage = $memoryUsage
    }
}

function Show-ProgressDashboard {
    param($progressData)

    Clear-Host
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "       SO8Tプロジェクト進捗監視ダッシュボード" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "ワークツリー: $($progressData.WorktreeName)" -ForegroundColor Yellow
    Write-Host "最終更新: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
    Write-Host ""

    # トレーニング進捗
    Write-Host "🔄 トレーニング進捗" -ForegroundColor Green
    $train = $progressData.Training
    Write-Host "   状態: $($train.Status)" -ForegroundColor $(if ($train.Status -eq "実行中") { "Green" } else { "Gray" })
    Write-Host "   最終更新: $($train.LastUpdate)"
    Write-Host "   エポック: $($train.CurrentEpoch)/$($train.TotalEpochs)"
    Write-Host "   Loss: $($train.Loss)"
    Write-Host "   進捗: [$($train.Progress)%]"
    Write-Host "   ログ: $($train.LogFile)"
    Write-Host ""

    # データセット進捗
    Write-Host "📊 データセット進捗" -ForegroundColor Blue
    $ds = $progressData.Dataset
    Write-Host "   データセット数: $($ds.DatasetCount)"
    Write-Host "   クレンジング済み: $($ds.CleanedCount)"
    Write-Host "   インベントリ: $(if ($ds.InventoryExists) { "存在" } else { "未作成" })"
    Write-Host "   最終更新: $($ds.LastInventoryUpdate)"
    Write-Host "   総サイズ: $($ds.TotalSizeGB) GB"
    Write-Host ""

    # GGUF変換進捗
    Write-Host "🔧 GGUF変換進捗" -ForegroundColor Magenta
    $gguf = $progressData.GGUF
    Write-Host "   モデル数: $($gguf.ModelCount)"
    Write-Host "   総ファイル数: $($gguf.TotalFiles)"
    Write-Host "   総サイズ: $($gguf.TotalSizeGB) GB"
    Write-Host "   最終変換: $($gguf.LastConversion)"
    Write-Host ""

    # テスト進捗
    Write-Host "🧪 テスト進捗" -ForegroundColor Yellow
    $test = $progressData.Test
    Write-Host "   総テスト数: $($test.TotalTests)"
    Write-Host "   最終テスト: $($test.LastTest)"
    Write-Host "   カテゴリ:"
    foreach ($category in $test.Categories.GetEnumerator()) {
        Write-Host "     $($category.Key): $($category.Value)"
    }
    Write-Host ""

    # 実装ログ進捗
    Write-Host "📝 実装ログ進捗" -ForegroundColor Red
    $impl = $progressData.Implementation
    Write-Host "   総ログ数: $($impl.TotalLogs)"
    Write-Host "   完了ログ数: $($impl.CompletedLogs)"
    Write-Host "   完了率: $([math]::Round(($impl.CompletedLogs / $impl.TotalLogs) * 100, 1))%"
    Write-Host "   最近のログ:"
    foreach ($log in $impl.RecentLogs | Select-Object -First 5) {
        $status = if ($log.IsCompleted) { "[完了]" } else { "[作業中]" }
        Write-Host "     $status $($log.Date) $($log.Name)" -ForegroundColor $(if ($log.IsCompleted) { "Green" } else { "White" })
    }
    Write-Host ""

    # システムリソース
    Write-Host "💻 システムリソース" -ForegroundColor Gray
    $sys = $progressData.System
    Write-Host "   Dドライブ使用量: $($sys.DDrive.UsedGB)/$($sys.DDrive.TotalGB) GB ($($sys.DDrive.UsagePercent)%)"
    Write-Host "   CPU使用率: $($sys.CPUUsage)%"
    Write-Host "   メモリ使用率: $($sys.MemoryUsage)%"
    Write-Host ""

    # 全体進捗バー
    $overallProgress = [math]::Round((
        ($train.Progress * 0.3) +  # トレーニング 30%
        (($ds.DatasetCount / 10) * 100 * 0.2) +  # データセット 20% (目標10個)
        (($gguf.ModelCount / 5) * 100 * 0.15) +  # GGUF 15% (目標5モデル)
        (($test.TotalTests / 20) * 100 * 0.15) +  # テスト 15% (目標20テスト)
        (($impl.CompletedLogs / $impl.TotalLogs) * 100 * 0.2)  # 実装 20%
    ), 1)

    Write-Host "🎯 全体進捗: $($overallProgress)%" -ForegroundColor Cyan
    $progressBar = "[" + ("█" * [math]::Floor($overallProgress / 5)) + ("░" * (20 - [math]::Floor($overallProgress / 5))) + "]"
    Write-Host $progressBar -ForegroundColor Cyan
}

# メイン処理
$worktreeName = Get-WorktreeName

if (-not $Quiet) {
    Write-Host "SO8Tプロジェクト進捗監視を開始します..." -ForegroundColor Green
    Write-Host "ワークツリー: $worktreeName" -ForegroundColor Yellow
    Write-Host ""
}

do {
    try {
        # 各進捗データを収集
        $progressData = @{
            WorktreeName = $worktreeName
            Training = Get-TrainingProgress
            Dataset = Get-DatasetProgress
            GGUF = Get-GGUFProgress
            Test = Get-TestProgress
            Implementation = Get-ImplementationProgress
            System = Get-SystemResources
        }

        if (-not $Quiet) {
            Show-ProgressDashboard -progressData $progressData
        }

        if ($Continuous) {
            Write-Host "次回更新まで $IntervalSeconds 秒待機中..." -ForegroundColor DarkGray
            Start-Sleep -Seconds $IntervalSeconds
        }
    } catch {
        Write-Host "エラーが発生しました: $($_.Exception.Message)" -ForegroundColor Red
        if ($Continuous) {
            Start-Sleep -Seconds $IntervalSeconds
        }
    }
} while ($Continuous)

# 単発実行時は最後にオーディオ通知
if (-not $Continuous) {
    Write-Host "[AUDIO] 進捗監視完了、通知を再生します..." -ForegroundColor Green

    $audioFile = "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
    if (Test-Path $audioFile) {
        try {
            Add-Type -AssemblyName System.Windows.Forms
            $player = New-Object System.Media.SoundPlayer $audioFile
            $player.PlaySync()
            Write-Host "[OK] marisa_owattaze.wav を再生しました" -ForegroundColor Green
        } catch {
            Write-Host "[WARNING] オーディオ再生に失敗しました: $($_.Exception.Message)" -ForegroundColor Yellow
            [System.Console]::Beep(1000, 500)
        }
    } else {
        Write-Host "[WARNING] オーディオファイルが見つかりません" -ForegroundColor Yellow
        [System.Console]::Beep(1000, 500)
    }
}
