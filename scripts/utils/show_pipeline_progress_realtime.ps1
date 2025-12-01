#!/usr/bin/env powershell
# -*- coding: utf-8 -*-
<#
.SYNOPSIS
    SO8Tパイプライン実行状況をリアルタイムで最前面表示

.DESCRIPTION
    パイプライン実行中のログをリアルタイムで監視し、常時最前面に表示します。
    パイプラインの実行状況、進捗、システムリソースを継続的に監視します。
#>

param(
    [switch]$NoAudio,
    [switch]$NoTopMost,
    [string]$LogFile = "so8t_automated_pipeline.log",
    [int]$UpdateInterval = 2
)

# UTF-8エンコーディング設定
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Continue"

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

# ウィンドウを最前面に設定
function Set-WindowTopMost {
    param([string]$Title)

    if ($NoTopMost) { return }

    try {
        Add-Type @"
        using System;
        using System.Runtime.InteropServices;

        public class Win32 {
            [DllImport("user32.dll")]
            public static extern IntPtr FindWindow(string lpClassName, string lpWindowName);

            [DllImport("user32.dll")]
            public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);

            public static readonly IntPtr HWND_TOPMOST = new IntPtr(-1);
            public const uint SWP_NOSIZE = 0x0001;
            public const uint SWP_NOMOVE = 0x0002;
            public const uint TOPMOST_FLAGS = SWP_NOMOVE | SWP_NOSIZE;
        }
"@

        $hWnd = [Win32]::FindWindow($null, $Title)
        if ($hWnd -ne [IntPtr]::Zero) {
            [Win32]::SetWindowPos($hWnd, [Win32]::HWND_TOPMOST, 0, 0, 0, 0, [Win32]::TOPMOST_FLAGS)
        }
    } catch {
        # 無視
    }
}

# ログファイル監視クラス
class LogMonitor {
    [string]$LogFilePath
    [long]$LastPosition
    [System.Collections.Generic.List[string]]$RecentLines

    LogMonitor([string]$path) {
        $this.LogFilePath = $path
        $this.LastPosition = 0
        $this.RecentLines = New-Object System.Collections.Generic.List[string]
    }

    [System.Collections.Generic.List[string]]GetNewLines() {
        $newLines = New-Object System.Collections.Generic.List[string]

        try {
            if (Test-Path $this.LogFilePath) {
                $fileInfo = Get-Item $this.LogFilePath
                if ($fileInfo.Length -gt $this.LastPosition) {
                    $stream = [System.IO.File]::Open($this.LogFilePath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
                    $stream.Position = $this.LastPosition

                    $reader = New-Object System.IO.StreamReader($stream)
                    while (-not $reader.EndOfStream) {
                        $line = $reader.ReadLine()
                        if ($line) {
                            $newLines.Add($line)
                        }
                    }

                    $this.LastPosition = $stream.Position
                    $reader.Close()
                    $stream.Close()
                }
            }
        } catch {
            # ファイルアクセスエラーは無視
        }

        # 最近の行を更新
        foreach ($line in $newLines) {
            $this.RecentLines.Add($line)
        }

        # 最大50行に制限
        while ($this.RecentLines.Count -gt 50) {
            $this.RecentLines.RemoveAt(0)
        }

        return $newLines
    }

    [string[]]GetRecentLines([int]$count = 20) {
        $start = [Math]::Max(0, $this.RecentLines.Count - $count)
        return $this.RecentLines.GetRange($start, $this.RecentLines.Count - $start).ToArray()
    }
}

# パイプラインステータス取得
function Get-PipelineStatus {
    param([LogMonitor]$logMonitor)

    $status = @{
        IsRunning = $false
        CurrentStage = "unknown"
        Progress = 0
        StartTime = $null
        LastUpdate = Get-Date
        ErrorCount = 0
        WarningCount = 0
    }

    $recentLines = $logMonitor.GetRecentLines(100)

    foreach ($line in $recentLines) {
        # パイプライン開始検出
        if ($line -match "SO\(8\)T Automated Pipeline.*starting" -or $line -match "Starting complete SO\(8\)T pipeline") {
            $status.IsRunning = $true
        }

        # ステージ検出
        if ($line -match "Starting SFT training") {
            $status.CurrentStage = "sft_training"
            $status.Progress = 10
        } elseif ($line -match "Starting PPO training") {
            $status.CurrentStage = "ppo_training"
            $status.Progress = 30
        } elseif ($line -match "Starting GGUF conversion") {
            $status.CurrentStage = "gguf_conversion"
            $status.Progress = 60
        } elseif ($line -match "Starting AB testing") {
            $status.CurrentStage = "ab_testing"
            $status.Progress = 80
        } elseif ($line -match "Starting HF upload") {
            $status.CurrentStage = "hf_upload"
            $status.Progress = 95
        } elseif ($line -match "Pipeline completed successfully") {
            $status.CurrentStage = "completed"
            $status.Progress = 100
            $status.IsRunning = $false
        }

        # エラー検出
        if ($line -match "ERROR|FAILED|CRITICAL") {
            $status.ErrorCount++
        }

        # 警告検出
        if ($line -match "WARNING|WARN") {
            $status.WarningCount++
        }
    }

    return $status
}

# システムリソース取得
function Get-SystemResources {
    $resources = @{
        CPUUsage = 0
        MemoryUsage = 0
        DiskUsage = 0
        GPUUsage = 0
        GPUTemperature = 0
    }

    try {
        # CPU使用率
        $cpuCounter = Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction SilentlyContinue
        if ($cpuCounter) {
            $resources.CPUUsage = [math]::Round($cpuCounter.CounterSamples[0].CookedValue, 1)
        }
    } catch {
        $resources.CPUUsage = 0
    }

    try {
        # メモリ使用率
        $os = Get-CimInstance Win32_OperatingSystem
        $totalMemory = $os.TotalVisibleMemorySize
        $freeMemory = $os.FreePhysicalMemory
        $resources.MemoryUsage = [math]::Round((($totalMemory - $freeMemory) / $totalMemory) * 100, 1)
    } catch {
        $resources.MemoryUsage = 0
    }

    try {
        # Dドライブ使用率
        $dDrive = Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='D:'"
        if ($dDrive) {
            $resources.DiskUsage = [math]::Round((($dDrive.Size - $dDrive.FreeSpace) / $dDrive.Size) * 100, 1)
        }
    } catch {
        $resources.DiskUsage = 0
    }

    try {
        # GPU情報
        $gpuInfo = nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>$null
        if ($gpuInfo) {
            $parts = $gpuInfo -split ','
            if ($parts.Count -ge 2) {
                $resources.GPUUsage = [int]$parts[0].Trim()
                $resources.GPUTemperature = [int]$parts[1].Trim()
            }
        }
    } catch {
        $resources.GPUUsage = 0
        $resources.GPUTemperature = 0
    }

    return $resources
}

# プログレスバーの表示
function Show-ProgressBar {
    param([int]$percentage, [string]$label = "")

    $width = 50
    $filled = [math]::Floor(($percentage / 100) * $width)
    $empty = $width - $filled

    $bar = "[" + ("█" * $filled) + ("░" * $empty) + "]"

    if ($label) {
        return "$label $percentage% $bar"
    } else {
        return "$percentage% $bar"
    }
}

# メイン表示関数
function Show-PipelineMonitor {
    param([LogMonitor]$logMonitor, [hashtable]$pipelineStatus, [hashtable]$resources, [string]$worktreeName)

    Clear-Host

    # ヘッダー
    Write-Host "╔══════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║                     SO8T パイプライン実行状況 リアルタイム監視                ║" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

    # 基本情報
    Write-Host "📊 実行情報" -ForegroundColor Yellow
    Write-Host "   ワークツリー: $worktreeName" -ForegroundColor White
    Write-Host "   最終更新: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor White
    Write-Host "   監視ログ: $LogFile" -ForegroundColor White
    Write-Host ""

    # パイプラインステータス
    Write-Host "🔄 パイプラインステータス" -ForegroundColor Green
    $statusColor = if ($pipelineStatus.IsRunning) { "Green" } elseif ($pipelineStatus.CurrentStage -eq "completed") { "Cyan" } else { "Gray" }
    Write-Host "   実行状態: $($pipelineStatus.IsRunning)" -ForegroundColor $statusColor
    Write-Host "   現在のステージ: $($pipelineStatus.CurrentStage)" -ForegroundColor $statusColor

    # 進捗バー
    $progressBar = Show-ProgressBar -percentage $pipelineStatus.Progress -label "全体進捗"
    Write-Host "   $progressBar" -ForegroundColor Cyan

    Write-Host "   エラー数: $($pipelineStatus.ErrorCount)" -ForegroundColor $(if ($pipelineStatus.ErrorCount -gt 0) { "Red" } else { "White" })
    Write-Host "   警告数: $($pipelineStatus.WarningCount)" -ForegroundColor $(if ($pipelineStatus.WarningCount -gt 0) { "Yellow" } else { "White" })
    Write-Host ""

    # システムリソース
    Write-Host "💻 システムリソース" -ForegroundColor Magenta
    Write-Host "   CPU使用率: $($resources.CPUUsage)%" -ForegroundColor $(if ($resources.CPUUsage -gt 80) { "Red" } elseif ($resources.CPUUsage -gt 60) { "Yellow" } else { "Green" })
    Write-Host "   メモリ使用率: $($resources.MemoryUsage)%" -ForegroundColor $(if ($resources.MemoryUsage -gt 85) { "Red" } elseif ($resources.MemoryUsage -gt 70) { "Yellow" } else { "Green" })
    Write-Host "   Dドライブ使用率: $($resources.DiskUsage)%" -ForegroundColor $(if ($resources.DiskUsage -gt 90) { "Red" } elseif ($resources.DiskUsage -gt 80) { "Yellow" } else { "Green" })

    if ($resources.GPUUsage -gt 0) {
        Write-Host "   GPU使用率: $($resources.GPUUsage)%" -ForegroundColor $(if ($resources.GPUUsage -gt 90) { "Red" } elseif ($resources.GPUUsage -gt 70) { "Yellow" } else { "Green" })
        Write-Host "   GPU温度: $($resources.GPUTemperature)°C" -ForegroundColor $(if ($resources.GPUTemperature -gt 75) { "Red" } elseif ($resources.GPUTemperature -gt 65) { "Yellow" } else { "Green" })
    }
    Write-Host ""

    # 最新ログ
    Write-Host "📝 最新ログ (直近20行)" -ForegroundColor Blue
    Write-Host "───────────────────────────────────────────────────────────────────────────────" -ForegroundColor Gray

    $recentLines = $logMonitor.GetRecentLines(20)
    if ($recentLines.Count -eq 0) {
        Write-Host "   (ログなし - パイプラインが開始されていない可能性があります)" -ForegroundColor Gray
    } else {
        foreach ($line in $recentLines) {
            # ログ行の色分け
            if ($line -match "ERROR|FAILED|CRITICAL") {
                Write-Host "   $line" -ForegroundColor Red
            } elseif ($line -match "WARNING|WARN") {
                Write-Host "   $line" -ForegroundColor Yellow
            } elseif ($line -match "SUCCESS|OK|completed|Initialized") {
                Write-Host "   $line" -ForegroundColor Green
            } elseif ($line -match "Alpha Gate|ALPHA|orthogonal|intermediate|rotation gate|PET") {
                Write-Host "   $line" -ForegroundColor Cyan
            } elseif ($line -match "Step|STEP") {
                Write-Host "   $line" -ForegroundColor Magenta
            } else {
                Write-Host "   $line" -ForegroundColor White
            }
        }
    }

    Write-Host ""
    Write-Host "🔄 監視中... (Ctrl+Cで停止)" -ForegroundColor DarkGray
    Write-Host "   更新間隔: ${UpdateInterval}秒" -ForegroundColor DarkGray
}

# パイプライン実行関数
function Start-PipelineExecution {
    param([string]$pythonExe, [string]$logFile)

    Write-Host "パイプライン実行を開始します..." -ForegroundColor Green

    # Pythonプロセスをバックグラウンドで起動
    $process = Start-Process -FilePath $pythonExe -ArgumentList "so8t_automated_pipeline.py", "--autostart" -RedirectStandardOutput $logFile -RedirectStandardError $logFile -PassThru -NoNewWindow

    return $process
}

# メイン関数
function Main {
    $worktreeName = Get-WorktreeName

    # ウィンドウタイトル設定
    $host.UI.RawUI.WindowTitle = "SO8T パイプライン監視 - $worktreeName"

    Write-Host "SO8T パイプライン実行状況 リアルタイム監視を開始します..." -ForegroundColor Green
    Write-Host "ワークツリー: $worktreeName" -ForegroundColor Yellow
    Write-Host ""

    # Python実行ファイルの検索
    $pythonExe = "py"
    if (Test-Path "C:\Python312\python.exe") {
        $pythonExe = "C:\Python312\python.exe"
    } elseif (Test-Path "C:\Python311\python.exe") {
        $pythonExe = "C:\Python311\python.exe"
    } elseif (Test-Path "C:\Python310\python.exe") {
        $pythonExe = "C:\Python310\python.exe"
    }

    Write-Host "使用Python: $pythonExe" -ForegroundColor White
    Write-Host "監視ログファイル: $LogFile" -ForegroundColor White
    Write-Host ""

    # パイプライン実行開始
    $pipelineProcess = Start-PipelineExecution -pythonExe $pythonExe -logFile $LogFile

    # ログ監視開始
    $logMonitor = [LogMonitor]::new($LogFile)

    # 初期待機
    Start-Sleep -Seconds 3

    try {
        while ($true) {
            # ウィンドウを最前面に設定
            Set-WindowTopMost -Title "SO8T パイプライン監視 - $worktreeName"

            # 新しいログ行を取得
            $newLines = $logMonitor.GetNewLines()

            # パイプラインステータスを取得
            $pipelineStatus = Get-PipelineStatus -logMonitor $logMonitor

            # システムリソースを取得
            $resources = Get-SystemResources

            # 画面表示更新
            Show-PipelineMonitor -logMonitor $logMonitor -pipelineStatus $pipelineStatus -resources $resources -worktreeName $worktreeName

            # プロセスが終了しているかチェック
            if ($pipelineProcess.HasExited) {
                Write-Host ""
                Write-Host "パイプライン実行が終了しました (終了コード: $($pipelineProcess.ExitCode))" -ForegroundColor Yellow

                # 完了時の最終表示
                $pipelineStatus.IsRunning = $false
                $pipelineStatus.CurrentStage = if ($pipelineProcess.ExitCode -eq 0) { "completed" } else { "failed" }
                $pipelineStatus.Progress = 100

                Show-PipelineMonitor -logMonitor $logMonitor -pipelineStatus $pipelineStatus -resources $resources -worktreeName $worktreeName

                break
            }

            # 更新間隔待機
            Start-Sleep -Seconds $UpdateInterval
        }
    } catch {
        Write-Host ""
        Write-Host "監視が中断されました: $($_.Exception.Message)" -ForegroundColor Red
    } finally {
        # プロセスがまだ実行中なら終了
        if (-not $pipelineProcess.HasExited) {
            Write-Host "パイプラインを終了します..." -ForegroundColor Yellow
            $pipelineProcess.Kill()
        }

        # 完了音声通知
        if (-not $NoAudio) {
            Write-Host ""
            Write-Host "[AUDIO] パイプライン監視完了、通知を再生します..." -ForegroundColor Green

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

        Write-Host ""
        Write-Host "SO8T パイプライン監視を終了します。" -ForegroundColor Cyan
    }
}

# メイン実行
Main
