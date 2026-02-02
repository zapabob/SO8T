#!/usr/bin/env pwsh
# GGUF量子化評価パイプライン進捗監視スクリプト
# PowerShellを使用したリアルタイム進捗可視化

param(
    [Parameter(Mandatory=$true)]
    [string]$PipelineId,

    [Parameter(Mandatory=$false)]
    [int]$UpdateInterval = 2,

    [Parameter(Mandatory=$false)]
    [switch]$Detailed
)

# コンソール設定
$Host.UI.RawUI.WindowTitle = "GGUF量子化評価パイプライン - 進捗監視"
$ErrorActionPreference = "Stop"

# 色設定
$Colors = @{
    Info = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Progress = "Blue"
    Header = "Magenta"
}

function Write-ColoredOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )

    Write-Host $Message -ForegroundColor $Color
}

function Get-PipelineProgress {
    param([string]$PipelineId)

    try {
        # パイプライン状態ファイル読み込み
        $progressFile = "quantization_evaluation_output\pipeline_progress_$PipelineId.json"

        if (Test-Path $progressFile) {
            $progressData = Get-Content $progressFile -Raw | ConvertFrom-Json

            return @{
                Current = $progressData.current_phase
                Total = $progressData.total_phases
                Percentage = $progressData.percentage
                Phase = $progressData.phase_name
                Elapsed = $progressData.elapsed_seconds
                ETA = $progressData.estimated_remaining
                Status = $progressData.status
                ProcessId = $progressData.process_id
            }
        } else {
            # デフォルト値
            return @{
                Current = 0
                Total = 5
                Percentage = 0
                Phase = "初期化中"
                Elapsed = 0
                ETA = "不明"
                Status = "running"
                ProcessId = $null
            }
        }
    }
    catch {
        Write-ColoredOutput "進捗データ読み込みエラー: $($_.Exception.Message)" $Colors.Error
        return $null
    }
}

function Show-ResourceUsage {
    param([int]$ProcessId = $null)

    try {
        if ($ProcessId) {
            # 特定のプロセスリソース使用量取得
            $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
            if ($process) {
                $cpu = $process.CPU
                $memoryMB = [math]::Round($process.WorkingSet64 / 1MB, 2)
                $threads = $process.Threads.Count
            } else {
                $cpu = 0
                $memoryMB = 0
                $threads = 0
            }
        } else {
            # システム全体のリソース使用量
            $cpu = (Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction SilentlyContinue).CounterSamples[0].CookedValue
            $cpu = [math]::Round($cpu, 1)

            $memory = Get-Counter '\Memory\% Committed Bytes In Use' -ErrorAction SilentlyContinue
            $memoryPercent = [math]::Round($memory.CounterSamples[0].CookedValue, 1)

            $totalMemoryGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
            $usedMemoryGB = [math]::Round(($totalMemoryGB * $memoryPercent / 100), 1)
        }

        # GPU情報取得（NVIDIA GPUの場合）
        try {
            $gpuInfo = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
            if ($LASTEXITCODE -eq 0) {
                $gpuData = $gpuInfo -split ','
                $gpuUtil = [int]$gpuData[0]
                $gpuMemoryUsed = [int]$gpuData[1]
                $gpuMemoryTotal = [int]$gpuData[2]
            }
        }
        catch {
            $gpuUtil = $null
        }

        Write-ColoredOutput "┌─ リソース使用量 ──────────────────────────────┐" $Colors.Header

        if ($ProcessId) {
            Write-ColoredOutput ("│ CPU使用率: {0,6} % │ メモリ: {1,6} MB │" -f $cpu, $memoryMB) $Colors.Info
            Write-ColoredOutput ("│ スレッド数: {0,4}   │                    │" -f $threads) $Colors.Info
        } else {
            Write-ColoredOutput ("│ CPU使用率: {0,6} % │ メモリ: {1,4}/{2,4} GB │" -f $cpu, $usedMemoryGB, $totalMemoryGB) $Colors.Info
        }

        if ($gpuUtil -ne $null) {
            Write-ColoredOutput ("│ GPU使用率: {0,6} % │ VRAM: {1,4}/{2,4} MB │" -f $gpuUtil, $gpuMemoryUsed, $gpuMemoryTotal) $Colors.Info
        }

        Write-ColoredOutput "└─────────────────────────────────────────────┘" $Colors.Header

    }
    catch {
        Write-ColoredOutput "リソース使用量取得エラー: $($_.Exception.Message)" $Colors.Warning
    }
}

function Show-ProgressBar {
    param(
        [int]$Current,
        [int]$Total,
        [string]$Activity,
        [string]$Status = "",
        [int]$Width = 50
    )

    $percentage = if ($Total -gt 0) { [math]::Min(($Current / $Total) * 100, 100) } else { 0 }
    $filled = [math]::Floor($percentage / (100 / $Width))
    $empty = $Width - $filled

    $progressBar = "[" + ("█" * $filled) + ("░" * $empty) + "]"
    $percentText = ("{0,3}" -f [math]::Round($percentage)) + "%"

    Write-ColoredOutput "" # 改行
    Write-ColoredOutput "╔══════════════════════════════════════════════════════════════════════════════╗" $Colors.Header
    Write-ColoredOutput ("║ " + $Activity.PadRight(76) + " ║") $Colors.Header
    Write-ColoredOutput ("║ " + $progressBar + " " + $percentText + " ║") $Colors.Progress
    if ($Status) {
        Write-ColoredOutput ("║ " + $Status.PadRight(76) + " ║") $Colors.Info
    }
    Write-ColoredOutput "╚══════════════════════════════════════════════════════════════════════════════╝" $Colors.Header
}

function Format-TimeSpan {
    param([int]$Seconds)

    if ($Seconds -lt 60) {
        return "$Seconds秒"
    } elseif ($Seconds -lt 3600) {
        $minutes = [math]::Floor($Seconds / 60)
        $remainingSeconds = $Seconds % 60
        return ("{0}分{1}秒" -f $minutes, $remainingSeconds)
    } else {
        $hours = [math]::Floor($Seconds / 3600)
        $remainingMinutes = [math]::Floor(($Seconds % 3600) / 60)
        return ("{0}時間{1}分" -f $hours, $remainingMinutes)
    }
}

function Show-PhaseDetails {
    param([string]$Phase)

    $phaseDetails = @{
        "imatrix_collection" = @{
            Name = "imatrixデータ収集"
            Description = "モデルパラメータの重要度行列を計算中..."
            EstimatedTime = "5-15分"
        }
        "quantization" = @{
            Name = "GGUF量子化実行"
            Description = "imatrix保護を使用した量子化処理中..."
            EstimatedTime = "10-30分"
        }
        "evaluation" = @{
            Name = "統計的評価"
            Description = "複数ベンチマークでの性能評価中..."
            EstimatedTime = "20-60分"
        }
        "visualization" = @{
            Name = "結果可視化"
            Description = "エラーバー付きグラフとレポート生成中..."
            EstimatedTime = "2-5分"
        }
        "documentation" = @{
            Name = "学術文書生成"
            Description = "スコアカードと分析レポート作成中..."
            EstimatedTime = "1-3分"
        }
    }

    if ($phaseDetails.ContainsKey($Phase)) {
        $details = $phaseDetails[$Phase]
        Write-ColoredOutput "" # 改行
        Write-ColoredOutput ("📋 現在のPhase: " + $details.Name) $Colors.Info
        Write-ColoredOutput ("📝 " + $details.Description) $Colors.Info
        Write-ColoredOutput ("⏱️  推定時間: " + $details.EstimatedTime) $Colors.Info
    }
}

function Start-ProgressMonitoring {
    param(
        [string]$PipelineId,
        [int]$UpdateInterval,
        [bool]$Detailed
    )

    Write-ColoredOutput "🚀 GGUF量子化評価パイプライン進捗監視を開始します" $Colors.Success
    Write-ColoredOutput "Pipeline ID: $PipelineId" $Colors.Info
    Write-ColoredOutput "更新間隔: ${UpdateInterval}秒" $Colors.Info
    Write-ColoredOutput "" # 改行

    $startTime = Get-Date
    $lastUpdate = $startTime

    # 初期画面クリア
    Clear-Host

    while ($true) {
        try {
            $currentTime = Get-Date
            $progress = Get-PipelineProgress -PipelineId $PipelineId

            if ($null -eq $progress) {
                Write-ColoredOutput "進捗データを取得できません。パイプラインが実行中か確認してください。" $Colors.Warning
                Start-Sleep -Seconds $UpdateInterval
                continue
            }

            # 画面クリア
            Clear-Host

            # ヘッダー表示
            Write-ColoredOutput "╔══════════════════════════════════════════════════════════════════════════════╗" $Colors.Header
            Write-ColoredOutput "║              GGUF量子化評価パイプライン - リアルタイム監視                  ║" $Colors.Header
            Write-ColoredOutput "╚══════════════════════════════════════════════════════════════════════════════╝" $Colors.Header
            Write-ColoredOutput "" # 改行

            # 基本情報表示
            $elapsed = [math]::Round(($currentTime - $startTime).TotalSeconds)
            Write-ColoredOutput ("🕐 開始時刻: " + $startTime.ToString("yyyy/MM/dd HH:mm:ss")) $Colors.Info
            Write-ColoredOutput ("⏱️  経過時間: " + (Format-TimeSpan $elapsed)) $Colors.Info

            if ($progress.ETA -and $progress.ETA -ne "不明") {
                Write-ColoredOutput ("⏳ 推定残り: " + $progress.ETA) $Colors.Info
            }

            Write-ColoredOutput ("🔢 Pipeline ID: " + $PipelineId) $Colors.Info
            Write-ColoredOutput "" # 改行

            # Phase情報表示
            Show-PhaseDetails -Phase $progress.Phase

            # 進捗バー表示
            $phaseDisplay = if ($progress.Phase) { $progress.Phase } else { "実行中" }
            Show-ProgressBar -Current $progress.Current -Total $progress.Total -Activity $phaseDisplay -Status ("Phase " + ($progress.Current + 1) + "/" + $progress.Total)

            # 詳細情報（-Detailedオプション時）
            if ($Detailed) {
                Write-ColoredOutput "" # 改行
                Write-ColoredOutput "📊 詳細情報:" $Colors.Info
                Write-ColoredOutput ("   完了Phase: " + $progress.Current + "/" + $progress.Total) $Colors.Info
                Write-ColoredOutput ("   進捗率: " + ("{0:N1}" -f $progress.Percentage) + "%") $Colors.Info
                Write-ColoredOutput ("   ステータス: " + $progress.Status) $Colors.Info
            }

            # リソース使用量表示
            Show-ResourceUsage -ProcessId $progress.ProcessId

            # フッター
            Write-ColoredOutput "" # 改行
            Write-ColoredOutput "💡 ヒント: Ctrl+Cで監視を停止 | -Detailedで詳細表示" $Colors.Info
            Write-ColoredOutput ("🔄 次回更新まで: " + $UpdateInterval + "秒") $Colors.Info

            # 完了チェック
            if ($progress.Status -eq "completed") {
                Write-ColoredOutput "" # 改行
                Write-ColoredOutput "🎉 パイプライン実行が完了しました！" $Colors.Success
                Write-ColoredOutput "📊 結果は quantization_evaluation_output/ ディレクトリに保存されています。" $Colors.Success
                break
            }

            if ($progress.Status -eq "failed") {
                Write-ColoredOutput "" # 改行
                Write-ColoredOutput "❌ パイプライン実行が失敗しました。" $Colors.Error
                Write-ColoredOutput "📋 エラーログを確認してください。" $Colors.Error
                break
            }

        }
        catch {
            Write-ColoredOutput "監視エラー: $($_.Exception.Message)" $Colors.Error
        }

        # 更新間隔待機
        $lastUpdate = $currentTime
        Start-Sleep -Seconds $UpdateInterval
    }
}

# メイン処理
try {
    # 管理者権限チェック（オプション）
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-ColoredOutput "ℹ️  管理者権限で実行するとより詳細なシステム情報が取得できます。" $Colors.Warning
    }

    # 監視開始
    Start-ProgressMonitoring -PipelineId $PipelineId -UpdateInterval $UpdateInterval -Detailed:$Detailed

}
catch {
    Write-ColoredOutput "致命的エラー: $($_.Exception.Message)" $Colors.Error
    Write-ColoredOutput "スタックトレース:" $Colors.Error
    Write-ColoredOutput $_.ScriptStackTrace $Colors.Error
    exit 1
}

Write-ColoredOutput "" # 改行
Write-ColoredOutput "👋 進捗監視を終了します。" $Colors.Info