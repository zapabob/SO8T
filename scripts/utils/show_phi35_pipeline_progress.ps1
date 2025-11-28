# Phi-3.5 SO8T Pipeline Progress Display
# パイプラインの進捗状況をデスクトップ表示

Write-Host "=== Phi-3.5 SO8T Pipeline Progress ===" -ForegroundColor Green

# パイプライン状態ファイルの確認
$StateFile = "D:/webdataset/pipeline_state/phi35_pipeline_state.json"
$CheckpointBase = "D:/webdataset/checkpoints/training"

if (Test-Path $StateFile) {
    try {
        $State = Get-Content $StateFile -Raw | ConvertFrom-Json

        Write-Host "Pipeline Status:" -ForegroundColor Cyan
        Write-Host "  Current Step: $($State.current_step + 1) / 5" -ForegroundColor White
        Write-Host "  Completed Steps: $($State.completed_steps -join ', ')" -ForegroundColor Green

        if ($State.last_run) {
            Write-Host "  Last Run: $($State.last_run)" -ForegroundColor Gray
        }

        if ($State.errors -and $State.errors.Count -gt 0) {
            Write-Host "  Errors: $($State.errors.Count)" -ForegroundColor Red
            foreach ($Error in $State.errors) {
                Write-Host "    - $($Error.step): $($Error.error)" -ForegroundColor Red
            }
        }

    } catch {
        Write-Host "Error reading pipeline state: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "No active pipeline state found." -ForegroundColor Yellow
}

Write-Host ""

# トレーニングセッションの確認
$PipelineDirs = Get-ChildItem -Path $CheckpointBase -Directory -Filter "phi35_pipeline_*" | Sort-Object LastWriteTime -Descending

if ($PipelineDirs) {
    Write-Host "Recent Pipeline Sessions:" -ForegroundColor Cyan

    foreach ($PipelineDir in $PipelineDirs | Select-Object -First 5) {
        $SessionName = $PipelineDir.Name
        $SessionPath = $PipelineDir.FullName

        Write-Host "Session: $SessionName" -ForegroundColor White

        # 完了チェック
        $FinalModel = Join-Path $SessionPath "final_model"
        if (Test-Path $FinalModel) {
            Write-Host "  Completion: COMPLETED" -ForegroundColor Green
        } else {
            Write-Host "  Completion: IN PROGRESS" -ForegroundColor Yellow
        }

        # アニーリング結果確認
        $AnnealingFile = Join-Path $SessionPath "alpha_gate_annealing_results.json"
        if (Test-Path $AnnealingFile) {
            try {
                $Annealing = Get-Content $AnnealingFile -Raw | ConvertFrom-Json
                $FinalAlpha = $Annealing.final_alpha
                $Transitions = $Annealing.phase_transitions.Count
                Write-Host "  Alpha Annealing: α = $([math]::Round($FinalAlpha, 4)) ($Transitions transitions)" -ForegroundColor Cyan
            } catch {
                Write-Host "  Alpha Annealing: Results available" -ForegroundColor Cyan
            }
        }

        # チェックポイント数
        $Checkpoints = Get-ChildItem -Path $SessionPath -Filter "checkpoint_*" -Directory
        Write-Host "  Checkpoints: $($Checkpoints.Count)" -ForegroundColor Gray

        Write-Host ""
    }
} else {
    Write-Host "No pipeline sessions found." -ForegroundColor Yellow
}

Write-Host ""

# データセット状況
$DatasetFiles = @(
    "D:/webdataset/phi35_integrated/phi35_ppo_optimized_integrated.jsonl",
    "D:/webdataset/integrated_dataset_full.jsonl",
    "D:/webdataset/integrated_dataset.jsonl"
)

Write-Host "Dataset Status:" -ForegroundColor Cyan
foreach ($DatasetFile in $DatasetFiles) {
    if (Test-Path $DatasetFile) {
        $SizeMB = [math]::Round((Get-Item $DatasetFile).Length / 1MB, 2)
        $LineCount = (Get-Content $DatasetFile | Measure-Object -Line).Lines
        Write-Host "  $(Split-Path $DatasetFile -Leaf): ${LineCount} samples (${SizeMB}MB)" -ForegroundColor Green
    } else {
        Write-Host "  $(Split-Path $DatasetFile -Leaf): NOT FOUND" -ForegroundColor Red
    }
}

Write-Host ""

# GGUFモデル状況
$GgufDir = "D:/webdataset/gguf_models/phi35_so8t_thinking"
if (Test-Path $GgufDir) {
    $GgufFiles = Get-ChildItem -Path $GgufDir -Filter "*.gguf"
    Write-Host "GGUF Models:" -ForegroundColor Cyan
    foreach ($GgufFile in $GgufFiles) {
        $SizeGB = [math]::Round($GgufFile.Length / 1GB, 2)
        Write-Host "  $($GgufFile.Name): ${SizeGB}GB" -ForegroundColor Green
    }
} else {
    Write-Host "GGUF Models: Not generated yet" -ForegroundColor Yellow
}

Write-Host ""

# 推定残り時間計算
if (Test-Path $StateFile) {
    try {
        $State = Get-Content $StateFile -Raw | ConvertFrom-Json
        $CurrentStep = $State.current_step
        $CompletedCount = $State.completed_steps.Count

        # 各ステップの推定時間（分）
        $StepTimes = @{
            0 = 30   # HF dataset collection
            1 = 60   # Dataset integration
            2 = 120  # Phi-3.5 conversion
            3 = 2880 # PPO training (48 hours)
            4 = 180  # Evaluation
        }

        $ElapsedTime = 0
        for ($i = 0; $i -lt $CompletedCount; $i++) {
            $ElapsedTime += $StepTimes[$i]
        }

        $RemainingTime = 0
        for ($i = $CurrentStep; $i -lt 5; $i++) {
            $RemainingTime += $StepTimes[$i]
        }

        Write-Host "Time Estimate:" -ForegroundColor Cyan
        Write-Host "  Elapsed: $([math]::Round($ElapsedTime / 60, 1)) hours" -ForegroundColor White
        Write-Host "  Remaining: $([math]::Round($RemainingTime / 60, 1)) hours" -ForegroundColor Yellow

        if ($RemainingTime -gt 1440) { # 24時間以上
            Write-Host "  Note: PPO training may take 2-3 days with current settings" -ForegroundColor Yellow
        }

    } catch {
        # エラー時はスキップ
    }
}

Write-Host ""
Write-Host "=== Progress Display Complete ===" -ForegroundColor Green
Write-Host "Last updated: $(Get-Date)" -ForegroundColor Gray
