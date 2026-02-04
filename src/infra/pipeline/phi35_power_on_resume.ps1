# Phi-3.5 SO8T Pipeline Power-On Auto Resume
# 電源投入時に未完了のパイプラインを自動再開

param(
    [switch]$Force = $false
)

Write-Host "=== Phi-3.5 SO8T Pipeline Power-On Auto Resume ===" -ForegroundColor Green
Write-Host "Checking for incomplete Phi-3.5 pipeline sessions..." -ForegroundColor Cyan

# プロジェクトディレクトリに移動
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

# Python環境設定
$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\so8t-mmllm\src;$env:PYTHONPATH"

# 未完了セッションの検索
$CheckpointBase = "D:/webdataset/checkpoints/training"
$PipelineDirs = Get-ChildItem -Path $CheckpointBase -Directory -Filter "phi35_pipeline_*" | Sort-Object LastWriteTime -Descending

$FoundIncomplete = $false

foreach ($PipelineDir in $PipelineDirs) {
    $PipelineName = $PipelineDir.Name
    $PipelinePath = $PipelineDir.FullName

    Write-Host "Checking session: $PipelineName" -ForegroundColor Yellow

    # 完了チェック（final_modelの存在）
    $FinalModelPath = Join-Path $PipelinePath "final_model"
    if (Test-Path $FinalModelPath) {
        Write-Host "  Status: COMPLETED (final model exists)" -ForegroundColor Green
        continue
    }

    # パイプライン状態ファイルの確認
    $StateFile = Join-Path $PipelinePath "pipeline_state.json"
    if (Test-Path $StateFile) {
        try {
            $State = Get-Content $StateFile -Raw | ConvertFrom-Json
            $CurrentStep = $State.current_step
            $CompletedSteps = $State.completed_steps
            $TotalSteps = 5  # パイプラインの総ステップ数

            Write-Host "  Status: INCOMPLETE" -ForegroundColor Yellow
            Write-Host "  Progress: $($CompletedSteps.Count)/$TotalSteps steps completed" -ForegroundColor Cyan
            Write-Host "  Current Step: $CurrentStep ($($State.steps[$CurrentStep]))" -ForegroundColor Cyan
            Write-Host "  Last Run: $($State.last_run)" -ForegroundColor Gray

            # 再開確認
            $Resume = $Force
            if (!$Force) {
                $Response = Read-Host "Resume this pipeline session? (y/n)"
                $Resume = $Response -eq "y" -or $Response -eq "Y"
            }

            if ($Resume) {
                Write-Host "Resuming Phi-3.5 pipeline from step $CurrentStep..." -ForegroundColor Green

                # パイプライン再開実行
                $PythonCmd = @"
import sys
sys.path.insert(0, '.')
from scripts.pipeline.phi35_so8t_thinking_pipeline import Phi35SO8TThinkingPipeline
pipeline = Phi35SO8TThinkingPipeline('configs/train_phi35_so8t_annealing.yaml')
pipeline.pipeline_state = $State
pipeline.run_pipeline(resume=True)
"@

                try {
                    $PythonCmd | python
                    Write-Host "Pipeline resumed successfully!" -ForegroundColor Green
                    $FoundIncomplete = $true
                    break  # 最初の未完了セッションのみ再開
                } catch {
                    Write-Host "Pipeline resume failed: $($_.Exception.Message)" -ForegroundColor Red
                }
            }

        } catch {
            Write-Host "  Status: ERROR - Invalid state file" -ForegroundColor Red
        }
    } else {
        Write-Host "  Status: NO STATE FILE" -ForegroundColor Gray
    }

    Write-Host ""
}

if (!$FoundIncomplete) {
    Write-Host "No incomplete Phi-3.5 pipeline sessions found." -ForegroundColor Cyan

    # 新規パイプライン開始の確認
    if ($Force) {
        Write-Host "Starting new Phi-3.5 pipeline session..." -ForegroundColor Green
        & "scripts/pipeline/run_phi35_pipeline.bat"
    } else {
        $StartNew = Read-Host "Start new Phi-3.5 pipeline session? (y/n)"
        if ($StartNew -eq "y" -or $StartNew -eq "Y") {
            & "scripts/pipeline/run_phi35_pipeline.bat"
        }
    }
}

Write-Host "Phi-3.5 power-on resume check completed at $(Get-Date)" -ForegroundColor Green
