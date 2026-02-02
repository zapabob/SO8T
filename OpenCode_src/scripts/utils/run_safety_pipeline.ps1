# Safety-Aware SO8T Pipeline Runner
# Windows PowerShell用の安全重視SO8Tパイプライン実行スクリプト

param(
    [string]$Config = "configs/train_safety.yaml",
    [string]$DataDir = "data",
    [string]$OutputDir = "chk",
    [int]$Seed = 42,
    [switch]$NoResume = $false,
    [switch]$ShowStructure = $true
)

Write-Host "🚀 Safety-Aware SO8T Pipeline Runner" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# プロジェクト構造を表示
if ($ShowStructure) {
    Write-Host "`n📁 Project Structure:" -ForegroundColor Yellow
    Get-ChildItem -Recurse -File | Select-Object FullName | Format-Table -AutoSize
    Write-Host ""
}

# パラメータを表示
Write-Host "📋 Pipeline Parameters:" -ForegroundColor Cyan
Write-Host "  Config: $Config"
Write-Host "  Data Directory: $DataDir"
Write-Host "  Output Directory: $OutputDir"
Write-Host "  Seed: $Seed"
Write-Host "  No Resume: $NoResume"
Write-Host ""

# 必要なファイルの存在確認
$requiredFiles = @(
    "agents/cli.py",
    "train_safety.py",
    "visualize_safety_training.py",
    "test_safety_inference.py",
    "scripts/impl_logger.py"
)

Write-Host "🔍 Checking required files..." -ForegroundColor Yellow
$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
        Write-Host "  ❌ Missing: $file" -ForegroundColor Red
    } else {
        Write-Host "  ✅ Found: $file" -ForegroundColor Green
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "`n❌ Missing required files. Please ensure all files are present." -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ All required files found!" -ForegroundColor Green

# パイプライン実行
Write-Host "`n🚀 Starting Safety-Aware SO8T Pipeline..." -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

try {
    # パイプライン実行
    $pipelineArgs = @(
        "-m", "agents.cli", "pipeline-safety",
        "--config", $Config,
        "--data_dir", $DataDir,
        "--output_dir", $OutputDir,
        "--seed", $Seed
    )
    
    if ($NoResume) {
        $pipelineArgs += "--no_resume"
    }
    
    Write-Host "`n📚 Executing: py -3 $($pipelineArgs -join ' ')" -ForegroundColor Cyan
    
    # パイプライン実行
    $result = & py -3 @pipelineArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n🎉 Pipeline completed successfully!" -ForegroundColor Green
        
    # 結果ファイルの確認
    Write-Host "`n📊 Generated Files:" -ForegroundColor Yellow
    $resultFiles = @(
        "$OutputDir/safety_model_best.pt",
        "$OutputDir/safety_training_log.jsonl",
        "$OutputDir/safety_visualizations/",
        "$OutputDir/safety_test_results/",
        "$OutputDir/safety_demonstration_results/",
        "_docs/"
    )
        
        foreach ($file in $resultFiles) {
            if (Test-Path $file) {
                Write-Host "  ✅ $file" -ForegroundColor Green
            } else {
                Write-Host "  ❌ $file (not found)" -ForegroundColor Red
            }
        }
        
        # 実装ログの確認
        $logFiles = Get-ChildItem "_docs" -Filter "*安全重視SO8T*.md" -ErrorAction SilentlyContinue
        if ($logFiles) {
            Write-Host "`n📝 Implementation Logs:" -ForegroundColor Yellow
            foreach ($logFile in $logFiles) {
                Write-Host "  📄 $($logFile.Name)" -ForegroundColor Green
            }
        }
        
        Write-Host "`n🎯 Safety-Aware SO8T Pipeline completed successfully!" -ForegroundColor Green
        Write-Host "   Check the output files for detailed results." -ForegroundColor White
        
    } else {
        Write-Host "`n❌ Pipeline failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    
} catch {
    Write-Host "`n❌ Pipeline execution failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "`n🏁 Script completed." -ForegroundColor Green
