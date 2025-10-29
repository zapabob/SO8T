# Safety-Aware SO8T Complete Pipeline Runner (PowerShell)
# CLIなしで学習推論実証を完全実行するPowerShellスクリプト

param(
    [string]$Config = "configs/train_safety.yaml",
    [string]$DataDir = "data",
    [string]$OutputDir = "chk",
    [int]$Seed = 42,
    [switch]$NoResume = $false,
    [switch]$SkipTraining = $false,
    [switch]$SkipVisualization = $false,
    [switch]$SkipTesting = $false,
    [switch]$SkipDemonstration = $false
)

Write-Host "================================================================================" -ForegroundColor Green
Write-Host "🚀 Safety-Aware SO8T Complete Pipeline Runner" -ForegroundColor Green
Write-Host "   学習推論実証の完全実行システム" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""

# Pythonの存在確認
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+ and try again." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# 必要なファイルの確認
$requiredFiles = @(
    "train_safety.py",
    "visualize_safety_training.py",
    "test_safety_inference.py", 
    "demonstrate_safety_inference.py",
    "scripts/impl_logger.py",
    "configs/train_safety.yaml"
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
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "`n✅ All required files found!" -ForegroundColor Green

# パイプライン実行
Write-Host "`n🚀 Starting Safety-Aware SO8T Pipeline..." -ForegroundColor Green
Write-Host ""

$pipelineArgs = @(
    "run_safety_complete.py",
    "--config", $Config,
    "--data_dir", $DataDir,
    "--output_dir", $OutputDir,
    "--seed", $Seed
)

if ($NoResume) {
    $pipelineArgs += "--no_resume"
}
if ($SkipTraining) {
    $pipelineArgs += "--skip_training"
}
if ($SkipVisualization) {
    $pipelineArgs += "--skip_visualization"
}
if ($SkipTesting) {
    $pipelineArgs += "--skip_testing"
}
if ($SkipDemonstration) {
    $pipelineArgs += "--skip_demonstration"
}

try {
    $result = & python @pipelineArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n🎉 Pipeline completed successfully!" -ForegroundColor Green
        Write-Host "📁 Check the output files for detailed results." -ForegroundColor White
        
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
        
    } else {
        Write-Host "`n❌ Pipeline failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
    
} catch {
    Write-Host "`n❌ Pipeline execution failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n🏁 Script completed." -ForegroundColor Green
Read-Host "Press Enter to exit"
