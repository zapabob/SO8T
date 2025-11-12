# ========================================
# SO8T Auto Resume Setup Script (Quick Launch)
# プロジェクトディレクトリに移動してから実行
# ========================================

# プロジェクトルートパス
$PROJECT_ROOT = "C:\Users\downl\Desktop\SO8T"

# プロジェクトディレクトリに移動
if (Test-Path $PROJECT_ROOT) {
    Set-Location $PROJECT_ROOT
    Write-Host "[INFO] Changed to project directory: $PROJECT_ROOT" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Project directory not found: $PROJECT_ROOT" -ForegroundColor Red
    Write-Host "[ERROR] Please update PROJECT_ROOT in this script." -ForegroundColor Red
    exit 1
}

# セットアップスクリプトを実行
$setupScript = Join-Path $PROJECT_ROOT "scripts\setup_auto_resume.ps1"
if (Test-Path $setupScript) {
    Write-Host "[INFO] Executing setup script..." -ForegroundColor Green
    & $setupScript
} else {
    Write-Host "[ERROR] Setup script not found: $setupScript" -ForegroundColor Red
    exit 1
}

