# ==========================================
# AEGIS完全自動無人化システムセットアップ
# Windowsタスクスケジューラ自動登録スクリプト
# ==========================================

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Test,
    [string]$TaskName = "AEGIS_Autonomous_Training",
    [string]$UserName = $env:USERNAME
)

# 管理者権限チェック
function Test-Admin {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# タスク作成関数
function Install-AEGISTask {
    Write-Host "🔥 Installing AEGIS Autonomous Training System..." -ForegroundColor Cyan

    # プロジェクトディレクトリ
    if ($MyInvocation.MyCommand.Path) {
        $projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    } else {
        $projectDir = $PSScriptRoot
    }
    if (-not $projectDir -or $projectDir -notlike "*SO8T*") {
        $projectDir = "C:\Users\$env:USERNAME\Desktop\SO8T"
    }

    # 実行ファイルパス
    $actionPath = Join-Path $projectDir "auto_aegis_pipeline.bat"
    $workingDir = $projectDir

    Write-Host "   Project Dir: $projectDir"
    Write-Host "   Action Path: $actionPath"
    Write-Host "   Working Dir: $workingDir"

    # タスクが存在するかチェック
    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        Write-Host "   Removing existing task..."
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    # タスク作成
    $action = New-ScheduledTaskAction -Execute $actionPath -WorkingDirectory $workingDir

    # トリガー設定（システム起動時 + 毎日午前2時）
    $triggers = @()
    $triggers += New-ScheduledTaskTrigger -AtStartup
    $triggers += New-ScheduledTaskTrigger -Daily -At "02:00"

    # 設定
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable `
        -DontStopOnIdleEnd `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 5)

    # プリンシパル（最高権限で実行）
    $principal = New-ScheduledTaskPrincipal -UserId $UserName -LogonType Interactive -RunLevel Highest

    # タスク登録
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $triggers `
        -Settings $settings `
        -Principal $principal `
        -Description "AEGIS Autonomous Training System - Runs SO8T training pipeline automatically"

    Write-Host "✅ AEGIS Task installed successfully!" -ForegroundColor Green
    Write-Host "   Task will run at system startup and daily at 2:00 AM" -ForegroundColor Green
}

# タスク削除関数
function Uninstall-AEGISTask {
    Write-Host "🗑️ Uninstalling AEGIS Autonomous Training System..." -ForegroundColor Yellow

    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "✅ AEGIS Task uninstalled successfully!" -ForegroundColor Green
    } else {
        Write-Host "⚠️ AEGIS Task not found" -ForegroundColor Yellow
    }
}

# テスト関数
function Test-AEGISSystem {
    Write-Host "🧪 Testing AEGIS Autonomous System..." -ForegroundColor Cyan

    # Python環境チェック
    try {
        $pythonVersion = python --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
        } else {
            throw "Python not found"
        }
    } catch {
        Write-Host "❌ Python not found or not in PATH" -ForegroundColor Red
        return $false
    }

    # PyTorchチェック
    try {
        $torchCheck = python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ PyTorch: $torchCheck" -ForegroundColor Green
        }
    } catch {
        Write-Host "❌ PyTorch import failed" -ForegroundColor Red
        return $false
    }

    # CUDAチェック
    try {
        $cudaCheck = python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ CUDA: $cudaCheck" -ForegroundColor Green
        }
    } catch {
        Write-Host "❌ CUDA check failed" -ForegroundColor Red
    }

    # プロジェクトファイルチェック
    $projectDir = "C:\Users\$env:USERNAME\Desktop\SO8T"
    $requiredFiles = @(
        "auto_aegis_pipeline.bat",
        "scripts\training\rlpo_science_nsfw_automated.py",
        "scripts\utils\task_manager.py",
        "simple_rlpo_test.py"
    )

    foreach ($file in $requiredFiles) {
        $filePath = Join-Path $projectDir $file
        if (Test-Path $filePath) {
            Write-Host "✅ File exists: $file" -ForegroundColor Green
        } else {
            Write-Host "❌ File missing: $file" -ForegroundColor Red
            return $false
        }
    }

    # タスクステータスチェック
    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        $taskState = $existingTask.State
        Write-Host "✅ Scheduled Task: $TaskName (State: $taskState)" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Scheduled Task not found: $TaskName" -ForegroundColor Yellow
    }

    Write-Host "🎉 System test completed!" -ForegroundColor Green
    return $true
}

# メイン処理
if (-not (Test-Admin)) {
    Write-Host "❌ Administrator privileges required. Please run as administrator." -ForegroundColor Red
    exit 1
}

if ($Install) {
    Install-AEGISTask
} elseif ($Uninstall) {
    Uninstall-AEGISTask
} elseif ($Test) {
    $testResult = Test-AEGISSystem
    if (-not $testResult) {
        exit 1
    }
} else {
    Write-Host "AEGIS Autonomous Training System Setup" -ForegroundColor Cyan
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\setup_autonomous_system.ps1 -Install    # Install autonomous system"
    Write-Host "  .\setup_autonomous_system.ps1 -Uninstall  # Uninstall autonomous system"
    Write-Host "  .\setup_autonomous_system.ps1 -Test       # Test system components"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\setup_autonomous_system.ps1 -Install"
    Write-Host "  .\setup_autonomous_system.ps1 -Test"
    Write-Host ""
}
