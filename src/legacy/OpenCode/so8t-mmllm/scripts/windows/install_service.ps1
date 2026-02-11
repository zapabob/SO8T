# SO8T Windows常駐サービスインストールスクリプト
# バックグラウンド常駐、自動起動、システムトレイ、ホットキー

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Start,
    [switch]$Stop,
    [switch]$Status
)

$ServiceName = "SO8TAgent"
$ServiceDisplayName = "SO8T Secure Agent"
$ServiceDescription = "SO8T統合セキュアエージェント - 防衛・航空宇宙・運輸向けLLMOps"
$ServicePath = Join-Path $PSScriptRoot "..\..\src\agents\windows_service.py"
$PythonPath = (Get-Command py).Source

Write-Host "[INFO] SO8T Windows Service Management" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

function Install-SO8TService {
    Write-Host "`n[INSTALL] Installing SO8T Service..." -ForegroundColor Green
    
    # Python スクリプト確認
    if (-not (Test-Path $ServicePath)) {
        Write-Host "[ERROR] Service script not found: $ServicePath" -ForegroundColor Red
        return $false
    }
    
    # NSS M (Non-Sucking Service Manager) を使用
    # または sc.exe でサービス登録
    
    $ServiceCommand = "py -3 `"$ServicePath`" --service"
    
    try {
        # サービス登録
        sc.exe create $ServiceName binPath= "`"$PythonPath`" -3 `"$ServicePath`" --service" `
            DisplayName= $ServiceDisplayName `
            start= auto
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Service installed successfully" -ForegroundColor Green
            
            # サービス説明設定
            sc.exe description $ServiceName $ServiceDescription
            
            # 自動起動設定
            sc.exe config $ServiceName start= auto
            
            Write-Host "[INFO] Service will start automatically on boot" -ForegroundColor Cyan
            return $true
        } else {
            Write-Host "[ERROR] Service installation failed" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "[ERROR] Installation failed: $_" -ForegroundColor Red
        return $false
    }
}

function Uninstall-SO8TService {
    Write-Host "`n[UNINSTALL] Removing SO8T Service..." -ForegroundColor Yellow
    
    try {
        # サービス停止
        Stop-SO8TService
        
        # サービス削除
        sc.exe delete $ServiceName
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Service uninstalled successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[ERROR] Service uninstall failed" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "[ERROR] Uninstall failed: $_" -ForegroundColor Red
        return $false
    }
}

function Start-SO8TService {
    Write-Host "`n[START] Starting SO8T Service..." -ForegroundColor Green
    
    try {
        sc.exe start $ServiceName
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Service started successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[ERROR] Service start failed" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "[ERROR] Start failed: $_" -ForegroundColor Red
        return $false
    }
}

function Stop-SO8TService {
    Write-Host "`n[STOP] Stopping SO8T Service..." -ForegroundColor Yellow
    
    try {
        sc.exe stop $ServiceName
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Service stopped successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[WARNING] Service may not be running" -ForegroundColor Yellow
            return $false
        }
    } catch {
        Write-Host "[ERROR] Stop failed: $_" -ForegroundColor Red
        return $false
    }
}

function Get-SO8TServiceStatus {
    Write-Host "`n[STATUS] Checking SO8T Service status..." -ForegroundColor Cyan
    
    try {
        $Service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
        
        if ($Service) {
            Write-Host "`nService Information:" -ForegroundColor Cyan
            Write-Host "  Name: $($Service.Name)"
            Write-Host "  Display Name: $($Service.DisplayName)"
            Write-Host "  Status: $($Service.Status)" -ForegroundColor $(if ($Service.Status -eq 'Running') { 'Green' } else { 'Yellow' })
            Write-Host "  Start Type: $($Service.StartType)"
            
            # プロセス情報
            if ($Service.Status -eq 'Running') {
                $Process = Get-Process | Where-Object { $_.ProcessName -like "*python*" -and $_.CommandLine -like "*windows_service.py*" } | Select-Object -First 1
                if ($Process) {
                    Write-Host "`nProcess Information:"
                    Write-Host "  PID: $($Process.Id)"
                    Write-Host "  Memory: $([math]::Round($Process.WorkingSet64 / 1MB, 2)) MB"
                    Write-Host "  CPU Time: $($Process.TotalProcessorTime)"
                }
            }
            
            return $true
        } else {
            Write-Host "[INFO] Service not installed" -ForegroundColor Yellow
            return $false
        }
    } catch {
        Write-Host "[ERROR] Status check failed: $_" -ForegroundColor Red
        return $false
    }
}

function Show-Usage {
    Write-Host @"

Usage: install_service.ps1 [OPTION]

Options:
  -Install    Install SO8T service
  -Uninstall  Uninstall SO8T service
  -Start      Start SO8T service
  -Stop       Stop SO8T service
  -Status     Show service status

Examples:
  .\install_service.ps1 -Install
  .\install_service.ps1 -Start
  .\install_service.ps1 -Status

"@
}

# メイン処理
if ($Install) {
    Install-SO8TService
} elseif ($Uninstall) {
    Uninstall-SO8TService
} elseif ($Start) {
    Start-SO8TService
} elseif ($Stop) {
    Stop-SO8TService
} elseif ($Status) {
    Get-SO8TServiceStatus
} else {
    Show-Usage
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host ""

# 音声通知
if (Test-Path "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav") {
    Add-Type -AssemblyName System.Windows.Forms
    $player = New-Object System.Media.SoundPlayer "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
    $player.Play()
    Write-Host "[OK] Audio notification played" -ForegroundColor Green
}

