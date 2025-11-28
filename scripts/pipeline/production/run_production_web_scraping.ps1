# SO8T Webスクレイピング 本番環境実行 PowerShell版
# UTF-8エンコーディングで実行

param(
    [string]$ConfigFile = "configs\complete_automated_ab_pipeline.yaml",
    [switch]$SkipEnvironmentCheck = $false,
    [switch]$SkipConfigValidation = $false
)

# エンコーディング設定
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "SO8T Webスクレイピング 本番環境実行 (PowerShell版)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# 環境設定
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$CONFIG_FILE = Join-Path $PROJECT_ROOT $ConfigFile
$LOG_DIR = Join-Path $PROJECT_ROOT "logs"
$PYTHON = Join-Path $PROJECT_ROOT "venv\Scripts\python.exe"

# Pythonパスの確認
if (-not (Test-Path $PYTHON)) {
    $PYTHON = "python"
}

# ログディレクトリの作成
if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null
}

# タイムスタンプ
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

Write-Host "[INFO] 本番環境実行開始: $timestamp" -ForegroundColor Green
Write-Host "[INFO] プロジェクトルート: $PROJECT_ROOT" -ForegroundColor White
Write-Host "[INFO] 設定ファイル: $CONFIG_FILE" -ForegroundColor White
Write-Host "[INFO] ログディレクトリ: $LOG_DIR" -ForegroundColor White
Write-Host ""

# カレントディレクトリをプロジェクトルートに変更
Set-Location $PROJECT_ROOT

# 設定ファイルの確認
if (-not (Test-Path $CONFIG_FILE)) {
    Write-Host "[ERROR] 設定ファイルが見つかりません: $CONFIG_FILE" -ForegroundColor Red
    exit 1
}

# Phase 1: 環境確認
if (-not $SkipEnvironmentCheck) {
    Write-Host "[PHASE 1] 環境確認" -ForegroundColor Yellow
    Write-Host "[CHECK] 本番環境準備確認..." -ForegroundColor White
    
    $envCheckResult = & $PYTHON scripts\utils\check_production_environment.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARNING] 環境確認で警告がありますが、続行します" -ForegroundColor Yellow
    }
    Write-Host ""
}

# Phase 2: 設定ファイル検証
if (-not $SkipConfigValidation) {
    Write-Host "[PHASE 2] 設定ファイル検証" -ForegroundColor Yellow
    Write-Host "[CHECK] 設定ファイル検証..." -ForegroundColor White
    
    $configValidationResult = & $PYTHON scripts\utils\validate_config.py --config $CONFIG_FILE
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] 設定ファイル検証に失敗しました" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
}

# Phase 3: 本番環境実行
Write-Host "[PHASE 3] 本番環境実行（段階的スケールアップ）" -ForegroundColor Yellow
Write-Host "[INFO] Phase 1（webスクレイピング）を実行します" -ForegroundColor White
Write-Host "[INFO] この処理は長時間かかる場合があります" -ForegroundColor White
Write-Host "[INFO] チェックポイントが自動的に保存されます" -ForegroundColor White
Write-Host ""

$logFile = Join-Path $LOG_DIR "production_web_scraping_$timestamp.log"

# パイプライン実行
try {
    & $PYTHON scripts\pipelines\complete_data_pipeline.py --config $CONFIG_FILE 2>&1 | Tee-Object -FilePath $logFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "================================================================================" -ForegroundColor Green
        Write-Host "[SUCCESS] 本番環境実行完了" -ForegroundColor Green
        Write-Host "================================================================================" -ForegroundColor Green
        Write-Host "[INFO] 完了時刻: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor White
        Write-Host "[INFO] ログファイル: $logFile" -ForegroundColor White
        Write-Host "[INFO] 出力ディレクトリ: D:\webdataset" -ForegroundColor White
        Write-Host ""
        
        # 音声通知
        $audioFile = Join-Path $PROJECT_ROOT ".cursor\marisa_owattaze.wav"
        if (Test-Path $audioFile) {
            Add-Type -AssemblyName System.Windows.Forms
            $player = New-Object System.Media.SoundPlayer $audioFile
            $player.PlaySync()
            Write-Host "[OK] 音声通知送信完了" -ForegroundColor Green
        }
        
        exit 0
    } else {
        Write-Host ""
        Write-Host "================================================================================" -ForegroundColor Red
        Write-Host "[ERROR] 本番環境実行中にエラーが発生しました" -ForegroundColor Red
        Write-Host "================================================================================" -ForegroundColor Red
        Write-Host "[INFO] ログファイルを確認してください: $logFile" -ForegroundColor White
        Write-Host "[INFO] チェックポイントから復旧できます" -ForegroundColor White
        Write-Host ""
        
        # エラー音声通知
        $audioFile = Join-Path $PROJECT_ROOT ".cursor\marisa_owattaze.wav"
        if (Test-Path $audioFile) {
            Add-Type -AssemblyName System.Windows.Forms
            $player = New-Object System.Media.SoundPlayer $audioFile
            $player.PlaySync()
            Write-Host "[ERROR] エラー通知送信完了" -ForegroundColor Red
        }
        
        exit 1
    }
} catch {
    Write-Host "[ERROR] 実行エラー: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

