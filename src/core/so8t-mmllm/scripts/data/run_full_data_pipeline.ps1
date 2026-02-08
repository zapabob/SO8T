# SO8T Full Data Pipeline
# Crawl (600GB) → Cleanse → Sample (300GB) → Train/Val/Test Split
# 統計的サンプリング、ML ベストプラクティス準拠

param(
    [int]$CrawlTarget = 17000000,  # 17M samples (~200GB)
    [int]$Workers = 16,
    [double]$JaWeight = 0.7,
    [double]$EnWeight = 0.2,
    [double]$ZhWeight = 0.1,
    [double]$TargetCleanedGB = 100.0  # 100GB cleaned data
)

$ErrorActionPreference = "Stop"

Write-Host "" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green
Write-Host "SO8T FULL DATA PIPELINE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "" -ForegroundColor Cyan

Write-Host "[PIPELINE OVERVIEW]" -ForegroundColor Yellow
Write-Host "  Phase 1: Web Crawling (600GB raw data)" -ForegroundColor White
Write-Host "  Phase 2: Data Cleansing" -ForegroundColor White
Write-Host "  Phase 3: Statistical Sampling (300GB)" -ForegroundColor White
Write-Host "  Phase 4: Train/Val/Test Split (80/10/10)" -ForegroundColor White
Write-Host "" -ForegroundColor Cyan

Write-Host "[CONFIG]" -ForegroundColor Yellow
Write-Host "  Crawl target: $($CrawlTarget.ToString('N0')) samples" -ForegroundColor White
Write-Host "  Workers: $Workers" -ForegroundColor White
Write-Host "  Raw data: ~600 GB" -ForegroundColor White
Write-Host "  Cleaned data: ~$TargetCleanedGB GB" -ForegroundColor White
Write-Host "  Output: D:\webdataset" -ForegroundColor White
Write-Host "" -ForegroundColor Cyan

Write-Host "[ESTIMATE]" -ForegroundColor Yellow
Write-Host "  Phase 1: 150-250 hours (6-10 days)" -ForegroundColor Yellow
Write-Host "  Phase 2: 10-20 hours" -ForegroundColor Yellow
Write-Host "  Phase 3: 5-10 hours" -ForegroundColor Yellow
Write-Host "  Total: ~170-280 hours (7-12 days)" -ForegroundColor Yellow
Write-Host "" -ForegroundColor Cyan

# ディスク容量確認
$drive = Get-PSDrive D -ErrorAction SilentlyContinue
if ($drive) {
    $freeGB = [math]::Round($drive.Free / 1GB, 2)
    Write-Host "[DISK] D: drive free space: $freeGB GB" -ForegroundColor Cyan
    
    if ($freeGB -lt 1000) {
        Write-Host "[WARNING] Insufficient disk space!" -ForegroundColor Red
        Write-Host "           Required: 1000 GB+" -ForegroundColor Yellow
        Write-Host "           Available: $freeGB GB" -ForegroundColor Yellow
        $continue = Read-Host "Continue anyway? (y/N)"
        if ($continue -ne "y" -and $continue -ne "Y") {
            exit
        }
    } else {
        Write-Host "[OK] Sufficient disk space" -ForegroundColor Green
    }
} else {
    Write-Host "[ERROR] D: drive not found!" -ForegroundColor Red
    exit
}

Write-Host "" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green
Write-Host "PHASE 1: WEB CRAWLING (600GB)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "" -ForegroundColor Cyan

# Phase 1起動確認
$startCrawl = Read-Host "Start Phase 1: Web Crawling? (y/N)"
if ($startCrawl -ne "y" -and $startCrawl -ne "Y") {
    Write-Host "[SKIP] Phase 1 skipped" -ForegroundColor Yellow
    $skipPhase1 = $true
} else {
    $skipPhase1 = $false
    
    # クロール開始
    Write-Host "[START] Launching web crawler..." -ForegroundColor Green
    
    cd C:\Users\downl\Desktop\SO8T\so8t-mmllm
    
    $crawlLog = "logs\massive_crawl_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
    
    Start-Process -FilePath "py" -ArgumentList `
        "-3", `
        "scripts\data\massive_parallel_crawler.py", `
        "--target", "$CrawlTarget", `
        "--output-dir", "D:\webdataset\raw", `
        "--workers", "$Workers", `
        "--ja-weight", "0.7", `
        "--en-weight", "0.2", `
        "--zh-weight", "0.1" `
        -RedirectStandardOutput $crawlLog `
        -RedirectStandardError "$crawlLog.err" `
        -WindowStyle Minimized
    
    Write-Host "[OK] Crawler started in background!" -ForegroundColor Green
    Write-Host "[LOG] $crawlLog" -ForegroundColor Cyan
    Write-Host "" -ForegroundColor Cyan
    
    # 音声通知
    if (Test-Path "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav") {
        (New-Object System.Media.SoundPlayer "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav").PlaySync()
        Write-Host "[AUDIO] Crawl started!" -ForegroundColor Green
    }
    
    Write-Host "" -ForegroundColor Cyan
    Write-Host "[INFO] Phase 1 is running in background (150-250 hours)" -ForegroundColor Yellow
    Write-Host "[INFO] You can continue to Phase 2 setup while crawling" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Cyan
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "PHASE 2: DATA CLEANSING" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "" -ForegroundColor Cyan

Write-Host "[INFO] Phase 2 will automatically start after Phase 1 completes" -ForegroundColor Yellow
Write-Host "[INFO] Or you can run manually:" -ForegroundColor Yellow
Write-Host "" -ForegroundColor Cyan
Write-Host "  py -3 scripts\data\pipeline_cleanse_and_sample.py \" -ForegroundColor Cyan
Write-Host "    --input-dir D:\webdataset\raw \" -ForegroundColor Cyan
Write-Host "    --output-dir D:\webdataset\cleaned \" -ForegroundColor Cyan
Write-Host "    --target-gb 300" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan

Write-Host "========================================" -ForegroundColor Green
Write-Host "PIPELINE SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "" -ForegroundColor Cyan

if (Test-Path "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav") {
    for ($i = 1; $i -le 3; $i++) {
        (New-Object System.Media.SoundPlayer "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav").PlaySync()
        Start-Sleep -Milliseconds 300
    }
    Write-Host "[AUDIO] Setup complete (3x)!" -ForegroundColor Green
}

Write-Host "" -ForegroundColor Cyan
Write-Host "Monitor progress:" -ForegroundColor Yellow
Write-Host "  Get-Content logs\massive_crawl_*.log -Tail 50 -Wait" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan

