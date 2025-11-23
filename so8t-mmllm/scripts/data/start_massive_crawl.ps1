# SO8T Massive Parallel Web Crawler - PowerShell Launcher
# UTF-8 encoding, background execution, progress monitoring

param(
    [int]$Target = 10000000,
    [int]$Workers = 16,
    [double]$JaWeight = 0.7,
    [double]$EnWeight = 0.2,
    [double]$ZhWeight = 0.1
)

Write-Host "" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green
Write-Host "SO8T MASSIVE PARALLEL WEB CRAWLER" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "" -ForegroundColor Cyan

Write-Host "[CONFIG]" -ForegroundColor Yellow
Write-Host "  Target: $($Target.ToString('N0')) samples" -ForegroundColor White
Write-Host "  Workers: $Workers parallel processes" -ForegroundColor White
Write-Host "  Language weights:" -ForegroundColor White
Write-Host "    Japanese: $($JaWeight * 100)%" -ForegroundColor Green
Write-Host "    English: $($EnWeight * 100)%" -ForegroundColor White
Write-Host "    Chinese: $($ZhWeight * 100)%" -ForegroundColor White
Write-Host "" -ForegroundColor Cyan

Write-Host "[SOURCES]" -ForegroundColor Yellow
Write-Host "  - National Diet Library" -ForegroundColor White
Write-Host "  - Wikipedia (JA/EN/ZH)" -ForegroundColor White
Write-Host "  - Government ministries (12+)" -ForegroundColor White
Write-Host "  - eGov (electronic government)" -ForegroundColor White
Write-Host "  - Nikkei 225 companies (37+)" -ForegroundColor White
Write-Host "  - Culture & education sites" -ForegroundColor White
Write-Host "" -ForegroundColor Cyan

Write-Host "[ESTIMATE]" -ForegroundColor Yellow
Write-Host "  Duration: 50-100 hours" -ForegroundColor Yellow
Write-Host "  Data size: 50-100 GB" -ForegroundColor Yellow
Write-Host "  Disk required: 150 GB+" -ForegroundColor Red
Write-Host "" -ForegroundColor Cyan

# ディスク容量確認
$drive = Get-PSDrive C
$freeGB = [math]::Round($drive.Free / 1GB, 2)

if ($freeGB -lt 150) {
    Write-Host "[WARNING] Low disk space: $freeGB GB" -ForegroundColor Red
    Write-Host "           Recommended: 150 GB+" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Cyan
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        Write-Host "[CANCELLED] Crawler not started" -ForegroundColor Red
        exit
    }
} else {
    Write-Host "[OK] Disk space: $freeGB GB" -ForegroundColor Green
}

Write-Host "" -ForegroundColor Cyan
Write-Host "[START] Launching massive crawler..." -ForegroundColor Green

# カレントディレクトリ設定
Set-Location C:\Users\downl\Desktop\SO8T\so8t-mmllm

# ログファイル
$logFile = "logs\massive_crawl_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$errFile = "$logFile.err"

# バックグラウンド起動
Start-Process -FilePath "py" -ArgumentList `
    "-3", `
    "scripts\data\massive_parallel_crawler.py", `
    "--target", "$Target", `
    "--workers", "$Workers", `
    "--ja-weight", "$JaWeight", `
    "--en-weight", "$EnWeight", `
    "--zh-weight", "$ZhWeight" `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError $errFile `
    -WindowStyle Minimized

Write-Host "[OK] Crawler started in background!" -ForegroundColor Green
Write-Host "" -ForegroundColor Cyan
Write-Host "[LOGS]" -ForegroundColor Yellow
Write-Host "  Output: $logFile" -ForegroundColor White
Write-Host "  Errors: $errFile" -ForegroundColor White
Write-Host "" -ForegroundColor Cyan

Write-Host "[MONITOR]" -ForegroundColor Yellow
Write-Host "  Real-time log:" -ForegroundColor White
Write-Host "    Get-Content '$logFile' -Tail 50 -Wait" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan
Write-Host "  Process status:" -ForegroundColor White
Write-Host "    Get-Process python | Where-Object {`$_.WorkingSet -gt 100MB}" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan

Write-Host "[STOP]" -ForegroundColor Yellow
Write-Host "  To stop crawling:" -ForegroundColor White
Write-Host "    Get-Process python | Where-Object {`$_.MainWindowTitle -like '*MassiveCrawl*'} | Stop-Process" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan

# 音声通知
if (Test-Path "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav") {
    Write-Host "[AUDIO] Playing start notification..." -ForegroundColor Yellow
    (New-Object System.Media.SoundPlayer "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav").PlaySync()
    Write-Host "[AUDIO] Notification played!" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Audio file not found" -ForegroundColor Yellow
}

Write-Host "" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green
Write-Host "MASSIVE CRAWL RUNNING IN BACKGROUND" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "" -ForegroundColor Cyan

# 5秒待機してログ確認
Write-Host "[WAIT] Waiting 5 seconds for initialization..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host "" -ForegroundColor Cyan
Write-Host "[LOG] Initial output:" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Cyan
Get-Content $logFile -Tail 20 -ErrorAction SilentlyContinue
Write-Host "----------------------------------------" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan

Write-Host "[SUCCESS] Crawler is running!" -ForegroundColor Green
Write-Host "Check progress regularly with: Get-Content '$logFile' -Tail 30 -Wait" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan


































