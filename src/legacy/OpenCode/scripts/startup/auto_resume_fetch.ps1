# Auto-Resume Citation Fetcher Startup Script
# PowerShellバージョン（推奨）

$ErrorActionPreference = "SilentlyContinue"

$ProjectRoot = "C:\Users\downl\Desktop\SO8T"
$LogFile = "$ProjectRoot\logs\arxiv_100k_fetch.log"
$CheckpointDir = "$ProjectRoot\data\sunset_pipeline\raw\arxiv_citations"
$CheckpointBase = "arxiv_top_100k_2024-2026_checkpoint"

# 最新のチェックポイントを探す
$CheckpointFiles = @(
    "$CheckpointDir\$CheckpointBase.json",
    "$CheckpointDir\$CheckpointBase.1.json",
    "$CheckpointDir\$CheckpointBase.2.json",
    "$CheckpointDir\$CheckpointBase.3.json"
)

$HasCheckpoint = $false
foreach ($cp in $CheckpointFiles) {
    if (Test-Path $cp) {
        $content = Get-Content $cp | ConvertFrom-Json
        if ($content.papers_count -lt $content.max_papers) {
            $HasCheckpoint = $true
            break
        }
    }
}

if ($HasCheckpoint) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "[$timestamp] Auto-resume started from checkpoint"
    
    # バックグラウンドで実行
    Start-Process -NoNewWindow -FilePath "py" -ArgumentList @(
        "-3.12",
        "$ProjectRoot\scripts\data_processing\citation_fetcher.py",
        "--source", "arxiv",
        "--max-papers", "100000",
        "--output", "$ProjectRoot\data\sunset_pipeline\raw\arxiv_citations\arxiv_top_100k_2024-2026.jsonl",
        "--verbose"
    ) -RedirectStandardOutput "$ProjectRoot\logs\arxiv_100k_fetch.log" -RedirectStandardError "$ProjectRoot\logs\arxiv_100k_fetch_error.log"
    
    Write-Host "Citation fetch auto-resumed in background"
} else {
    Write-Host "No incomplete checkpoint found, skipping auto-resume"
}
