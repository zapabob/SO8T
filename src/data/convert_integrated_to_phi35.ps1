# Phi-3.5 Thinkingフォーマット変換スクリプト
# PowerShell版

param(
    [string]$InputFile = "D:/webdataset/integrated_dataset.jsonl",
    [string]$OutputDir = "D:/webdataset/phi35_integrated",
    [float]$CotWeight = 3.0
)

Write-Host "=== Phi-3.5 Thinking Format Conversion ===" -ForegroundColor Green
Write-Host "Input: $InputFile" -ForegroundColor Cyan
Write-Host "Output: $OutputDir" -ForegroundColor Cyan
Write-Host "CoT Weight: $CotWeight" -ForegroundColor Cyan

# 出力ディレクトリの作成
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Pythonスクリプトの実行
$pythonCmd = @"
import sys
sys.path.insert(0, '.')
from scripts.data.convert_integrated_to_phi35 import main
import sys
sys.argv = ['convert_integrated_to_phi35.py', '--input', '$InputFile', '--output', '$OutputDir', '--cot-weight', '$CotWeight']
main()
"@

Write-Host "Executing Python conversion script..." -ForegroundColor Yellow

try {
    $pythonCmd | python
    Write-Host "Conversion completed successfully!" -ForegroundColor Green
} catch {
    Write-Host "Error during conversion: $($_.Exception.Message)" -ForegroundColor Red
}

# オーディオ通知
Write-Host "Playing completion notification..." -ForegroundColor Green
try {
    & "scripts\utils\play_audio_notification.ps1"
} catch {
    Write-Host "Audio notification failed" -ForegroundColor Yellow
}

Write-Host "Phi-3.5 conversion process completed." -ForegroundColor Green
