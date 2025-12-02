# AEGIS-phi3.5-v2.0 Training Script
# ノーベル賞・フィールズ賞級推論機能を統合したHFモデルのトレーニング

param(
    [int]$Epochs = 3,
    [int]$BatchSize = 1,
    [float]$LearningRate = 1e-5,
    [switch]$Use4Bit = $true,
    [switch]$EnableMathematicalReasoning = $true,
    [string]$ReasoningFormat = "nobel_fields",
    [switch]$TestAfterTraining = $true,
    [switch]$SkipDatasetGeneration = $false
)

# UTF-8エンコーディング設定
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

# スクリプト開始
Write-Host "🎯 AEGIS-phi3.5-v2.0 トレーニング開始" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "モデル名: AEGIS-phi3.5-v2.0" -ForegroundColor Yellow
Write-Host "エポック数: $Epochs" -ForegroundColor Yellow
Write-Host "バッチサイズ: $BatchSize" -ForegroundColor Yellow
Write-Host "学習率: $LearningRate" -ForegroundColor Yellow
Write-Host "4bit量子化: $Use4Bit" -ForegroundColor Yellow
Write-Host "数学推論: $EnableMathematicalReasoning" -ForegroundColor Yellow
Write-Host "推論フォーマット: $ReasoningFormat" -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Cyan

# コマンド構築
$command = "py scripts/run_aegis_phi35_v2_training.py"
$command += " --model_name AEGIS-phi3.5-v2.0"
$command += " --epochs $Epochs"
$command += " --batch_size $BatchSize"
$command += " --learning_rate $LearningRate"

if ($Use4Bit) {
    $command += " --use_4bit"
}

if ($EnableMathematicalReasoning) {
    $command += " --enable_mathematical_reasoning"
}

$command += " --reasoning_format $ReasoningFormat"

if ($TestAfterTraining) {
    $command += " --test_after_training"
}

if ($SkipDatasetGeneration) {
    $command += " --skip_dataset_generation"
}

Write-Host "実行コマンド: $command" -ForegroundColor Green
Write-Host ""

try {
    # コマンド実行
    Invoke-Expression $command

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ AEGIS-phi3.5-v2.0 トレーニング成功！" -ForegroundColor Green
        Write-Host "HFモデルにノーベル賞・フィールズ賞級の推論機能が統合されました。" -ForegroundColor Green

        # オーディオ通知
        $audioFile = "$PSScriptRoot\..\.cursor\marisa_owattaze.wav"
        if (Test-Path $audioFile) {
            Write-Host "[AUDIO] トレーニング完了通知を再生中..." -ForegroundColor Green
            try {
                Add-Type -AssemblyName System.Windows.Forms
                $player = New-Object System.Media.SoundPlayer $audioFile
                $player.PlaySync()
                Write-Host "[OK] オーディオ通知再生成功" -ForegroundColor Green
            } catch {
                Write-Host "[WARNING] オーディオ再生失敗: $($_.Exception.Message)" -ForegroundColor Yellow
                [System.Console]::Beep(1000, 500)
            }
        } else {
            Write-Host "[WARNING] オーディオファイルが見つかりません" -ForegroundColor Yellow
            [System.Console]::Beep(1000, 500)
        }

    } else {
        Write-Host "`n❌ AEGISトレーニング失敗 (終了コード: $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }

} catch {
    Write-Host "`n❌ スクリプト実行エラー: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "`n🎉 AEGIS-phi3.5-v2.0 トレーニング完了！" -ForegroundColor Cyan
Write-Host "高度知能AIシステムの統合が完了しました。" -ForegroundColor Cyan
