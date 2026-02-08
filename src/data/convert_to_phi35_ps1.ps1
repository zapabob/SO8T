# Phi-3.5 Thinkingフォーマット変換 (PowerShellネイティブ版)

param(
    [string]$InputFile = "D:/webdataset/integrated_dataset.jsonl",
    [string]$OutputDir = "D:/webdataset/phi35_integrated",
    [float]$CotWeight = 3.0
)

Write-Host "=== Phi-3.5 Thinking Format Conversion (PowerShell) ===" -ForegroundColor Green
Write-Host "Input: $InputFile" -ForegroundColor Cyan
Write-Host "Output: $OutputDir" -ForegroundColor Cyan
Write-Host "CoT Weight: $CotWeight" -ForegroundColor Cyan

# 出力ディレクトリの作成
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "Created output directory: $OutputDir" -ForegroundColor Green
}

# 統計変数
$stats = @{
    total = 0
    converted = 0
    cotWeighted = 0
    finalSamples = 0
}

# Phi-3.5変換関数
function Convert-ToPhi35Format {
    param([string]$text, [string]$datasetName)

    if ([string]::IsNullOrWhiteSpace($text) -or $text.Length -lt 10) {
        return $null
    }

    # データセットタイプ分類
    $datasetType = "General_Task"
    $nameLower = $datasetName.ToLower()
    $textLower = $text.ToLower()

    if ($nameLower -match "reasoning|thinking|cot|chain") {
        $datasetType = "CoT_Reasoning"
    } elseif ($nameLower -match "math|gsm8k|mmlu") {
        $datasetType = "CoT_Math"
    } elseif ($nameLower -match "code|programming|starcoder") {
        $datasetType = "CoT_Coding"
    } elseif ($nameLower -match "safety|ethics|nsfw") {
        $datasetType = "Safety_Ethics"
    }

    # Thinking構造構築
    $thinkingParts = @{
        task = "Task understood."
        safety = "Safety check passed."
        logic = "Applying logical reasoning."
        ethics = "Considering ethical aspects."
        practical = "Considering practical aspects."
        creative = "Exploring creative solutions."
    }

    # データセットタイプ別カスタマイズ
    if ($datasetType -match "CoT_") {
        $thinkingParts.task = "$($datasetType.Split('_')[1]) reasoning task understood."
        $thinkingParts.logic = "Applying step-by-step reasoning process."
    } elseif ($datasetType -eq "Safety_Ethics") {
        $thinkingParts.safety = "Carefully evaluating safety and ethical implications."
        $thinkingParts.ethics = "Carefully considering ethical implications."
    }

    # Thinkingテキスト生成
    $thinkingText = ""
    foreach ($part in $thinkingParts.GetEnumerator()) {
        $thinkingText += "<think-$($part.Key)>$($part.Value)</think-$($part.Key)>`n"
    }

    # 最終回答生成
    $finalAnswer = if ($text.Length -gt 200) { $text.Substring(0, 200) + "..." } else { $text }

    $phi35Format = "$thinkingText<final>$finalAnswer</final>"

    return @{
        source_dataset = $datasetName
        original_text = $text.Substring(0, [Math]::Min(1000, $text.Length))
        phi35_thinking = $phi35Format
        dataset_type = $datasetType
        is_cot = $datasetType.StartsWith("CoT_")
        language = "unknown"
        processing_timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
    }
}

function Test-IsCotSample {
    param([string]$datasetName, [string]$text)

    $cotIndicators = @("reasoning", "thinking", "cot", "chain", "math", "code", "calculate", "solve", "explain", "analyze")

    foreach ($indicator in $cotIndicators) {
        if ($datasetName.ToLower().Contains($indicator) -or $text.ToLower().Contains($indicator)) {
            return $true
        }
    }
    return $false
}

# メイン処理
Write-Host "Loading integrated dataset..." -ForegroundColor Yellow

$samples = @()
$phi35Samples = @()

try {
    $lines = Get-Content $InputFile -Encoding UTF8
    Write-Host "Loaded $($lines.Count) lines from input file" -ForegroundColor Green

    foreach ($line in $lines) {
        $stats.total++
        try {
            $sample = $line | ConvertFrom-Json

            # Phi-3.5フォーマットに変換
            $phi35Sample = Convert-ToPhi35Format -text $sample.text -datasetName $sample.dataset

            if ($phi35Sample) {
                $stats.converted++
                $phi35Samples += $phi35Sample

                # CoTデータは重みづけ
                if (Test-IsCotSample -datasetName $sample.dataset -text $sample.text) {
                    for ($i = 1; $i -lt $CotWeight; $i++) {
                        $phi35Samples += $phi35Sample.PSObject.Copy()
                        $stats.cotWeighted++
                    }
                }
            }
        } catch {
            Write-Host "Error processing line $($stats.total): $($_.Exception.Message)" -ForegroundColor Red
        }

        # 進捗表示
        if ($stats.total % 1000 -eq 0) {
            Write-Host "Processed $($stats.total) samples..." -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "Error loading input file: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# シャッフル
Write-Host "Shuffling samples..." -ForegroundColor Yellow
$phi35Samples = $phi35Samples | Sort-Object { Get-Random }

# 保存
$outputFile = Join-Path $OutputDir "phi35_ppo_optimized_integrated.jsonl"
Write-Host "Saving $($phi35Samples.Count) samples to $outputFile" -ForegroundColor Green

$phi35Samples | ConvertTo-Json -Compress -Depth 10 | Out-File $outputFile -Encoding UTF8

# 統計情報保存
$stats.finalSamples = $phi35Samples.Count
$stats.cot_weight_multiplier = $CotWeight
$stats.processing_timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")

$statsFile = Join-Path $OutputDir "phi35_conversion_stats.json"
$stats | ConvertTo-Json -Depth 10 | Out-File $statsFile -Encoding UTF8

# レポート生成
$reportFile = Join-Path $OutputDir "phi35_conversion_report.md"
$report = @"
# Phi-3.5 Thinking Format Conversion Report

## 概要
- **処理日時**: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))
- **元サンプル数**: $($stats.total)
- **変換完了**: $($stats.converted)
- **CoT重みづけ**: $($stats.cotWeighted)
- **最終データセット**: $($stats.finalSamples)

## Phi-3.5 Thinkingフォーマット
```
<think-task>タスク理解</think-task>
<think-safety>安全性評価</think-safety>
<think-logic>論理的思考</think-logic>
<think-ethics>倫理的考慮</think-ethics>
<think-practical>実用的考察</think-practical>
<think-creative>創造的アプローチ</think-creative>
<final>最終回答</final>
```

## PPO最適化特徴
- CoTデータ重みづけ係数: $CotWeight
- シャッフル適用: 有効
- CoTサンプル判定: 自動

## 出力ファイル
- `phi35_ppo_optimized_integrated.jsonl`: PPO最適化統合データセット
- `phi35_conversion_stats.json`: 統計情報
"@

$report | Out-File $reportFile -Encoding UTF8

Write-Host "=== Conversion Summary ===" -ForegroundColor Green
Write-Host "Original samples: $($stats.total)" -ForegroundColor Cyan
Write-Host "Converted samples: $($stats.converted)" -ForegroundColor Cyan
Write-Host "CoT weighted: $($stats.cotWeighted)" -ForegroundColor Cyan
Write-Host "Final dataset: $($stats.finalSamples)" -ForegroundColor Cyan
Write-Host "Report saved to: $reportFile" -ForegroundColor Green

Write-Host "Phi-3.5 conversion completed successfully!" -ForegroundColor Green
