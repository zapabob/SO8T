# SO8T×マルチモーダルLLM llama.cpp変換スクリプト
# llama.cpp-masterを使用してHugging FaceモデルをGGUFに変換

param(
    [string]$ModelPath = "./outputs",
    [string]$OutputDir = "./gguf_models",
    [string]$ModelName = "so8t-qwen2vl-2b",
    [string]$Quantization = "q8_0",
    [string]$LlamaCppPath = "C:\Users\downl\Desktop\SO8T\llama.cpp-master"
)

Write-Host "🔄 SO8T×マルチモーダルLLM llama.cpp変換開始..." -ForegroundColor Green

# 出力ディレクトリの作成
Write-Host "📁 出力ディレクトリを作成中..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# llama.cppディレクトリに移動
Write-Host "🔧 llama.cppディレクトリに移動中..." -ForegroundColor Yellow
Set-Location $LlamaCppPath

# モデル変換を実行
Write-Host "🎯 モデル変換を実行中..." -ForegroundColor Yellow
Write-Host "  入力モデル: $ModelPath" -ForegroundColor Cyan
Write-Host "  出力ディレクトリ: $OutputDir" -ForegroundColor Cyan
Write-Host "  量子化: $Quantization" -ForegroundColor Cyan

try {
    # convert_hf_to_gguf.pyを実行
    $convertCommand = @(
        "py", "convert_hf_to_gguf.py",
        $ModelPath,
        "--outfile", "$OutputDir\$ModelName.gguf",
        "--outtype", $Quantization,
        "--verbose"
    )
    
    Write-Host "🚀 変換コマンド実行: $($convertCommand -join ' ')" -ForegroundColor Green
    
    & $convertCommand[0] $convertCommand[1..($convertCommand.Length-1)]
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ モデル変換成功！" -ForegroundColor Green
        Write-Host "📁 出力ファイル: $OutputDir\$ModelName.gguf" -ForegroundColor Cyan
        
        # ファイルサイズを確認
        $outputFile = "$OutputDir\$ModelName.gguf"
        if (Test-Path $outputFile) {
            $fileSize = (Get-Item $outputFile).Length / 1GB
            Write-Host "📊 ファイルサイズ: $([math]::Round($fileSize, 2)) GB" -ForegroundColor Cyan
        }
        
        # Modelfileを作成
        Write-Host "📝 Modelfileを作成中..." -ForegroundColor Yellow
        $modelfileContent = @"
FROM $outputFile

TEMPLATE """{{ if .System }}}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}"""

# SO8T×マルチモーダルLLM Model Card
# SO(8)群回転ゲート + PET正則化 + OCR要約 + SQLite監査

SYSTEM """You are SO8T×マルチモーダルLLM, an advanced multimodal language model with SO(8) group structure and enhanced safety features.

Key Features:
- SO(8) Group Structure: 8-dimensional rotation gates for enhanced reasoning
- PET Regularization: Second-order difference penalty for smooth outputs
- OCR Summary: Local image processing with privacy protection
- SQLite Audit: Complete decision logging and policy tracking

Capabilities:
- Multimodal understanding (text + images)
- Safe and responsible AI responses
- Local OCR processing (no external data sharing)
- Comprehensive audit logging

Safety Guidelines:
- Always prioritize user safety and privacy
- Process images locally without external sharing
- Log all decisions for transparency
- Escalate complex ethical decisions when needed

You provide helpful, accurate, and safe responses while maintaining complete privacy and auditability."""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 32768
PARAMETER num_predict 2048
"@
        
        $modelfilePath = "$OutputDir\$ModelName.Modelfile"
        $modelfileContent | Out-File -FilePath $modelfilePath -Encoding UTF8
        
        Write-Host "✅ Modelfile作成完了: $modelfilePath" -ForegroundColor Green
        
        # 変換結果のサマリー
        Write-Host "`n📊 変換結果サマリー" -ForegroundColor Green
        Write-Host "=" * 50 -ForegroundColor Green
        Write-Host "モデル名: $ModelName" -ForegroundColor Cyan
        Write-Host "量子化: $Quantization" -ForegroundColor Cyan
        Write-Host "GGUFファイル: $outputFile" -ForegroundColor Cyan
        Write-Host "Modelfile: $modelfilePath" -ForegroundColor Cyan
        Write-Host "ファイルサイズ: $([math]::Round($fileSize, 2)) GB" -ForegroundColor Cyan
        
        # Ollamaモデル作成の指示
        Write-Host "`n🦙 Ollamaモデル作成手順:" -ForegroundColor Yellow
        Write-Host "1. ollama create $ModelName -f `"$modelfilePath`"" -ForegroundColor White
        Write-Host "2. ollama run $ModelName" -ForegroundColor White
        
    } else {
        Write-Error "❌ モデル変換に失敗しました (終了コード: $LASTEXITCODE)"
        exit 1
    }
    
} catch {
    Write-Error "❌ 変換中にエラーが発生しました: $($_.Exception.Message)"
    exit 1
} finally {
    # 元のディレクトリに戻る
    Set-Location "C:\Users\downl\Desktop\SO8T"
}

Write-Host "`n✅ llama.cpp変換完了！" -ForegroundColor Green
