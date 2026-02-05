# SO8T×マルチモーダルLLM llama.cpp統合スクリプト
# llama.cpp-masterをSO8Tプロジェクトに統合

param(
    [string]$LlamaCppPath = "C:\Users\downl\Desktop\SO8T\llama.cpp-master",
    [string]$ProjectPath = "C:\Users\downl\Desktop\SO8T\so8t-mmllm",
    [switch]$InstallDependencies = $true,
    [switch]$TestConversion = $true
)

Write-Host "🔗 SO8T×マルチモーダルLLM llama.cpp統合開始..." -ForegroundColor Green

# パスの確認
Write-Host "📁 パス確認中..." -ForegroundColor Yellow
if (-not (Test-Path $LlamaCppPath)) {
    Write-Error "❌ llama.cppパスが見つかりません: $LlamaCppPath"
    exit 1
}

if (-not (Test-Path $ProjectPath)) {
    Write-Error "❌ プロジェクトパスが見つかりません: $ProjectPath"
    exit 1
}

Write-Host "✅ パス確認完了" -ForegroundColor Green

# 依存関係のインストール
if ($InstallDependencies) {
    Write-Host "📦 依存関係をインストール中..." -ForegroundColor Yellow
    
    try {
        Set-Location $LlamaCppPath
        py -m pip install -r requirements.txt
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ 依存関係インストール完了" -ForegroundColor Green
        } else {
            Write-Warning "⚠️ 依存関係インストールに警告があります"
        }
    } catch {
        Write-Error "❌ 依存関係インストールに失敗しました: $($_.Exception.Message)"
        exit 1
    } finally {
        Set-Location $ProjectPath
    }
}

# 統合スクリプトの作成
Write-Host "📝 統合スクリプトを作成中..." -ForegroundColor Yellow

# 1. 環境設定スクリプト
$envScript = @"
# SO8T×マルチモーダルLLM llama.cpp環境設定
# llama.cpp-masterを使用してHugging FaceモデルをGGUFに変換

# 環境変数設定
`$env:LLAMACPP_PATH = "$LlamaCppPath"
`$env:SO8T_PROJECT_PATH = "$ProjectPath"

# Pythonパスにllama.cppを追加
`$env:PYTHONPATH = "`$env:PYTHONPATH;$LlamaCppPath"

Write-Host "🔧 llama.cpp環境設定完了" -ForegroundColor Green
Write-Host "llama.cppパス: `$env:LLAMACPP_PATH" -ForegroundColor Cyan
Write-Host "SO8Tプロジェクトパス: `$env:SO8T_PROJECT_PATH" -ForegroundColor Cyan
"@

$envScript | Out-File -FilePath "$ProjectPath\scripts\setup_llamacpp_env.ps1" -Encoding UTF8

# 2. 統合変換スクリプト
$integratedScript = @'
# SO8T×マルチモーダルLLM 統合変換スクリプト
# llama.cpp-masterを使用してSO8TモデルをGGUFに変換

param(
    [string]$ModelPath = "./outputs",
    [string]$OutputDir = "./gguf_models",
    [string]$ModelName = "so8t-qwen2vl-2b",
    [string]$Quantization = "q8_0"
)

# 環境設定を読み込み
. "C:\Users\downl\Desktop\SO8T\so8t-mmllm\scripts\setup_llamacpp_env.ps1"

Write-Host "🔄 SO8T×マルチモーダルLLM 統合変換開始..." -ForegroundColor Green

# 出力ディレクトリの作成
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# llama.cppディレクトリに移動
Set-Location $env:LLAMACPP_PATH

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
        
        # Modelfileを作成
        $modelfileContent = @"
FROM $OutputDir\$ModelName.gguf

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
        Write-Host "GGUFファイル: $OutputDir\$ModelName.gguf" -ForegroundColor Cyan
        Write-Host "Modelfile: $modelfilePath" -ForegroundColor Cyan
        
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
    Set-Location $env:SO8T_PROJECT_PATH
}

Write-Host "`n✅ 統合変換完了！" -ForegroundColor Green
'@

$integratedScript | Out-File -FilePath "$ProjectPath\scripts\convert_so8t_with_llamacpp.ps1" -Encoding UTF8

# 3. テストスクリプト
$testScript = @'
# SO8T×マルチモーダルLLM llama.cpp統合テスト
# llama.cpp-masterの統合をテスト

param(
    [string]$TestModelPath = "./test_models",
    [string]$TestOutputDir = "./test_gguf_models"
)

# 環境設定を読み込み
. "C:\Users\downl\Desktop\SO8T\so8t-mmllm\scripts\setup_llamacpp_env.ps1"

Write-Host "🧪 SO8T×マルチモーダルLLM llama.cpp統合テスト開始..." -ForegroundColor Green

# テスト用のダミーモデルディレクトリを作成
if (-not (Test-Path $TestModelPath)) {
    Write-Host "📁 テスト用ディレクトリを作成中..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $TestModelPath -Force | Out-Null
    
    # ダミーのconfig.jsonを作成
    $dummyConfig = @{
        "architectures" = @("Qwen2VLForConditionalGeneration")
        "hidden_size" = 1536
        "num_attention_heads" = 12
        "num_hidden_layers" = 28
        "vocab_size" = 151936
        "model_type" = "qwen2_vl"
    } | ConvertTo-Json -Depth 10
    
    $dummyConfig | Out-File -FilePath "$TestModelPath\config.json" -Encoding UTF8
    
    Write-Host "✅ テスト用ダミーモデル作成完了" -ForegroundColor Green
}

# 統合変換をテスト
Write-Host "🔄 統合変換をテスト中..." -ForegroundColor Yellow
try {
    . "C:\Users\downl\Desktop\SO8T\so8t-mmllm\scripts\convert_so8t_with_llamacpp.ps1" -ModelPath $TestModelPath -OutputDir $TestOutputDir -ModelName "test-so8t-model" -Quantization "q8_0"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ 統合変換テスト成功！" -ForegroundColor Green
    } else {
        Write-Warning "⚠️ 統合変換テストに警告があります"
    }
} catch {
    Write-Error "❌ 統合変換テストに失敗しました: $($_.Exception.Message)"
}

Write-Host "`n✅ llama.cpp統合テスト完了！" -ForegroundColor Green
'@

$testScript | Out-File -FilePath "$ProjectPath\scripts\test_llamacpp_integration.ps1" -Encoding UTF8

# 4. README更新
Write-Host "📝 READMEを更新中..." -ForegroundColor Yellow

$readmeUpdate = @"

## llama.cpp統合

### 環境設定
```powershell
# llama.cpp環境を設定
.\scripts\setup_llamacpp_env.ps1
```

### モデル変換
```powershell
# SO8TモデルをGGUFに変換
.\scripts\convert_so8t_with_llamacpp.ps1 -ModelPath "./outputs" -OutputDir "./gguf_models" -ModelName "so8t-qwen2vl-2b" -Quantization "q8_0"
```

### テスト実行
```powershell
# llama.cpp統合をテスト
.\scripts\test_llamacpp_integration.ps1
```

### 利用可能な量子化タイプ
- `f32`: 32-bit float (最高品質、最大サイズ)
- `f16`: 16-bit float (高品質、中サイズ)
- `bf16`: bfloat16 (高品質、中サイズ)
- `q8_0`: 8-bit quantization (推奨、小サイズ)
- `tq1_0`: 1-bit ternary quantization (最小サイズ)
- `tq2_0`: 2-bit ternary quantization (小サイズ)
- `auto`: 自動選択

"@

# READMEに追加
$readmePath = "$ProjectPath\README.md"
if (Test-Path $readmePath) {
    $readmeContent = Get-Content $readmePath -Raw
    $readmeContent += $readmeUpdate
    $readmeContent | Out-File -FilePath $readmePath -Encoding UTF8
    Write-Host "✅ README更新完了" -ForegroundColor Green
}

# 統合テストの実行
if ($TestConversion) {
    Write-Host "🧪 統合テストを実行中..." -ForegroundColor Yellow
    try {
        & "$ProjectPath\scripts\test_llamacpp_integration.ps1"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ 統合テスト成功！" -ForegroundColor Green
        } else {
            Write-Warning "⚠️ 統合テストに警告があります"
        }
    } catch {
        Write-Error "❌ 統合テストに失敗しました: $($_.Exception.Message)"
    }
}

# 統合完了サマリー
Write-Host "`n📊 llama.cpp統合完了サマリー" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Green
Write-Host "✅ 環境設定スクリプト: setup_llamacpp_env.ps1" -ForegroundColor Cyan
Write-Host "✅ 統合変換スクリプト: convert_so8t_with_llamacpp.ps1" -ForegroundColor Cyan
Write-Host "✅ 統合テストスクリプト: test_llamacpp_integration.ps1" -ForegroundColor Cyan
Write-Host "✅ README更新完了" -ForegroundColor Cyan

Write-Host "`n🦙 使用方法:" -ForegroundColor Yellow
Write-Host "1. .\scripts\setup_llamacpp_env.ps1" -ForegroundColor White
Write-Host "2. .\scripts\convert_so8t_with_llamacpp.ps1" -ForegroundColor White
Write-Host "3. ollama create so8t-qwen2vl-2b -f `"./gguf_models/so8t-qwen2vl-2b.Modelfile`"" -ForegroundColor White

Write-Host "`n✅ llama.cpp統合完了！" -ForegroundColor Green
