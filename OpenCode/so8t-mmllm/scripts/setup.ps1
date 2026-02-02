# SO8T×マルチモーダルLLM（ローカル）セットアップスクリプト
# RTX3060 12GB環境用

Write-Host "🚀 SO8T×マルチモーダルLLM セットアップ開始..." -ForegroundColor Green

# 仮想環境の作成とアクティベート
Write-Host "📦 仮想環境を作成中..." -ForegroundColor Yellow
py -3 -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Error "仮想環境の作成に失敗しました"
    exit 1
}

Write-Host "🔧 仮想環境をアクティベート中..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Error "仮想環境のアクティベートに失敗しました"
    exit 1
}

# pipのアップグレード
Write-Host "⬆️ pipをアップグレード中..." -ForegroundColor Yellow
py -3 -m pip install --upgrade pip

# PyTorchのインストール（CUDA 12.1対応）
Write-Host "🔥 PyTorch (CUDA 12.1) をインストール中..." -ForegroundColor Yellow
py -3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# その他の依存関係のインストール
Write-Host "📚 依存関係をインストール中..." -ForegroundColor Yellow
py -3 -m pip install -r requirements.txt

# Qwen2-VL-2B-Instructのダウンロード確認
Write-Host "🔍 Qwen2-VL-2B-Instructの存在確認..." -ForegroundColor Yellow
$qwenPath = "..\Qwen2-VL-2B-Instruct"
if (Test-Path $qwenPath) {
    Write-Host "✅ Qwen2-VL-2B-Instructが見つかりました" -ForegroundColor Green
} else {
    Write-Warning "⚠️ Qwen2-VL-2B-Instructが見つかりません。手動でダウンロードしてください。"
}

# 設定ファイルの作成
Write-Host "⚙️ 設定ファイルを作成中..." -ForegroundColor Yellow

# モデル設定
$modelConfig = @{
    "model_name" = "Qwen2-VL-2B-Instruct"
    "model_path" = "..\Qwen2-VL-2B-Instruct"
    "hidden_size" = 1536
    "num_attention_heads" = 12
    "num_hidden_layers" = 28
    "intermediate_size" = 8960
    "vocab_size" = 151936
    "max_position_embeddings" = 32768
    "torch_dtype" = "bfloat16"
    "device_map" = "auto"
} | ConvertTo-Json -Depth 3

$modelConfig | Out-File -FilePath "configs\model.qwen2vl-2b.json" -Encoding UTF8

# 学習設定
$trainConfig = @{
    "learning_rate" = 2e-4
    "batch_size" = 1
    "gradient_accumulation_steps" = 8
    "num_epochs" = 3
    "warmup_steps" = 100
    "max_grad_norm" = 1.0
    "weight_decay" = 0.01
    "lora_rank" = 64
    "lora_alpha" = 128
    "lora_dropout" = 0.1
    "target_modules" = @("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    "rotation_gate_enabled" = $true
    "pet_loss_enabled" = $true
    "pet_lambda_schedule" = @{
        "warmup_steps" = 100
        "main_steps" = 1000
        "anneal_steps" = 200
        "max_lambda" = 0.1
    }
} | ConvertTo-Json -Depth 3

$trainConfig | Out-File -FilePath "configs\train.qlora.json" -Encoding UTF8

# SQLiteスキーマの作成
Write-Host "🗄️ SQLiteスキーマを作成中..." -ForegroundColor Yellow
$sqlSchema = @"
-- SO8T×マルチモーダルLLM 監査データベーススキーマ
-- WALモード + synchronous=FULL で耐久性を重視

PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA cache_size=10000;
PRAGMA temp_store=MEMORY;

-- 判断ログテーブル
CREATE TABLE IF NOT EXISTS decision_log(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
    input_hash TEXT NOT NULL,
    decision TEXT CHECK(decision IN ('ALLOW','ESCALATE','DENY')) NOT NULL,
    confidence REAL NOT NULL,
    reasoning TEXT,
    meta JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ポリシー状態テーブル
CREATE TABLE IF NOT EXISTS policy_state(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
    policy_name TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    policy_content JSON NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- アイデンティティ契約テーブル
CREATE TABLE IF NOT EXISTS identity_contract(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
    contract_name TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    contract_content JSON NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 監査ログテーブル
CREATE TABLE IF NOT EXISTS audit_log(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
    change_type TEXT NOT NULL,
    change_description TEXT NOT NULL,
    change_data JSON,
    user_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- インデックスの作成
CREATE INDEX IF NOT EXISTS idx_decision_log_ts ON decision_log(ts);
CREATE INDEX IF NOT EXISTS idx_decision_log_hash ON decision_log(input_hash);
CREATE INDEX IF NOT EXISTS idx_policy_state_active ON policy_state(is_active);
CREATE INDEX IF NOT EXISTS idx_identity_contract_active ON identity_contract(is_active);
CREATE INDEX IF NOT EXISTS idx_audit_log_ts ON audit_log(ts);
"@

$sqlSchema | Out-File -FilePath "sql\schema.sql" -Encoding UTF8

# 初期データの挿入
$initData = @"
-- 初期ポリシー状態
INSERT OR IGNORE INTO policy_state (policy_name, policy_version, policy_content) VALUES 
('safety_policy', '1.0', '{"harmful_content": "DENY", "sensitive_info": "ESCALATE", "general": "ALLOW"}'),
('privacy_policy', '1.0', '{"image_processing": "LOCAL_ONLY", "data_retention": "7_DAYS", "external_sharing": "FORBIDDEN"}');

-- 初期アイデンティティ契約
INSERT OR IGNORE INTO identity_contract (contract_name, contract_version, contract_content) VALUES 
('ai_assistant_contract', '1.0', '{"role": "helpful_assistant", "capabilities": ["text_generation", "image_analysis", "reasoning"], "limitations": ["no_harmful_content", "privacy_respect", "factual_accuracy"]}');

-- 初期監査ログ
INSERT OR IGNORE INTO audit_log (change_type, change_description, change_data) VALUES 
('system_init', 'SO8T×マルチモーダルLLM初期化', '{"version": "1.0", "features": ["rotation_gate", "pet_loss", "ocr_summary", "sqlite_audit"]}');
"@

$initData | Out-File -FilePath "sql\init_data.sql" -Encoding UTF8

Write-Host "✅ セットアップ完了！" -ForegroundColor Green
Write-Host "📁 プロジェクト構造:" -ForegroundColor Cyan
Write-Host "  so8t-mmllm/" -ForegroundColor White
Write-Host "  ├── src/                    # ソースコード" -ForegroundColor White
Write-Host "  ├── configs/               # 設定ファイル" -ForegroundColor White
Write-Host "  ├── sql/                   # SQLiteスキーマ" -ForegroundColor White
Write-Host "  ├── scripts/               # 実行スクリプト" -ForegroundColor White
Write-Host "  ├── eval/                  # 評価スクリプト" -ForegroundColor White
Write-Host "  └── requirements.txt       # 依存関係" -ForegroundColor White

Write-Host "🚀 次のステップ:" -ForegroundColor Yellow
Write-Host "  1. 仮想環境をアクティベート: .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. 学習開始: .\scripts\train.ps1" -ForegroundColor White
Write-Host "  3. 評価実行: .\scripts\eval.ps1" -ForegroundColor White
