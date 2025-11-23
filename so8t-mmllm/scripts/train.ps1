# SO8T×マルチモーダルLLM 学習スクリプト
# RTX3060 12GB環境用

param(
    [string]$ConfigPath = "configs/train.qlora.json",
    [string]$ModelPath = "../Qwen2-VL-2B-Instruct",
    [string]$OutputDir = "./outputs",
    [switch]$EnableRotation = $true,
    [switch]$EnablePET = $true,
    [int]$BatchSize = 1,
    [int]$Epochs = 3
)

Write-Host "🚀 SO8T×マルチモーダルLLM 学習開始..." -ForegroundColor Green

# 仮想環境のアクティベート
Write-Host "🔧 仮想環境をアクティベート中..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# 設定ファイルの存在確認
if (-not (Test-Path $ConfigPath)) {
    Write-Error "設定ファイルが見つかりません: $ConfigPath"
    exit 1
}

# モデルパスの存在確認
if (-not (Test-Path $ModelPath)) {
    Write-Error "モデルパスが見つかりません: $ModelPath"
    exit 1
}

# 出力ディレクトリの作成
Write-Host "📁 出力ディレクトリを作成中..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# 学習スクリプトの実行
Write-Host "🎯 学習を開始中..." -ForegroundColor Yellow

$trainScript = @"
import sys
import os
import json
import torch
from pathlib import Path

# パスを追加
sys.path.append('src')

from training.qlora import SO8TQLoRATrainer
from modules.qwen2vl_wrapper import create_so8t_qwen2vl_model
from io.ocr_summary import OCRSummaryProcessor
from audit.sqlite_logger import SQLiteAuditLogger

def main():
    print("🚀 SO8T×マルチモーダルLLM 学習開始...")
    
    # 設定を読み込み
    with open('$ConfigPath', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # デバイス情報を表示
    print(f"🔧 デバイス: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"💾 メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "CPU使用")
    
    # 学習器を初期化
    trainer = SO8TQLoRATrainer(
        model_path='$ModelPath',
        config_path='$ConfigPath',
        output_dir='$OutputDir'
    )
    
    # サンプルデータセットを作成（実際の学習では適切なデータセットを使用）
    print("📊 サンプルデータセットを作成中...")
    
    # 簡単なテキスト生成タスクのサンプルデータ
    sample_texts = [
        "画像を説明してください。",
        "この写真には何が写っていますか？",
        "視覚的な内容を分析してください。",
        "画像の詳細を教えてください。",
        "この画像から何が分かりますか？"
    ]
    
    # トークナイザーを取得
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('$ModelPath')
    
    # データセットを作成
    train_dataset = []
    for text in sample_texts:
        # 入力とラベルを同じにする（自己回帰学習）
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        labels = inputs["input_ids"].clone()
        
        train_dataset.append({
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        })
    
    print(f"📈 データセットサイズ: {len(train_dataset)}")
    
    # OCR要約プロセッサを初期化
    print("🔍 OCR要約プロセッサを初期化中...")
    ocr_processor = OCRSummaryProcessor()
    
    # SQLite監査ロガーを初期化
    print("🗄️ SQLite監査ロガーを初期化中...")
    audit_logger = SQLiteAuditLogger(db_path="$OutputDir/audit.db")
    
    # 学習開始
    print("🎯 学習を開始中...")
    try:
        trainer.train(train_dataset)
        print("✅ 学習完了！")
        
        # 監査ログに学習完了を記録
        audit_logger.log_audit(
            change_type="training_complete",
            change_description="SO8T×マルチモーダルLLM学習完了",
            change_data={
                "config_path": "$ConfigPath",
                "model_path": "$ModelPath",
                "output_dir": "$OutputDir",
                "dataset_size": len(train_dataset),
                "rotation_enabled": $EnableRotation,
                "pet_enabled": $EnablePET
            }
        )
        
        # 簡単な推論テスト
        print("🧪 推論テストを実行中...")
        test_result = trainer.generate("画像を説明してください。")
        print(f"📝 生成結果: {test_result}")
        
    except Exception as e:
        print(f"❌ 学習中にエラーが発生しました: {str(e)}")
        
        # エラーを監査ログに記録
        audit_logger.log_audit(
            change_type="training_error",
            change_description="学習中にエラーが発生",
            change_data={"error": str(e)}
        )
        
        raise e

if __name__ == "__main__":
    main()
"@

# 学習スクリプトを実行
$trainScript | py -3

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 学習完了！" -ForegroundColor Green
    Write-Host "📁 出力ディレクトリ: $OutputDir" -ForegroundColor Cyan
    Write-Host "🗄️ 監査データベース: $OutputDir/audit.db" -ForegroundColor Cyan
} else {
    Write-Error "❌ 学習中にエラーが発生しました"
    exit 1
}
