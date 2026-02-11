# SO8T×マルチモーダルLLM A/Bテストスクリプト
# 4条件（回転有/無×PET有/無）の比較学習

param(
    [string]$ModelPath = "../Qwen2-VL-2B-Instruct",
    [string]$ConfigPath = "configs/train.qlora.json",
    [string]$OutputDir = "./ab_test_results",
    [int]$Epochs = 2,
    [int]$BatchSize = 1
)

Write-Host "🧪 SO8T×マルチモーダルLLM A/Bテスト開始..." -ForegroundColor Green

# 仮想環境のアクティベート
Write-Host "🔧 仮想環境をアクティベート中..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# 出力ディレクトリの作成
Write-Host "📁 出力ディレクトリを作成中..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# A/Bテストスクリプトの実行
Write-Host "🎯 A/Bテストを実行中..." -ForegroundColor Yellow

$abTestScript = @"
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# パスを追加
sys.path.append('src')

from training.trainer_with_pet import SO8TIntegratedTrainer
from modules.qwen2vl_wrapper import create_so8t_qwen2vl_model
from io.ocr_summary import OCRSummaryProcessor
from audit.sqlite_logger import SQLiteAuditLogger
from eval.metrics import SO8TEvaluator

def create_test_dataset():
    """テストデータセットを作成"""
    print("📊 テストデータセットを作成中...")
    
    # サンプルテキストデータ
    sample_texts = [
        "画像を説明してください。",
        "この写真には何が写っていますか？",
        "視覚的な内容を分析してください。",
        "画像の詳細を教えてください。",
        "この画像から何が分かりますか？",
        "写真の内容を要約してください。",
        "画像に写っている物体を特定してください。",
        "視覚的な情報を解釈してください。",
        "画像の特徴を説明してください。",
        "写真の分析結果を教えてください。"
    ]
    
    # トークナイザーを取得
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('$ModelPath')
    
    # データセットを作成
    dataset = []
    for text in sample_texts:
        # 入力とラベルを同じにする（自己回帰学習）
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        labels = inputs["input_ids"].clone()
        
        dataset.append({
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        })
    
    print(f"📈 データセットサイズ: {len(dataset)}")
    return dataset

def run_ab_test():
    """A/Bテストを実行"""
    print("🧪 SO8T×マルチモーダルLLM A/Bテスト開始...")
    
    # テストデータセットを作成
    test_dataset = create_test_dataset()
    
    # 4条件の設定
    test_conditions = [
        {
            "name": "baseline",
            "description": "ベースライン（回転なし、PETなし）",
            "rotation_enabled": False,
            "pet_enabled": False
        },
        {
            "name": "rotation_only",
            "description": "回転ゲートのみ",
            "rotation_enabled": True,
            "pet_enabled": False
        },
        {
            "name": "pet_only",
            "description": "PET損失のみ",
            "rotation_enabled": False,
            "pet_enabled": True
        },
        {
            "name": "full_so8t",
            "description": "完全SO8T（回転+PET）",
            "rotation_enabled": True,
            "pet_enabled": True
        }
    ]
    
    results = {}
    
    for condition in test_conditions:
        print(f"\\n🎯 条件: {condition['description']}")
        print("=" * 50)
        
        # 条件別の設定を作成
        condition_config = {
            "learning_rate": 2e-4,
            "batch_size": $BatchSize,
            "gradient_accumulation_steps": 8,
            "num_epochs": $Epochs,
            "warmup_steps": 50,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "lora_rank": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "rotation_gate_enabled": condition["rotation_enabled"],
            "pet_loss_enabled": condition["pet_enabled"],
            "pet_lambda_schedule": {
                "max_lambda": 0.1,
                "warmup_steps": 25,
                "main_steps": 100,
                "anneal_steps": 25
            }
        }
        
        # 設定ファイルを保存
        config_path = f"$OutputDir/config_{condition['name']}.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(condition_config, f, indent=2, ensure_ascii=False)
        
        # 出力ディレクトリ
        condition_output_dir = f"$OutputDir/{condition['name']}"
        os.makedirs(condition_output_dir, exist_ok=True)
        
        try:
            # 学習開始時間を記録
            start_time = time.time()
            
            # 学習器を初期化
            trainer = SO8TIntegratedTrainer(
                model_path='$ModelPath',
                config_path=config_path,
                output_dir=condition_output_dir
            )
            
            # 学習実行
            print(f"🚀 学習開始: {condition['description']}")
            trainer.train(test_dataset)
            
            # 学習時間を記録
            training_time = time.time() - start_time
            
            # 簡単な推論テスト
            print(f"🧪 推論テスト実行中...")
            test_prompts = [
                "画像を説明してください。",
                "この写真には何が写っていますか？",
                "視覚的な内容を分析してください。"
            ]
            
            inference_results = []
            for prompt in test_prompts:
                try:
                    response = trainer.generate_with_ocr(prompt)
                    inference_results.append({
                        "prompt": prompt,
                        "response": response,
                        "success": True
                    })
                except Exception as e:
                    inference_results.append({
                        "prompt": prompt,
                        "response": f"ERROR: {str(e)}",
                        "success": False
                    })
            
            # 結果を記録
            results[condition['name']] = {
                "condition": condition,
                "training_time": training_time,
                "inference_results": inference_results,
                "success_rate": np.mean([r["success"] for r in inference_results]),
                "output_dir": condition_output_dir,
                "config_path": config_path
            }
            
            print(f"✅ 条件完了: {condition['description']}")
            print(f"   学習時間: {training_time:.2f}秒")
            print(f"   成功率: {results[condition['name']]['success_rate']:.3f}")
            
        except Exception as e:
            print(f"❌ 条件失敗: {condition['description']} - {str(e)}")
            results[condition['name']] = {
                "condition": condition,
                "error": str(e),
                "success": False
            }
    
    return results

def analyze_results(results):
    """結果を分析"""
    print("\\n📊 A/Bテスト結果分析")
    print("=" * 60)
    
    # 成功した条件のみを分析
    successful_results = {k: v for k, v in results.items() if v.get("success", True) and "error" not in v}
    
    if not successful_results:
        print("❌ 成功した条件がありません")
        return
    
    # 学習時間の比較
    print("\\n⏱️ 学習時間比較:")
    for name, result in successful_results.items():
        print(f"  {result['condition']['description']}: {result['training_time']:.2f}秒")
    
    # 成功率の比較
    print("\\n📈 成功率比較:")
    for name, result in successful_results.items():
        print(f"  {result['condition']['description']}: {result['success_rate']:.3f}")
    
    # 最良の条件を特定
    best_condition = max(successful_results.items(), key=lambda x: x[1]['success_rate'])
    print(f"\\n🏆 最良の条件: {best_condition[1]['condition']['description']}")
    print(f"   成功率: {best_condition[1]['success_rate']:.3f}")
    print(f"   学習時間: {best_condition[1]['training_time']:.2f}秒")
    
    # 詳細な比較分析
    print("\\n🔍 詳細分析:")
    
    # 回転ゲートの効果
    rotation_conditions = [r for r in successful_results.values() if r['condition']['rotation_enabled']]
    no_rotation_conditions = [r for r in successful_results.values() if not r['condition']['rotation_enabled']]
    
    if rotation_conditions and no_rotation_conditions:
        rotation_avg = np.mean([r['success_rate'] for r in rotation_conditions])
        no_rotation_avg = np.mean([r['success_rate'] for r in no_rotation_conditions])
        print(f"  回転ゲート効果: {rotation_avg:.3f} vs {no_rotation_avg:.3f} (差: {rotation_avg - no_rotation_avg:+.3f})")
    
    # PET損失の効果
    pet_conditions = [r for r in successful_results.values() if r['condition']['pet_enabled']]
    no_pet_conditions = [r for r in successful_results.values() if not r['condition']['pet_enabled']]
    
    if pet_conditions and no_pet_conditions:
        pet_avg = np.mean([r['success_rate'] for r in pet_conditions])
        no_pet_avg = np.mean([r['success_rate'] for r in no_pet_conditions])
        print(f"  PET損失効果: {pet_avg:.3f} vs {no_pet_avg:.3f} (差: {pet_avg - no_pet_avg:+.3f})")
    
    return successful_results

def main():
    print("🧪 SO8T×マルチモーダルLLM A/Bテスト開始...")
    
    # デバイス情報を表示
    print(f"🔧 デバイス: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"💾 メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "CPU使用")
    
    # A/Bテストを実行
    results = run_ab_test()
    
    # 結果を分析
    successful_results = analyze_results(results)
    
    # 結果をJSONファイルに保存
    results_file = "$OutputDir/ab_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_conditions": len(results),
            "successful_conditions": len(successful_results) if successful_results else 0,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\\n📁 結果を保存しました: {results_file}")
    print("✅ A/Bテスト完了！")

if __name__ == "__main__":
    main()
"@

# A/Bテストスクリプトを実行
$abTestScript | py -3

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ A/Bテスト完了！" -ForegroundColor Green
    Write-Host "📁 結果ディレクトリ: $OutputDir" -ForegroundColor Cyan
    Write-Host "📊 結果ファイル: $OutputDir/ab_test_results.json" -ForegroundColor Cyan
} else {
    Write-Error "❌ A/Bテスト中にエラーが発生しました"
    exit 1
}
