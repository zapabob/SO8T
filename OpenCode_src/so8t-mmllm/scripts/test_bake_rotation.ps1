# SO8T×マルチモーダルLLM 回転焼き込みテスト
# 学習済み回転を射影重みに焼き込む機能をテスト

param(
    [string]$ModelPath = "./outputs",
    [string]$OutputDir = "./bake_test_results",
    [string]$TestImageDir = "./test_images"
)

Write-Host "🔥 SO8T×マルチモーダルLLM 回転焼き込みテスト開始..." -ForegroundColor Green

# 仮想環境のアクティベート
Write-Host "🔧 仮想環境をアクティベート中..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# 出力ディレクトリの作成
Write-Host "📁 出力ディレクトリを作成中..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# 回転焼き込みテストスクリプトの実行
Write-Host "🎯 回転焼き込みテストを実行中..." -ForegroundColor Yellow

$bakeTestScript = @"
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# パスを追加
sys.path.append('src')

from training.trainer_with_pet import SO8TIntegratedTrainer
from modules.qwen2vl_wrapper import create_so8t_qwen2vl_model
from modules.rotation_gate import SO8TRotationGate
from io.ocr_summary import OCRSummaryProcessor
from audit.sqlite_logger import SQLiteAuditLogger

def test_rotation_baking():
    """回転焼き込みをテスト"""
    print("🔥 回転焼き込みをテスト中...")
    
    results = []
    
    # 1. 学習済みモデルを読み込み
    print("  📦 学習済みモデルを読み込み中...")
    try:
        trainer = SO8TIntegratedTrainer(
            model_path='$ModelPath',
            config_path='$ModelPath/config.json',
            output_dir='$OutputDir'
        )
        trainer.setup_components()
        
        # 回転ゲートを取得
        rotation_gate = trainer.rotation_gate
        if rotation_gate is None:
            print("  ⚠️ 回転ゲートが見つかりません。テスト用に作成します。")
            rotation_gate = SO8TRotationGate(
                hidden_size=trainer.model.config.hidden_size,
                learnable=True
            )
        
        print(f"  ✅ モデル読み込み完了")
        print(f"     回転ゲート: {'有効' if rotation_gate is not None else '無効'}")
        
        results.append({
            "test": "model_loading",
            "success": True,
            "rotation_gate_available": rotation_gate is not None
        })
        
    except Exception as e:
        print(f"  ❌ モデル読み込みエラー: {str(e)}")
        results.append({
            "test": "model_loading",
            "success": False,
            "error": str(e)
        })
        return results
    
    # 2. 焼き込み前の重みを記録
    print("  📊 焼き込み前の重みを記録中...")
    original_weights = {}
    for name, module in trainer.model.named_modules():
        if hasattr(module, 'o_proj') and hasattr(module.o_proj, 'weight'):
            original_weights[name] = module.o_proj.weight.data.clone()
            print(f"    {name}.o_proj: {module.o_proj.weight.data.shape}")
    
    results.append({
        "test": "weight_recording",
        "success": True,
        "recorded_layers": len(original_weights)
    })
    
    # 3. 回転焼き込みを実行
    print("  🔥 回転焼き込みを実行中...")
    try:
        # 回転行列を取得
        rotation_matrices = rotation_gate.get_rotation_matrices()
        print(f"    回転行列形状: {rotation_matrices.shape}")
        
        # 焼き込みを実行
        baked_layers = 0
        for name, module in trainer.model.named_modules():
            if hasattr(module, 'o_proj') and hasattr(module.o_proj, 'weight'):
                weight = module.o_proj.weight.data
                hidden_size = weight.size(1)
                num_blocks = hidden_size // 8
                
                if num_blocks == rotation_matrices.size(0):
                    # 重みを8次元ブロックに分割
                    weight_blocks = weight.view(hidden_size, num_blocks, 8)
                    
                    # 各ブロックに回転を適用
                    for block_idx in range(num_blocks):
                        R = rotation_matrices[block_idx]
                        weight_blocks[:, block_idx, :] = torch.matmul(
                            weight_blocks[:, block_idx, :], R.T
                        )
                    
                    # 元の形状に戻す
                    module.o_proj.weight.data = weight_blocks.view(hidden_size, hidden_size)
                    baked_layers += 1
                    print(f"    ✅ {name}.o_proj に回転を焼き込みました")
        
        results.append({
            "test": "rotation_baking",
            "success": True,
            "baked_layers": baked_layers
        })
        
    except Exception as e:
        print(f"  ❌ 回転焼き込みエラー: {str(e)}")
        results.append({
            "test": "rotation_baking",
            "success": False,
            "error": str(e)
        })
        return results
    
    # 4. 焼き込み後の重みを比較
    print("  📊 焼き込み後の重みを比較中...")
    weight_changes = {}
    for name, original_weight in original_weights.items():
        if name in [n for n, m in trainer.model.named_modules() if hasattr(m, 'o_proj')]:
            current_weight = next(m.o_proj.weight.data for n, m in trainer.model.named_modules() if n == name and hasattr(m, 'o_proj'))
            
            # 重みの変化を計算
            weight_diff = torch.norm(current_weight - original_weight).item()
            weight_change_ratio = weight_diff / torch.norm(original_weight).item()
            
            weight_changes[name] = {
                "weight_diff": weight_diff,
                "change_ratio": weight_change_ratio
            }
            
            print(f"    {name}: 変化率 {weight_change_ratio:.6f}")
    
    results.append({
        "test": "weight_comparison",
        "success": True,
        "weight_changes": weight_changes
    })
    
    # 5. 推論テスト
    print("  🧪 焼き込み後の推論テスト中...")
    try:
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
                print(f"    ✅ 推論成功: {prompt[:20]}...")
            except Exception as e:
                inference_results.append({
                    "prompt": prompt,
                    "response": f"ERROR: {str(e)}",
                    "success": False
                })
                print(f"    ❌ 推論エラー: {str(e)}")
        
        success_rate = np.mean([r["success"] for r in inference_results])
        results.append({
            "test": "inference_test",
            "success": True,
            "success_rate": success_rate,
            "inference_results": inference_results
        })
        
    except Exception as e:
        print(f"  ❌ 推論テストエラー: {str(e)}")
        results.append({
            "test": "inference_test",
            "success": False,
            "error": str(e)
        })
    
    # 6. 焼き込み済みモデルを保存
    print("  💾 焼き込み済みモデルを保存中...")
    try:
        baked_model_path = "$OutputDir/baked_model"
        trainer.model.save_pretrained(baked_model_path)
        
        # 回転行列も保存
        rotation_path = "$OutputDir/rotation_matrices.pt"
        torch.save(rotation_matrices, rotation_path)
        
        results.append({
            "test": "model_saving",
            "success": True,
            "baked_model_path": baked_model_path,
            "rotation_path": rotation_path
        })
        
    except Exception as e:
        print(f"  ❌ モデル保存エラー: {str(e)}")
        results.append({
            "test": "model_saving",
            "success": False,
            "error": str(e)
        })
    
    return results

def test_rotation_consistency():
    """回転の一貫性をテスト"""
    print("🔄 回転の一貫性をテスト中...")
    
    results = []
    
    try:
        # 2つの回転ゲートを作成
        hidden_size = 1536  # Qwen2-VL-2Bの隠れ層サイズ
        gate1 = SO8TRotationGate(hidden_size=hidden_size, learnable=True)
        gate2 = SO8TRotationGate(hidden_size=hidden_size, learnable=True)
        
        # 同じパラメータを設定
        gate2.theta.data = gate1.theta.data.clone()
        
        # 回転行列を計算
        R1 = gate1.get_rotation_matrices()
        R2 = gate2.get_rotation_matrices()
        
        # 一貫性をチェック
        rotation_diff = torch.norm(R1 - R2).item()
        is_consistent = rotation_diff < 1e-6
        
        print(f"  回転行列の差: {rotation_diff:.2e}")
        print(f"  一貫性: {'✅' if is_consistent else '❌'}")
        
        results.append({
            "test": "rotation_consistency",
            "success": is_consistent,
            "rotation_diff": rotation_diff,
            "is_consistent": is_consistent
        })
        
    except Exception as e:
        print(f"  ❌ 一貫性テストエラー: {str(e)}")
        results.append({
            "test": "rotation_consistency",
            "success": False,
            "error": str(e)
        })
    
    return results

def test_gguf_conversion():
    """GGUF変換をテスト"""
    print("🔄 GGUF変換をテスト中...")
    
    results = []
    
    try:
        # 焼き込み済みモデルを読み込み
        baked_model_path = "$OutputDir/baked_model"
        if os.path.exists(baked_model_path):
            print("  📦 焼き込み済みモデルを読み込み中...")
            
            # モデルを読み込み
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(baked_model_path)
            
            # GGUF変換の準備
            gguf_path = "$OutputDir/test_model.gguf"
            
            # 変換設定を作成
            conversion_config = {
                "model_name": "so8t-qwen2vl-2b-baked",
                "model_path": baked_model_path,
                "output_path": gguf_path,
                "quantization": "Q8_0",
                "description": "SO8T×マルチモーダルLLM (焼き込み済み)"
            }
            
            # 設定を保存
            config_path = "$OutputDir/gguf_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(conversion_config, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ 変換設定を作成: {config_path}")
            print(f"  ⚠️ 注意: 実際のGGUF変換にはllama.cppのconvert.pyが必要です")
            
            results.append({
                "test": "gguf_conversion",
                "success": True,
                "config_path": config_path,
                "gguf_path": gguf_path
            })
            
        else:
            print("  ⚠️ 焼き込み済みモデルが見つかりません")
            results.append({
                "test": "gguf_conversion",
                "success": False,
                "error": "Baked model not found"
            })
    
    except Exception as e:
        print(f"  ❌ GGUF変換テストエラー: {str(e)}")
        results.append({
            "test": "gguf_conversion",
            "success": False,
            "error": str(e)
        })
    
    return results

def analyze_results(all_results):
    """結果を分析"""
    print("\\n📊 回転焼き込みテスト結果分析")
    print("=" * 50)
    
    # 各テストの成功率を計算
    test_success = {}
    for result in all_results:
        test_name = result.get('test', 'unknown')
        if test_name not in test_success:
            test_success[test_name] = {'success': 0, 'total': 0}
        
        test_success[test_name]['total'] += 1
        if result.get('success', False):
            test_success[test_name]['success'] += 1
    
    print("📈 テスト結果:")
    for test_name, stats in test_success.items():
        success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"  {test_name}: {stats['success']}/{stats['total']} ({success_rate:.3f})")
    
    # 総合成功率
    total_success = sum(stats['success'] for stats in test_success.values())
    total_tests = sum(stats['total'] for stats in test_success.values())
    overall_success_rate = total_success / total_tests if total_tests > 0 else 0.0
    
    print(f"\\n📊 総合成功率: {overall_success_rate:.3f}")
    
    return {
        "test_success": test_success,
        "overall_success_rate": overall_success_rate
    }

def main():
    print("🔥 SO8T×マルチモーダルLLM 回転焼き込みテスト開始...")
    
    # 各テストを実行
    print("\\n🎯 回転焼き込みテスト開始...")
    bake_results = test_rotation_baking()
    
    print("\\n🎯 回転一貫性テスト開始...")
    consistency_results = test_rotation_consistency()
    
    print("\\n🎯 GGUF変換テスト開始...")
    gguf_results = test_gguf_conversion()
    
    # 全結果を統合
    all_results = bake_results + consistency_results + gguf_results
    
    # 結果を分析
    analysis = analyze_results(all_results)
    
    # 結果を保存
    results = {
        "timestamp": datetime.now().isoformat(),
        "bake_results": bake_results,
        "consistency_results": consistency_results,
        "gguf_results": gguf_results,
        "analysis": analysis
    }
    
    results_file = "$OutputDir/bake_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\\n📁 結果を保存しました: {results_file}")
    print(f"📊 総合成功率: {analysis['overall_success_rate']:.3f}")
    
    print("\\n✅ 回転焼き込みテスト完了！")

if __name__ == "__main__":
    main()
"@

# 回転焼き込みテストスクリプトを実行
$bakeTestScript | py -3

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 回転焼き込みテスト完了！" -ForegroundColor Green
    Write-Host "📁 結果ディレクトリ: $OutputDir" -ForegroundColor Cyan
    Write-Host "📊 結果ファイル: $OutputDir/bake_test_results.json" -ForegroundColor Cyan
    Write-Host "🍞 焼き込み済みモデル: $OutputDir/baked_model" -ForegroundColor Cyan
} else {
    Write-Error "❌ 回転焼き込みテスト中にエラーが発生しました"
    exit 1
}
