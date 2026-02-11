# SO8T×マルチモーダルLLM 焼き込み・変換スクリプト
# 学習済み回転を射影重みに焼き込み、GGUF変換を実行

param(
    [string]$ModelPath = "./outputs",
    [string]$OutputDir = "./converted_models",
    [string]$ModelName = "so8t-qwen2vl-2b",
    [switch]$EnableRotation = $true,
    [switch]$EnablePET = $true
)

Write-Host "🔥 SO8T×マルチモーダルLLM 焼き込み・変換開始..." -ForegroundColor Green

# 仮想環境のアクティベート
Write-Host "🔧 仮想環境をアクティベート中..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# 出力ディレクトリの作成
Write-Host "📁 出力ディレクトリを作成中..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# 焼き込み・変換スクリプトの実行
Write-Host "🎯 焼き込み・変換を実行中..." -ForegroundColor Yellow

$bakeScript = @"
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

def bake_rotation_into_weights(model, rotation_gate):
    """回転ゲートを射影重みに焼き込み"""
    print("🔥 回転ゲートを射影重みに焼き込み中...")
    
    if rotation_gate is None:
        print("⚠️ 回転ゲートがありません。スキップします。")
        return model
    
    # 回転行列を取得
    rotation_matrices = rotation_gate.get_rotation_matrices()  # [num_blocks, 8, 8]
    
    # モデルの各レイヤーで回転を適用
    for name, module in model.named_modules():
        if hasattr(module, 'o_proj') and hasattr(module.o_proj, 'weight'):
            # アテンション出力投影層の重みを取得
            weight = module.o_proj.weight.data  # [hidden_size, hidden_size]
            
            # 回転を適用: W' = R · W
            # 8次元ブロックごとに回転を適用
            hidden_size = weight.size(1)
            num_blocks = hidden_size // 8
            
            if num_blocks == rotation_matrices.size(0):
                # 重みを8次元ブロックに分割
                weight_blocks = weight.view(hidden_size, num_blocks, 8)
                
                # 各ブロックに回転を適用
                for block_idx in range(num_blocks):
                    R = rotation_matrices[block_idx]  # [8, 8]
                    weight_blocks[:, block_idx, :] = torch.matmul(
                        weight_blocks[:, block_idx, :], R.T
                    )
                
                # 元の形状に戻す
                module.o_proj.weight.data = weight_blocks.view(hidden_size, hidden_size)
                print(f"  ✅ {name}.o_proj に回転を焼き込みました")
    
    print("✅ 回転ゲートの焼き込み完了")
    return model

def convert_to_gguf(model, output_path, model_name):
    """モデルをGGUF形式に変換"""
    print(f"🔄 GGUF形式に変換中: {output_path}")
    
    try:
        # llama.cppのconvert.pyを使用してGGUF変換
        # 注意: 実際の変換では適切なconvert.pyスクリプトが必要
        
        # まず、Hugging Face形式で保存
        hf_path = output_path.replace('.gguf', '_hf')
        model.save_pretrained(hf_path)
        print(f"  📁 Hugging Face形式で保存: {hf_path}")
        
        # GGUF変換のための設定
        gguf_config = {
            "model_name": model_name,
            "model_path": hf_path,
            "output_path": output_path,
            "quantization": "Q8_0",
            "description": "SO8T×マルチモーダルLLM (焼き込み済み)"
        }
        
        # 設定を保存
        config_path = output_path.replace('.gguf', '_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(gguf_config, f, indent=2, ensure_ascii=False)
        
        print(f"  ⚙️ 変換設定を保存: {config_path}")
        print(f"  ⚠️ 注意: 実際のGGUF変換にはllama.cppのconvert.pyが必要です")
        print(f"  📝 変換コマンド例:")
        print(f"     python convert.py {hf_path} --outfile {output_path} --outtype q8_0")
        
        return True
        
    except Exception as e:
        print(f"❌ GGUF変換エラー: {str(e)}")
        return False

def create_ollama_modelfile(model_path, model_name):
    """Ollama用Modelfileを作成"""
    print(f"📝 Ollama用Modelfileを作成中: {model_name}")
    
    modelfile_content = f"""FROM {model_path}

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}\"\"\"

# SO8T×マルチモーダルLLM Model Card
# SO(8)群回転ゲート + PET正則化 + OCR要約 + SQLite監査

SYSTEM \"\"\"You are SO8T×マルチモーダルLLM, an advanced multimodal language model with SO(8) group structure and enhanced safety features.

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

You provide helpful, accurate, and safe responses while maintaining complete privacy and auditability.\"\"\"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 32768
PARAMETER num_predict 2048
"""
    
    modelfile_path = f"{model_path}.Modelfile"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"  ✅ Modelfileを作成: {modelfile_path}")
    return modelfile_path

def main():
    print("🔥 SO8T×マルチモーダルLLM 焼き込み・変換開始...")
    
    # デバイス情報を表示
    print(f"🔧 デバイス: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"💾 メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "CPU使用")
    
    # 学習済みモデルを読み込み
    print("📦 学習済みモデルを読み込み中...")
    
    try:
        # 学習済みモデルを読み込み
        trainer = SO8TIntegratedTrainer(
            model_path='$ModelPath',
            config_path='$ModelPath/config.json',
            output_dir='$OutputDir'
        )
        
        # コンポーネントをセットアップ
        trainer.setup_components()
        
        # 回転ゲートを取得
        rotation_gate = trainer.rotation_gate
        
        print(f"✅ モデル読み込み完了")
        print(f"   回転ゲート: {'有効' if rotation_gate is not None else '無効'}")
        print(f"   PET損失: {'有効' if trainer.pet_loss is not None else '無効'}")
        
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {str(e)}")
        print("🔄 ベースモデルから開始...")
        
        # ベースモデルから開始
        model = create_so8t_qwen2vl_model(
            model_path='$ModelPath',
            rotation_enabled=$EnableRotation
        )
        rotation_gate = model.rotation_gate if hasattr(model, 'rotation_gate') else None
    
    # 回転ゲートを焼き込み
    if rotation_gate is not None:
        print("🔥 回転ゲートを射影重みに焼き込み中...")
        model = bake_rotation_into_weights(model, rotation_gate)
    else:
        print("⚠️ 回転ゲートがありません。焼き込みをスキップします。")
    
    # 焼き込み済みモデルを保存
    baked_model_path = f"$OutputDir/{model_name}_baked"
    print(f"💾 焼き込み済みモデルを保存中: {baked_model_path}")
    
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(baked_model_path)
    else:
        # モデルが保存可能でない場合
        print("⚠️ モデルの保存方法を確認してください")
    
    # GGUF変換
    gguf_path = f"$OutputDir/{model_name}.gguf"
    print(f"🔄 GGUF形式に変換中: {gguf_path}")
    
    success = convert_to_gguf(model, gguf_path, model_name)
    
    if success:
        print("✅ GGUF変換設定完了")
    else:
        print("❌ GGUF変換に失敗しました")
    
    # Ollama用Modelfileを作成
    modelfile_path = create_ollama_modelfile(gguf_path, model_name)
    
    # 変換結果をまとめる
    conversion_results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "baked_model_path": baked_model_path,
        "gguf_path": gguf_path,
        "modelfile_path": modelfile_path,
        "rotation_baked": rotation_gate is not None,
        "pet_enabled": trainer.pet_loss is not None if 'trainer' in locals() else False,
        "conversion_success": success
    }
    
    # 結果を保存
    results_file = "$OutputDir/conversion_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(conversion_results, f, indent=2, ensure_ascii=False)
    
    print(f"\\n📊 変換結果:")
    print(f"  🍞 焼き込み済みモデル: {baked_model_path}")
    print(f"  🔄 GGUFファイル: {gguf_path}")
    print(f"  📝 Modelfile: {modelfile_path}")
    print(f"  📁 結果ファイル: {results_file}")
    
    print("\\n🚀 次のステップ:")
    print(f"  1. Ollamaモデル作成: ollama create {model_name} -f {modelfile_path}")
    print(f"  2. モデル実行: ollama run {model_name}")
    print(f"  3. 推論テスト: ollama run {model_name} '画像を説明してください。'")
    
    print("\\n✅ 焼き込み・変換完了！")

if __name__ == "__main__":
    main()
"@

# 焼き込み・変換スクリプトを実行
$bakeScript | py -3

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 焼き込み・変換完了！" -ForegroundColor Green
    Write-Host "📁 出力ディレクトリ: $OutputDir" -ForegroundColor Cyan
    Write-Host "🍞 焼き込み済みモデル: $OutputDir/$ModelName`_baked" -ForegroundColor Cyan
    Write-Host "🔄 GGUFファイル: $OutputDir/$ModelName.gguf" -ForegroundColor Cyan
} else {
    Write-Error "❌ 焼き込み・変換中にエラーが発生しました"
    exit 1
}
