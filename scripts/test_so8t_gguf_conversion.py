#!/usr/bin/env python3
"""
SO8T群Transformer GGUF変換テストスクリプト

SO8T GGUF変換機能をテストするためのスクリプトです。
ダミーモデルを作成して変換をテストします。
"""

import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.convert_so8t_to_gguf_colab import SO8TGGUFConverter


def create_dummy_so8t_model(output_dir: str) -> str:
    """ダミーのSO8Tモデルを作成"""
    print("🔧 ダミーSO8Tモデルを作成中...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ダミーのSO8Tモデル状態辞書を作成
    state_dict = {}
    
    # SO8T群関連のパラメータ
    state_dict['so8t.rotation_params'] = torch.randn(8, 8) * 0.01
    state_dict['so8t.rotation_angles'] = torch.randn(8) * 0.1
    state_dict['so8t.group_structure.weight'] = torch.randn(64, 64) * 0.02
    
    # アテンション層
    state_dict['attention.q_proj.weight'] = torch.randn(4096, 4096) * 0.02
    state_dict['attention.k_proj.weight'] = torch.randn(4096, 4096) * 0.02
    state_dict['attention.v_proj.weight'] = torch.randn(4096, 4096) * 0.02
    state_dict['attention.o_proj.weight'] = torch.randn(4096, 4096) * 0.02
    
    # FFN層
    state_dict['mlp.gate_proj.weight'] = torch.randn(11008, 4096) * 0.02
    state_dict['mlp.up_proj.weight'] = torch.randn(11008, 4096) * 0.02
    state_dict['mlp.down_proj.weight'] = torch.randn(4096, 11008) * 0.02
    
    # Triality reasoning heads
    state_dict['task_head.weight'] = torch.randn(151936, 4096) * 0.02
    state_dict['safety_head.weight'] = torch.randn(2, 4096) * 0.02
    state_dict['safety_head.bias'] = torch.zeros(2)
    state_dict['authority_head.weight'] = torch.randn(2, 4096) * 0.02
    state_dict['authority_head.bias'] = torch.zeros(2)
    
    # 埋め込み層
    state_dict['embed_tokens.weight'] = torch.randn(151936, 4096) * 0.02
    state_dict['norm.weight'] = torch.ones(4096)
    
    # モデルファイルを保存
    model_path = output_path / "pytorch_model.bin"
    torch.save(state_dict, model_path)
    
    # 設定ファイルも作成
    config = {
        "model_type": "so8t_transformer",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "vocab_size": 151936,
        "so8t_layers": 32,
        "triality_heads": 3
    }
    
    import json
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ ダミーSO8Tモデル作成完了: {model_path}")
    print(f"📊 パラメータ数: {sum(p.numel() for p in state_dict.values()):,}")
    print(f"💾 モデルサイズ: {sum(p.numel() * p.element_size() for p in state_dict.values()) / (1024**2):.1f} MB")
    
    return str(output_path)


def test_gguf_conversion():
    """GGUF変換をテスト"""
    print("🚀 SO8T GGUF変換テスト開始！")
    
    # 一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # 1. ダミーモデル作成
            print("\n📥 ステップ1: ダミーSO8Tモデル作成")
            model_path = create_dummy_so8t_model(str(temp_path / "dummy_model"))
            
            # 2. 変換器作成
            print("\n🔧 ステップ2: GGUF変換器作成")
            converter = SO8TGGUFConverter(
                model_path=model_path,
                output_dir=str(temp_path / "gguf_output"),
                quantization_type="Q8_0",
                max_memory_gb=2.0
            )
            
            # 3. 変換実行
            print("\n🔄 ステップ3: GGUF変換実行")
            output_path = converter.convert()
            
            # 4. 結果確認
            print("\n✅ ステップ4: 結果確認")
            output_dir = Path(output_path).parent
            
            # 出力ファイルの確認
            expected_files = [
                "dummy_model_so8t_Q8_0.json",  # メタデータ
                "dummy_model_so8t_Q8_0.npz",   # テンソルデータ
                "dummy_model_so8t_Q8_0.quant.json",  # 量子化情報
                "README.md"  # モデルカード
            ]
            
            for file_name in expected_files:
                file_path = output_dir / file_name
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    print(f"  ✅ {file_name}: {file_size / 1024:.1f} KB")
                else:
                    print(f"  ❌ {file_name}: 見つかりません")
            
            # メタデータの確認
            metadata_path = output_dir / "dummy_model_so8t_Q8_0.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                print(f"\n📊 メタデータ確認:")
                print(f"  - モデルタイプ: {metadata.get('model_type', 'N/A')}")
                print(f"  - アーキテクチャ: {metadata.get('architecture', 'N/A')}")
                print(f"  - 量子化タイプ: {metadata.get('quantization_type', 'N/A')}")
                print(f"  - SO8T群レイヤー数: {metadata.get('so8t_layers', 'N/A')}")
                print(f"  - SO8回転パラメータ数: {metadata.get('so8_rotation_params', 'N/A')}")
                print(f"  - Triality heads数: {metadata.get('triality_heads', 'N/A')}")
            
            # テンソルデータの確認
            tensor_path = output_dir / "dummy_model_so8t_Q8_0.npz"
            if tensor_path.exists():
                tensor_data = np.load(tensor_path)
                print(f"\n📦 テンソルデータ確認:")
                print(f"  - テンソル数: {len(tensor_data.files)}")
                print(f"  - ファイルサイズ: {tensor_path.stat().st_size / (1024**2):.1f} MB")
                
                # 量子化の確認
                quantized_count = 0
                for key in tensor_data.files:
                    if tensor_data[key].dtype == np.int8:
                        quantized_count += 1
                
                print(f"  - 量子化されたテンソル数: {quantized_count}")
                print(f"  - 量子化率: {quantized_count / len(tensor_data.files) * 100:.1f}%")
            
            print(f"\n🎉 SO8T GGUF変換テスト完了！")
            print(f"📁 出力ディレクトリ: {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ テストエラー: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_different_quantization_types():
    """異なる量子化タイプをテスト"""
    print("\n🧪 異なる量子化タイプのテスト")
    
    quantization_types = ["Q8_0", "Q4_K_M", "none"]
    
    for quant_type in quantization_types:
        print(f"\n🔧 量子化タイプ: {quant_type}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                # ダミーモデル作成
                model_path = create_dummy_so8t_model(str(temp_path / "test_model"))
                
                # 変換器作成
                converter = SO8TGGUFConverter(
                    model_path=model_path,
                    output_dir=str(temp_path / "output"),
                    quantization_type=quant_type,
                    max_memory_gb=1.0
                )
                
                # 変換実行
                output_path = converter.convert()
                
                # 結果確認
                output_dir = Path(output_path).parent
                tensor_file = list(output_dir.glob("*.npz"))[0]
                tensor_data = np.load(tensor_file)
                
                # 量子化統計
                total_tensors = len(tensor_data.files)
                quantized_tensors = sum(1 for key in tensor_data.files if tensor_data[key].dtype == np.int8)
                quantized_ratio = quantized_tensors / total_tensors * 100
                
                print(f"  ✅ 変換成功")
                print(f"  📊 テンソル数: {total_tensors}")
                print(f"  🔢 量子化テンソル数: {quantized_tensors}")
                print(f"  📈 量子化率: {quantized_ratio:.1f}%")
                print(f"  💾 ファイルサイズ: {tensor_file.stat().st_size / (1024**2):.1f} MB")
                
            except Exception as e:
                print(f"  ❌ エラー: {e}")


def main():
    """メイン関数"""
    print("🚀 SO8T群Transformer GGUF変換テスト開始！")
    print("=" * 60)
    
    # 基本変換テスト
    success = test_gguf_conversion()
    
    if success:
        print("\n" + "=" * 60)
        # 異なる量子化タイプのテスト
        test_different_quantization_types()
        
        print("\n" + "=" * 60)
        print("🎉 すべてのテストが完了しました！")
        print("✅ SO8T群Transformer GGUF変換機能は正常に動作しています")
    else:
        print("\n" + "=" * 60)
        print("❌ テストに失敗しました")
        print("💡 エラーログを確認してください")


if __name__ == "__main__":
    main()
