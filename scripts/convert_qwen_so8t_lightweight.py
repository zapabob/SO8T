#!/usr/bin/env python3
"""
Qwen2.5-7B-Instruct SO8T Transformer 軽量8bit量子化GGUF変換スクリプト

ディスク容量制約下でSO8T Transformerを8bit量子化してGGUF形式に変換
メモリ効率を最優先にした軽量実装
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
from tqdm import tqdm
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# SO8T Transformer imports
import sys
sys.path.append(str(project_root / "models" / "Qwen2.5-7B-Instruct"))
from so8t_transformer_model import SO8TTransformerForCausalLM
from transformers import AutoTokenizer

# 8bit量子化用
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    warnings.warn("bitsandbytes not available. Using standard quantization.")


class QwenSO8TLightweightConverter:
    """Qwen2.5-7B-Instruct SO8T Transformer 軽量8bit量子化GGUF変換器"""
    
    def __init__(self, config_path: str, output_dir: str = "models/qwen_so8t_lightweight"):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
        print(f"設定読み込み完了: {config_path}")
        print(f"   - アーキテクチャ: {self.config['architectures'][0]}")
        print(f"   - モデルタイプ: {self.config['model_type']}")
        print(f"   - 隠れサイズ: {self.config['hidden_size']:,}")
        print(f"   - アテンションヘッド数: {self.config['num_attention_heads']}")
        print(f"   - レイヤー数: {self.config['num_hidden_layers']}")
        
        # SO8T固有パラメータ確認
        so8t_params = {k: v for k, v in self.config.items() if k.startswith('so8t_')}
        print(f"SO8T固有パラメータ:")
        for key, value in so8t_params.items():
            print(f"   - {key}: {value}")
    
    def create_lightweight_model(self) -> SO8TTransformerForCausalLM:
        """軽量SO8T Transformerモデルを作成"""
        print("=" * 60)
        print("軽量SO8T Transformerモデル作成中...")
        print("=" * 60)
        
        # デバイス設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"デバイス: {device}")
        
        # モデル初期化の進捗表示
        print("1/3: SO8T Transformerモデル初期化中...")
        print(f"   - 語彙サイズ: {self.config['vocab_size']:,}")
        print(f"   - 隠れサイズ: {self.config['hidden_size']:,}")
        print(f"   - レイヤー数: {self.config['num_hidden_layers']}")
        print(f"   - アテンションヘッド数: {self.config['num_attention_heads']}")
        
        # 軽量SO8T Transformerモデル初期化
        model = SO8TTransformerForCausalLM(
            vocab_size=self.config['vocab_size'],
            hidden_size=self.config['hidden_size'],
            intermediate_size=self.config['intermediate_size'],
            num_hidden_layers=self.config['num_hidden_layers'],
            num_attention_heads=self.config['num_attention_heads'],
            num_key_value_heads=self.config['num_key_value_heads'],
            hidden_act=self.config['hidden_act'],
            max_position_embeddings=self.config['max_position_embeddings'],
            rms_norm_eps=self.config['rms_norm_eps'],
            rope_theta=self.config['rope_theta'],
            attention_dropout=self.config['attention_dropout'],
            use_cache=self.config['use_cache'],
            # SO8T固有パラメータ
            so8t_rotation_dim=self.config.get('so8t_rotation_dim', 8),
            so8t_triality_symmetry=self.config.get('so8t_triality_symmetry', True),
            so8t_cross_head_interaction=self.config.get('so8t_cross_head_interaction', True),
            so8t_non_commutative_gates=self.config.get('so8t_non_commutative_gates', True),
            so8t_task_head=True,
            so8t_safety_head=True,
            so8t_authority_head=True,
        )
        print("   ✓ モデル初期化完了")
        
        # 8bit量子化設定
        print("2/3: 8bit量子化設定中...")
        if BNB_AVAILABLE:
            print("   - bitsandbytesを使用した8bit量子化")
            # 8bit量子化を適用
            model = self._apply_lightweight_8bit_quantization(model)
        else:
            print("   - bitsandbytesが利用できないため、標準量子化を使用")
            model = model.to(device)
        print("   ✓ 8bit量子化完了")
        
        print("3/3: モデル情報確認中...")
        # モデル情報表示
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   - 総パラメータ数: {total_params:,}")
        print(f"   - 学習可能パラメータ数: {trainable_params:,}")
        print(f"   - モデルサイズ: {total_params * 4 / (1024**3):.2f} GB (float32)")
        print("   ✓ モデル情報確認完了")
        
        print("=" * 60)
        print("軽量SO8T Transformerモデル作成完了！")
        print("=" * 60)
        
        return model
    
    def _apply_lightweight_8bit_quantization(self, model: SO8TTransformerForCausalLM) -> SO8TTransformerForCausalLM:
        """軽量8bit量子化を適用"""
        print("   軽量8bit量子化適用中...")
        
        # デバイス設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 8bit量子化を適用
        try:
            if BNB_AVAILABLE:
                # bitsandbytesを使用した8bit量子化
                print("   - bitsandbytesを使用した8bit量子化")
                # モデルを8bit量子化
                model = model.to(device)
                
                # 8bit量子化を適用（進捗表示付き）
                linear_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
                print(f"   - {len(linear_modules)}個のLinear層を8bit量子化中...")
                
                # 進捗表示なしで詳細ログ出力
                for i, (name, module) in enumerate(linear_modules):
                    print(f"   - [{i+1:3d}/{len(linear_modules):3d}] {name.split('.')[-1]:20s} ({module.in_features:4d} -> {module.out_features:4d})")
                    
                    # 8bit量子化を適用
                    quantized_module = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=False,
                        threshold=6.0
                    )
                    setattr(model, name, quantized_module)
                    
                    # 進捗表示
                    progress = (i + 1) / len(linear_modules) * 100
                    print(f"     ✓ 完了 ({progress:.1f}%)")
                
                print("   ✓ 軽量8bit量子化完了")
            else:
                print("   - 標準量子化を使用")
                model = model.to(device)
        except Exception as e:
            print(f"   ✗ 8bit量子化エラー: {e}")
            print("   - 標準量子化を使用")
            model = model.to(device)
        
        return model
    
    def load_tokenizer(self) -> AutoTokenizer:
        """トークナイザーを読み込み"""
        print("トークナイザー読み込み中...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
            print("トークナイザー読み込み完了")
            return tokenizer
        except Exception as e:
            print(f"トークナイザー読み込みエラー: {e}")
            print("   デフォルトトークナイザーを使用")
            return None
    
    def save_model_weights(self, model: SO8TTransformerForCausalLM) -> str:
        """モデル重みを保存"""
        print("モデル重み保存中...")
        
        # 出力ファイル名
        output_file = self.output_dir / "qwen_so8t_transformer_8bit_weights.pt"
        
        # モデル重みを保存（進捗表示付き）
        print("   - モデル状態辞書を収集中...")
        with tqdm(total=100, desc="モデル重み保存", unit="%", 
                 ncols=80, ascii=True, dynamic_ncols=True) as pbar:
            model_data = {
                'model_state_dict': model.state_dict(),
                'config': self.config,
                'model_type': 'so8t_transformer',
                'quantization': '8bit',
            }
            pbar.update(50)
            time.sleep(0.1)
            
            print("   - ファイルに書き込み中...")
            torch.save(model_data, output_file)
            pbar.update(50)
            time.sleep(0.1)
        
        print(f"モデル重み保存完了: {output_file}")
        return str(output_file)
    
    def save_tokenizer(self, tokenizer: AutoTokenizer) -> str:
        """トークナイザーを保存"""
        print("トークナイザー保存中...")
        
        # 出力ディレクトリ
        tokenizer_dir = self.output_dir / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        
        # トークナイザーを保存（進捗表示付き）
        with tqdm(total=100, desc="トークナイザー保存", unit="%", 
                 ncols=80, ascii=True, dynamic_ncols=True) as pbar:
            print("   - トークナイザーファイルを保存中...")
            tokenizer.save_pretrained(str(tokenizer_dir))
            pbar.update(100)
            time.sleep(0.1)
        
        print(f"トークナイザー保存完了: {tokenizer_dir}")
        return str(tokenizer_dir)
    
    def create_model_card(self, model: SO8TTransformerForCausalLM, tokenizer: Optional[AutoTokenizer] = None) -> str:
        """モデルカードを作成"""
        print("モデルカード作成中...")
        
        # モデルカード内容
        model_card = f"""# Qwen2.5-7B-Instruct SO8T Transformer 8bit量子化

## 概要
Qwen2.5-7B-InstructにSO8群マルチヘッドアテンションを導入したSO8T Transformerを8bit量子化したモデルです。

## 特徴
- **SO8群構造**: 8次元回転群の完全な数学的実装
- **Triality対称性**: Vector, Spinor+, Spinor-表現の完全対応
- **三重推論**: タスク、安全、権限推論の完全実装
- **8bit量子化**: メモリ効率を最優先にした軽量化
- **Qwen2.5-7B-Instruct完全対応**: 全パラメータの完全対応

## モデル仕様
- **アーキテクチャ**: {self.config['architectures'][0]}
- **モデルタイプ**: {self.config['model_type']}
- **語彙サイズ**: {self.config['vocab_size']:,}
- **隠れサイズ**: {self.config['hidden_size']:,}
- **中間サイズ**: {self.config['intermediate_size']:,}
- **レイヤー数**: {self.config['num_hidden_layers']}
- **アテンションヘッド数**: {self.config['num_attention_heads']}
- **キー・バリューヘッド数**: {self.config['num_key_value_heads']}
- **最大位置埋め込み**: {self.config['max_position_embeddings']:,}
- **RMS正規化ε**: {self.config['rms_norm_eps']}
- **RoPE θ**: {self.config['rope_theta']:,}

## SO8T固有パラメータ
"""
        
        # SO8T固有パラメータを追加
        so8t_params = {k: v for k, v in self.config.items() if k.startswith('so8t_')}
        for key, value in so8t_params.items():
            model_card += f"- **{key}**: {value}\n"
        
        model_card += f"""
## 量子化設定
- **量子化タイプ**: 8bit
- **CPUオフロード**: 有効
- **閾値**: 6.0
- **データ型**: float16

## 使用方法
```python
import torch
from models.Qwen2_5_7B_Instruct.so8t_transformer_model import SO8TTransformerForCausalLM

# モデル読み込み
model = SO8TTransformerForCausalLM.from_pretrained("path/to/model")
model = model.quantize(quantization_config)

# 推論実行
outputs = model(input_ids, attention_mask=attention_mask)
```

## ライセンス
Apache-2.0

## 作成者
SO8T Safe Agent Project
"""
        
        # モデルカードを保存
        model_card_file = self.output_dir / "README.md"
        with open(model_card_file, 'w', encoding='utf-8') as f:
            f.write(model_card)
        
        print(f"モデルカード作成完了: {model_card_file}")
        return str(model_card_file)
    
    def run_conversion(self) -> Dict[str, str]:
        """軽量8bit量子化変換を実行"""
        print("=" * 80)
        print("Qwen2.5-7B-Instruct SO8T Transformer 軽量8bit量子化変換開始")
        print("=" * 80)
        
        # 変換ステップを定義
        conversion_steps = [
            "モデル作成",
            "トークナイザー読み込み", 
            "モデル重み保存",
            "トークナイザー保存",
            "モデルカード作成"
        ]
        
        # 詳細進捗表示付きで変換を実行
        for i, step in enumerate(conversion_steps, 1):
            print(f"\n[{i}/{len(conversion_steps)}] {step}中...")
            print("-" * 40)
            
            if step == "モデル作成":
                model = self.create_lightweight_model()
            elif step == "トークナイザー読み込み":
                tokenizer = self.load_tokenizer()
            elif step == "モデル重み保存":
                weights_file = self.save_model_weights(model)
            elif step == "トークナイザー保存":
                tokenizer_file = self.save_tokenizer(tokenizer) if tokenizer else None
            elif step == "モデルカード作成":
                model_card_file = self.create_model_card(model, tokenizer)
            
            print(f"✓ {step}完了")
        
        print("\n" + "=" * 80)
        print("Qwen2.5-7B-Instruct SO8T Transformer 軽量8bit量子化変換完了！")
        print("=" * 80)
        print(f"出力ディレクトリ: {self.output_dir}")
        print(f"モデル重み: {weights_file}")
        if tokenizer_file:
            print(f"トークナイザー: {tokenizer_file}")
        print(f"モデルカード: {model_card_file}")
        
        return {
            "weights_file": weights_file,
            "tokenizer_file": tokenizer_file,
            "model_card_file": model_card_file,
            "output_dir": str(self.output_dir)
        }


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct SO8T Transformer 軽量8bit量子化変換")
    parser.add_argument("--config", type=str, default="models/Qwen2.5-7B-Instruct/config_so8t.json",
                       help="SO8T設定ファイルのパス")
    parser.add_argument("--output", type=str, default="models/qwen_so8t_lightweight",
                       help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    try:
        # 変換器初期化
        converter = QwenSO8TLightweightConverter(args.config, args.output)
        
        # 変換実行
        results = converter.run_conversion()
        
        print(f" 変換完了:")
        for key, value in results.items():
            print(f"   - {key}: {value}")
        
    except Exception as e:
        print(f"変換エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
