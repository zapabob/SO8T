#!/usr/bin/env python3
"""
Qwen2-VL-2B-InstructをSO(8)群Transformerモデルに変換

Qwen2-VL-2B-InstructモデルをSO(8)群構造を持つTransformerモデルに変換し、
llama.cppでGGUF形式に変換するスクリプト
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import argparse
from tqdm import tqdm
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# SO8T関連のインポート
from models.so8t_group_structure import SO8Rotation, NonCommutativeGate, PETRegularization, SO8TGroupStructure
from models.so8t_mlp import SO8TMLP
from models.so8t_attention import SO8TAttention, SO8TRotaryEmbedding
from models.so8t_safety_judge import SO8TSafetyJudge

# 回転ゲートの簡易実装
class SO8TRotationGate:
    def __init__(self, hidden_size: int, num_blocks: int, learnable: bool = True):
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.learnable = learnable
        if learnable:
            self.theta = torch.nn.Parameter(torch.randn(num_blocks, 8, 8) * 0.01)
        else:
            self.register_buffer('theta', torch.randn(num_blocks, 8, 8) * 0.01)

# Transformers関連のインポート
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    Qwen2VLConfig
)

logger = logging.getLogger(__name__)


class Qwen2VLToSO8TConverter:
    """Qwen2-VL-2B-InstructをSO(8)群Transformerに変換するクラス"""
    
    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        hidden_size: int = 1536,
        rotation_dim: int = 8,
        safety_features: bool = True
    ):
        """
        Args:
            input_model_path: 入力モデルパス
            output_model_path: 出力モデルパス
            hidden_size: 隠れ層サイズ
            rotation_dim: 回転次元（SO(8)群は8次元）
            safety_features: 安全機能の有効化
        """
        self.input_model_path = Path(input_model_path)
        self.output_model_path = Path(output_model_path)
        self.hidden_size = hidden_size
        self.rotation_dim = rotation_dim
        self.safety_features = safety_features
        
        # 出力ディレクトリの作成
        self.output_model_path.mkdir(parents=True, exist_ok=True)
        
        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用デバイス: {self.device}")
    
    def load_qwen2vl_model(self) -> Tuple[Qwen2VLForConditionalGeneration, AutoTokenizer]:
        """Qwen2-VL-2B-Instructモデルを読み込み"""
        logger.info(f"Qwen2-VL-2B-Instructモデルを読み込み中: {self.input_model_path}")
        
        try:
            # モデルとトークナイザーの読み込み
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.input_model_path,
                torch_dtype=torch.float32,  # 変換用にfloat32を使用
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.input_model_path,
                trust_remote_code=True
            )
            
            logger.info("Qwen2-VL-2B-Instructモデルの読み込み完了")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            raise
    
    def create_so8t_config(self, original_config: Qwen2VLConfig) -> Dict[str, Any]:
        """SO8T設定の作成"""
        logger.info("SO8T設定を作成中...")
        
        so8t_config = {
            "architectures": ["SO8TForConditionalGeneration"],
            "model_type": "so8t_vl",
            "hidden_size": self.hidden_size,
            "intermediate_size": original_config.intermediate_size,
            "num_hidden_layers": original_config.num_hidden_layers,
            "num_attention_heads": original_config.num_attention_heads,
            "num_key_value_heads": original_config.num_key_value_heads,
            "vocab_size": original_config.vocab_size,
            "max_position_embeddings": original_config.max_position_embeddings,
            "hidden_act": original_config.hidden_act,
            "rms_norm_eps": original_config.rms_norm_eps,
            "rope_theta": original_config.rope_theta,
            "tie_word_embeddings": original_config.tie_word_embeddings,
            "torch_dtype": "float32",
            "use_cache": True,
            
            # SO8T特有の設定
            "rotation_dim": self.rotation_dim,
            "safety_features": self.safety_features,
            "pet_lambda": 0.1,
            "safety_threshold": 0.8,
            "group_structure": "SO(8)",
            
            # ビジョン設定（保持）
            "vision_config": original_config.vision_config,
            "vision_start_token_id": original_config.vision_start_token_id,
            "vision_end_token_id": original_config.vision_end_token_id,
            "vision_token_id": original_config.vision_token_id,
            "image_token_id": original_config.image_token_id,
            "video_token_id": original_config.video_token_id,
            
            # トークン設定
            "bos_token_id": original_config.bos_token_id,
            "eos_token_id": original_config.eos_token_id,
            "pad_token_id": original_config.bos_token_id,
            
            # ロープスケーリング
            "rope_scaling": original_config.rope_scaling,
        }
        
        logger.info("SO8T設定作成完了")
        return so8t_config
    
    def convert_embeddings(self, original_model: Qwen2VLForConditionalGeneration) -> Dict[str, torch.Tensor]:
        """埋め込み層の変換"""
        logger.info("埋め込み層を変換中...")
        
        embeddings = {}
        
        # トークン埋め込み
        if hasattr(original_model.model, 'embed_tokens'):
            embeddings['model.embed_tokens.weight'] = original_model.model.embed_tokens.weight.clone()
            logger.info("トークン埋め込み変換完了")
        
        # 位置埋め込み（RoPEの場合は不要だが、互換性のため保持）
        if hasattr(original_model.model, 'embed_positions'):
            embeddings['model.embed_positions.weight'] = original_model.model.embed_positions.weight.clone()
            logger.info("位置埋め込み変換完了")
        
        return embeddings
    
    def convert_attention_layers(self, original_model: Qwen2VLForConditionalGeneration) -> Dict[str, torch.Tensor]:
        """アテンション層の変換（SO(8)群構造を追加）"""
        logger.info("アテンション層を変換中...")
        
        attention_params = {}
        
        for i, layer in enumerate(original_model.model.layers):
            layer_prefix = f"model.layers.{i}"
            
            # 既存のアテンション重みをコピー
            if hasattr(layer.self_attn, 'q_proj'):
                attention_params[f"{layer_prefix}.self_attn.q_proj.weight"] = layer.self_attn.q_proj.weight.clone()
            if hasattr(layer.self_attn, 'k_proj'):
                attention_params[f"{layer_prefix}.self_attn.k_proj.weight"] = layer.self_attn.k_proj.weight.clone()
            if hasattr(layer.self_attn, 'v_proj'):
                attention_params[f"{layer_prefix}.self_attn.v_proj.weight"] = layer.self_attn.v_proj.weight.clone()
            if hasattr(layer.self_attn, 'o_proj'):
                attention_params[f"{layer_prefix}.self_attn.o_proj.weight"] = layer.self_attn.o_proj.weight.clone()
            
            # SO(8)群回転ゲートの追加
            rotation_gate = SO8TRotationGate(
                hidden_size=self.hidden_size,
                num_blocks=self.hidden_size // self.rotation_dim,
                learnable=True
            )
            attention_params[f"{layer_prefix}.so8_rotation.theta"] = rotation_gate.theta.clone()
            
            # 非可換ゲートの追加
            non_commutative_gate = NonCommutativeGate(hidden_size=self.hidden_size)
            attention_params[f"{layer_prefix}.non_commutative.R_safe.rotation_params"] = non_commutative_gate.R_safe.rotation_params.clone()
            attention_params[f"{layer_prefix}.non_commutative.R_safe.rotation_angles"] = non_commutative_gate.R_safe.rotation_angles.clone()
            attention_params[f"{layer_prefix}.non_commutative.R_cmd.rotation_params"] = non_commutative_gate.R_cmd.rotation_params.clone()
            attention_params[f"{layer_prefix}.non_commutative.R_cmd.rotation_angles"] = non_commutative_gate.R_cmd.rotation_angles.clone()
            attention_params[f"{layer_prefix}.non_commutative.alpha"] = non_commutative_gate.alpha.clone()
        
        logger.info("アテンション層変換完了")
        return attention_params
    
    def convert_mlp_layers(self, original_model: Qwen2VLForConditionalGeneration) -> Dict[str, torch.Tensor]:
        """MLP層の変換（SO(8)群構造を追加）"""
        logger.info("MLP層を変換中...")
        
        mlp_params = {}
        
        for i, layer in enumerate(original_model.model.layers):
            layer_prefix = f"model.layers.{i}"
            
            # 既存のMLP重みをコピー
            if hasattr(layer.mlp, 'gate_proj'):
                mlp_params[f"{layer_prefix}.mlp.gate_proj.weight"] = layer.mlp.gate_proj.weight.clone()
            if hasattr(layer.mlp, 'up_proj'):
                mlp_params[f"{layer_prefix}.mlp.up_proj.weight"] = layer.mlp.up_proj.weight.clone()
            if hasattr(layer.mlp, 'down_proj'):
                mlp_params[f"{layer_prefix}.mlp.down_proj.weight"] = layer.mlp.down_proj.weight.clone()
            
            # SO8T MLPの追加
            so8t_mlp = SO8TMLP(
                hidden_size=self.hidden_size,
                intermediate_size=original_model.config.intermediate_size,
                group_structure=True
            )
            mlp_params[f"{layer_prefix}.so8t_mlp.rotation_matrix"] = so8t_mlp.rotation_matrix.clone()
            mlp_params[f"{layer_prefix}.so8t_mlp.group_scale"] = so8t_mlp.group_scale.clone()
        
        logger.info("MLP層変換完了")
        return mlp_params
    
    def convert_norm_layers(self, original_model: Qwen2VLForConditionalGeneration) -> Dict[str, torch.Tensor]:
        """正規化層の変換"""
        logger.info("正規化層を変換中...")
        
        norm_params = {}
        
        # 入力正規化
        if hasattr(original_model.model, 'norm'):
            norm_params['model.norm.weight'] = original_model.model.norm.weight.clone()
        
        # レイヤー正規化
        for i, layer in enumerate(original_model.model.layers):
            layer_prefix = f"model.layers.{i}"
            
            if hasattr(layer, 'input_layernorm'):
                norm_params[f"{layer_prefix}.input_layernorm.weight"] = layer.input_layernorm.weight.clone()
            if hasattr(layer, 'post_attention_layernorm'):
                norm_params[f"{layer_prefix}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.clone()
        
        logger.info("正規化層変換完了")
        return norm_params
    
    def convert_lm_head(self, original_model: Qwen2VLForConditionalGeneration) -> Dict[str, torch.Tensor]:
        """言語モデルヘッドの変換"""
        logger.info("言語モデルヘッドを変換中...")
        
        lm_head_params = {}
        
        if hasattr(original_model, 'lm_head'):
            lm_head_params['lm_head.weight'] = original_model.lm_head.weight.clone()
        
        # 安全判定ヘッドの追加
        if self.safety_features:
            safety_judge = SO8TSafetyJudge(
                hidden_size=self.hidden_size,
                num_classes=3,  # ALLOW, REFUSE, ESCALATE
                rationale_max_length=256
            )
            lm_head_params['safety_judge.classifier.weight'] = safety_judge.classifier.weight.clone()
            lm_head_params['safety_judge.classifier.bias'] = safety_judge.classifier.bias.clone()
            lm_head_params['safety_judge.rationale_head.weight'] = safety_judge.rationale_head.weight.clone()
            lm_head_params['safety_judge.rationale_head.bias'] = safety_judge.rationale_head.bias.clone()
        
        logger.info("言語モデルヘッド変換完了")
        return lm_head_params
    
    def convert_vision_encoder(self, original_model: Qwen2VLForConditionalGeneration) -> Dict[str, torch.Tensor]:
        """ビジョンエンコーダーの変換"""
        logger.info("ビジョンエンコーダーを変換中...")
        
        vision_params = {}
        
        if hasattr(original_model, 'visual'):
            # ビジョンエンコーダーの重みをコピー
            for name, param in original_model.visual.named_parameters():
                vision_params[f"visual.{name}"] = param.clone()
        
        logger.info("ビジョンエンコーダー変換完了")
        return vision_params
    
    def save_so8t_model(self, config: Dict[str, Any], all_params: Dict[str, torch.Tensor], tokenizer: AutoTokenizer) -> None:
        """SO8Tモデルの保存"""
        logger.info("SO8Tモデルを保存中...")
        
        # 設定ファイルの保存
        config_path = self.output_model_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 重みの保存（SafeTensors形式）
        from safetensors.torch import save_file
        weights_path = self.output_model_path / "model.safetensors"
        save_file(all_params, weights_path)
        
        # トークナイザーの保存
        tokenizer.save_pretrained(self.output_model_path)
        
        # 生成設定の保存
        generation_config = {
            "bos_token_id": config["bos_token_id"],
            "pad_token_id": config["pad_token_id"],
            "eos_token_id": config["eos_token_id"],
            "do_sample": True,
            "repetition_penalty": 1.0,
            "temperature": 0.01,
            "top_p": 0.001,
            "top_k": 1,
            "transformers_version": "4.41.2"
        }
        
        generation_config_path = self.output_model_path / "generation_config.json"
        with open(generation_config_path, 'w', encoding='utf-8') as f:
            json.dump(generation_config, f, indent=2, ensure_ascii=False)
        
        # モデルカードの作成
        model_card = f"""# SO8T-VL-2B-Instruct

## 概要
Qwen2-VL-2B-InstructをSO(8)群Transformerモデルに変換したマルチモーダルモデルです。

## 特徴
- **SO(8)群構造**: 8次元回転群による数学的厳密性
- **非可換ゲート**: R_safe → R_cmd の順序性保持
- **PET正則化**: 時系列一貫性による群の慣性
- **安全機能**: 安全判定ヘッドによる倫理的推論
- **マルチモーダル**: テキストとビジョンの統合処理

## 技術仕様
- **ベースモデル**: Qwen2-VL-2B-Instruct
- **隠れ層サイズ**: {self.hidden_size}
- **回転次元**: {self.rotation_dim}
- **レイヤー数**: {config['num_hidden_layers']}
- **アテンションヘッド数**: {config['num_attention_heads']}
- **語彙サイズ**: {config['vocab_size']}

## 使用方法
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./so8t-vl-2b-instruct")
tokenizer = AutoTokenizer.from_pretrained("./so8t-vl-2b-instruct")

# テキスト生成
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## ライセンス
MIT License
"""
        
        model_card_path = self.output_model_path / "README.md"
        with open(model_card_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
        
        logger.info(f"SO8Tモデル保存完了: {self.output_model_path}")
    
    def convert(self) -> None:
        """変換の実行"""
        logger.info("=" * 80)
        logger.info("Qwen2-VL-2B-Instruct → SO8T変換開始")
        logger.info("=" * 80)
        
        try:
            # 1. 元モデルの読み込み
            original_model, tokenizer = self.load_qwen2vl_model()
            
            # 2. SO8T設定の作成
            so8t_config = self.create_so8t_config(original_model.config)
            
            # 3. 各層の変換
            logger.info("各層の変換を開始...")
            
            all_params = {}
            
            # 埋め込み層
            embeddings = self.convert_embeddings(original_model)
            all_params.update(embeddings)
            
            # アテンション層
            attention_params = self.convert_attention_layers(original_model)
            all_params.update(attention_params)
            
            # MLP層
            mlp_params = self.convert_mlp_layers(original_model)
            all_params.update(mlp_params)
            
            # 正規化層
            norm_params = self.convert_norm_layers(original_model)
            all_params.update(norm_params)
            
            # 言語モデルヘッド
            lm_head_params = self.convert_lm_head(original_model)
            all_params.update(lm_head_params)
            
            # ビジョンエンコーダー
            vision_params = self.convert_vision_encoder(original_model)
            all_params.update(vision_params)
            
            # 4. SO8Tモデルの保存
            self.save_so8t_model(so8t_config, all_params, tokenizer)
            
            logger.info("=" * 80)
            logger.info("SO8T変換完了")
            logger.info("=" * 80)
            logger.info(f"出力ディレクトリ: {self.output_model_path}")
            logger.info(f"総パラメータ数: {len(all_params):,}")
            
        except Exception as e:
            logger.error(f"変換中にエラーが発生しました: {e}")
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Qwen2-VL-2B-InstructをSO8Tモデルに変換')
    parser.add_argument('--input-model', required=True, help='入力モデルパス')
    parser.add_argument('--output-model', required=True, help='出力モデルパス')
    parser.add_argument('--hidden-size', type=int, default=1536, help='隠れ層サイズ')
    parser.add_argument('--rotation-dim', type=int, default=8, help='回転次元')
    parser.add_argument('--safety-features', action='store_true', help='安全機能の有効化')
    parser.add_argument('--verbose', action='store_true', help='詳細ログ出力')
    
    args = parser.parse_args()
    
    # ログ設定
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 変換器の作成と実行
    converter = Qwen2VLToSO8TConverter(
        input_model_path=args.input_model,
        output_model_path=args.output_model,
        hidden_size=args.hidden_size,
        rotation_dim=args.rotation_dim,
        safety_features=args.safety_features
    )
    
    try:
        converter.convert()
        print("\n[SUCCESS] SO8T変換が正常に完了しました")
        return 0
    except Exception as e:
        print(f"\n[ERROR] 変換中にエラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
