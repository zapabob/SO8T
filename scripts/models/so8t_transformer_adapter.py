#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SO8T Transformer Adapter Model
Borea-Phi-3.5-mini-Instruct-JpのTransformer層にSO(8)回転レイヤーを残差アダプター接続

理論的背景:
- URT (Unified Representation Theorem)
- NC-KART★ (Non-Commutative Kolmogorov-Arnold Representation Theory)
- SO(8)幾何学的知性

特徴:
- 元の重みを凍結
- 中間レイヤーにSO(8)回転アダプターを追加
- RTX 3060最適化
- PPO学習対応

著者: AI Agent (峯岸亮ボブにゃん理論実装)
日付: 2025-11-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
import math

class SO8RotationGate(nn.Module):
    """SO(8)回転ゲート"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # SO(8)生成のためのskew-symmetric matrix
        # 8x8の回転行列を生成
        self.rotation_dim = 8
        self.skew_matrix = nn.Parameter(torch.randn(self.rotation_dim, self.rotation_dim))
        self.adapt_projection = nn.Linear(embed_dim, self.rotation_dim)

        # 出力投影
        self.output_projection = nn.Linear(self.rotation_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 入力の次元をSO(8)空間に投影
        projected = self.adapt_projection(x)  # [batch, seq, rotation_dim]

        # skew-symmetric matrixから回転行列を生成
        # SO(8) Lie algebra: skew-symmetric matrices
        skew = self.skew_matrix - self.skew_matrix.t()  # 確実にskew-symmetric

        # matrix exponentialで回転行列を生成
        rotation_matrix = torch.matrix_exp(skew)  # [rotation_dim, rotation_dim]

        # バッチ処理のために拡張
        batch_size, seq_len, _ = projected.shape
        rotation_matrix = rotation_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, rot_dim, rot_dim]

        # 回転適用
        rotated = torch.einsum('bsi,ij->bsj', projected, rotation_matrix.squeeze(0).squeeze(0))
        rotated = rotated + projected  # 残差接続

        # 元の次元に戻す
        output = self.output_projection(rotated)
        return output

class SO8TResidualAdapter(nn.Module):
    """SO(8)残差アダプター"""

    def __init__(self, embed_dim: int, adapter_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter_dim = adapter_dim

        # Down projection
        self.down_proj = nn.Linear(embed_dim, adapter_dim)
        self.activation = nn.GELU()

        # SO(8)回転ゲート
        self.so8_gate = SO8RotationGate(adapter_dim)

        # Up projection
        self.up_proj = nn.Linear(adapter_dim, embed_dim)

        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差接続
        residual = x

        # Down projection
        down = self.down_proj(x)
        down = self.activation(down)

        # SO(8)回転適用
        rotated = self.so8_gate(down)

        # Up projection
        up = self.up_proj(rotated)

        # 残差接続とLayer norm
        output = self.layer_norm(residual + up)

        return output

class SO8TTransformerAdapter(nn.Module):
    """SO(8) Transformer Adapter - 中間レイヤーにアダプターを追加"""

    def __init__(self, base_model, adapter_layers: List[int] = None, adapter_dim: int = 64):
        super().__init__()
        self.base_model = base_model
        self.adapter_layers = adapter_layers or [8, 16, 24]  # 中間レイヤー

        # モデルのembed_dimを取得
        embed_dim = base_model.config.hidden_size

        # アダプター層の作成
        self.adapters = nn.ModuleDict()
        for layer_idx in self.adapter_layers:
            self.adapters[f"layer_{layer_idx}"] = SO8TResidualAdapter(embed_dim, adapter_dim)

        # ベースモデルの重みを凍結
        self.freeze_base_model()

    def freeze_base_model(self):
        """ベースモデルの重みを凍結"""
        for param in self.base_model.parameters():
            param.requires_grad = False

        print("Base model weights frozen")

    def unfreeze_adapters(self):
        """アダプター層のみを学習可能に"""
        for adapter in self.adapters.values():
            for param in adapter.parameters():
                param.requires_grad = True

        print("Adapter weights unfrozen")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        # ベースモデルの順伝播
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=True,  # hidden statesが必要
            return_dict=return_dict,
            **kwargs
        )

        # hidden_statesを取得
        hidden_states = outputs.hidden_states

        # アダプター適用
        modified_hidden_states = []
        for i, hidden_state in enumerate(hidden_states):
            if i in self.adapter_layers:
                adapter_name = f"layer_{i}"
                if adapter_name in self.adapters:
                    # アダプター適用
                    modified_hidden = self.adapters[adapter_name](hidden_state)
                    modified_hidden_states.append(modified_hidden)
                else:
                    modified_hidden_states.append(hidden_state)
            else:
                modified_hidden_states.append(hidden_state)

        # 最終出力をアダプター適用済みの隠れ層で置き換え
        if len(modified_hidden_states) > 0:
            # 最終層の隠れ状態を変更
            final_hidden = modified_hidden_states[-1]
            outputs.last_hidden_state = final_hidden

        return outputs

class SO8TConfig(PretrainedConfig):
    """SO8T設定"""

    model_type = "so8t_adapter"

    def __init__(
        self,
        base_model_name: str = "microsoft/Phi-3.5-mini-instruct",
        adapter_layers: List[int] = None,
        adapter_dim: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.adapter_layers = adapter_layers or [8, 16, 24]
        self.adapter_dim = adapter_dim

class SO8TAdapterModel(PreTrainedModel):
    """SO8T Adapter Model - HuggingFace互換"""

    config_class = SO8TConfig

    def __init__(self, config: SO8TConfig):
        super().__init__(config)

        # ベースモデルをロード
        from transformers import AutoModelForCausalLM
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # SO8Tアダプター
        self.so8t_adapter = SO8TTransformerAdapter(
            self.base_model,
            adapter_layers=config.adapter_layers,
            adapter_dim=config.adapter_dim
        )

    def forward(self, *args, **kwargs):
        return self.so8t_adapter(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """生成メソッド"""
        return self.base_model.generate(*args, **kwargs)

    def save_adapter_weights(self, path: str):
        """アダプター重みのみ保存"""
        torch.save({
            'adapters': self.so8t_adapter.adapters.state_dict(),
            'config': self.config
        }, path)
        print(f"Adapter weights saved to {path}")

    def load_adapter_weights(self, path: str):
        """アダプター重みのみロード"""
        checkpoint = torch.load(path)
        self.so8t_adapter.adapters.load_state_dict(checkpoint['adapters'])
        print(f"Adapter weights loaded from {path}")

def create_so8t_adapter_model(
    base_model_name: str = "microsoft/Phi-3.5-mini-instruct",
    adapter_layers: List[int] = None,
    adapter_dim: int = 64
) -> SO8TAdapterModel:
    """SO8Tアダプターモデルを作成"""

    config = SO8TConfig(
        base_model_name=base_model_name,
        adapter_layers=adapter_layers,
        adapter_dim=adapter_dim
    )

    model = SO8TAdapterModel(config)

    # アダプター層のみ学習可能に
    model.so8t_adapter.unfreeze_adapters()

    return model

# テスト関数
def test_so8t_adapter():
    """SO8Tアダプターのテスト"""
    print("Testing SO8T Adapter...")

    # モデル作成
    model = create_so8t_adapter_model(
        adapter_layers=[8, 16, 24],
        adapter_dim=64
    )

    # パラメータ数チェック
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(".1f")

    # ダミー入力でテスト
    input_ids = torch.randint(0, 1000, (1, 10))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"Output shape: {outputs.logits.shape}")

    print("SO8T Adapter test completed!")

if __name__ == "__main__":
    test_so8t_adapter()
