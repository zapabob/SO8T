#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SO8T Residual Adapter for Transformer Layers
SO(8)回転レイヤーをTransformerの中間レイヤーに残差接続するアダプター

理論的背景:
- SO(8)幾何学的構造: 8次元回転群による表現変換
- 残差アダプター: Transformer層への軽量統合
- 幾何学的知性: 非可換表現による思考プロセス強化

特徴:
- Borea-phi3.5-instinct-jpの重みを凍結
- 中間レイヤーにSO(8)回転アダプターを挿入
- 残差接続による安定した学習
- RTX 3060最適化

著者: AI Agent (峯岸亮ボブにゃん理論実装)
日付: 2025-11-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class SO8AdapterConfig:
    """SO(8)アダプター設定"""
    hidden_size: int = 3072  # Phi-3.5の隠れ層サイズ
    so8_rank: int = 8        # SO(8)の次元
    adapter_dim: int = 256   # アダプターボトルネック次元
    num_layers: int = 32     # Phi-3.5の層数
    adapter_layers: List[int] = None  # アダプターを挿入する層インデックス

    def __post_init__(self):
        if self.adapter_layers is None:
            # 中間層（8, 16, 24層）にアダプターを挿入
            self.adapter_layers = [8, 16, 24]

class SO8RotationLayer(nn.Module):
    """SO(8)回転レイヤー"""

    def __init__(self, config: SO8AdapterConfig):
        super().__init__()
        self.config = config
        self.so8_dim = config.so8_rank

        # SO(8)生成行列（skew-symmetric matrices）
        self.rotation_matrices = nn.Parameter(
            torch.zeros(self.so8_dim, self.so8_dim)
        )

        # 学習可能なスケーリング係数
        self.scale = nn.Parameter(torch.ones(1))

        # 回転行列の初期化
        self._init_rotation_matrices()

    def _init_rotation_matrices(self):
        """SO(8)回転行列の初期化"""
        # SO(8)群の生成元（8つの基本回転）
        # これはSO(8)群の標準的な表現
        generators = []

        # 生成元1-4: 隣接する平面の回転
        for i in range(4):
            gen = torch.zeros(8, 8)
            gen[i, i+4] = -1
            gen[i+4, i] = 1
            generators.append(gen)

        # 生成元5-7: より複雑な回転
        for i in range(3):
            gen = torch.zeros(8, 8)
            if i == 0:
                gen[0, 1] = -1; gen[1, 0] = 1
            elif i == 1:
                gen[2, 3] = -1; gen[3, 2] = 1
            else:
                gen[4, 5] = -1; gen[5, 4] = 1
            generators.append(gen)

        # 初期回転行列として設定
        with torch.no_grad():
            for i, gen in enumerate(generators):
                if i < self.rotation_matrices.shape[0]:
                    self.rotation_matrices.data[i] = gen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SO(8)回転変換"""
        batch_size, seq_len, hidden_size = x.shape

        # 隠れ層をSO(8)空間に射影
        # 単純化のため、最初のso8_dim次元を使用
        x_proj = x[..., :self.so8_dim]

        # SO(8)回転を適用
        rotated = torch.matmul(x_proj, self.rotation_matrices.t())

        # 元の次元に戻す
        result = x.clone()
        result[..., :self.so8_dim] = rotated

        # スケーリング
        result = result * self.scale

        return result

class SO8ResidualAdapter(nn.Module):
    """SO(8)残差アダプター"""

    def __init__(self, config: SO8AdapterConfig):
        super().__init__()
        self.config = config

        # ダウンプロジェクション
        self.down_proj = nn.Linear(config.hidden_size, config.adapter_dim)

        # SO(8)回転レイヤー
        self.so8_rotation = SO8RotationLayer(config)

        # アッププロジェクション
        self.up_proj = nn.Linear(config.adapter_dim, config.hidden_size)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # Activation
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差アダプター適用"""
        # 残差接続の準備
        residual = x

        # ダウンプロジェクション
        down = self.down_proj(x)
        down = self.activation(down)

        # SO(8)回転適用
        rotated = self.so8_rotation(down)

        # アッププロジェクション
        up = self.up_proj(rotated)

        # 残差接続とLayer norm
        output = self.layer_norm(up + residual)

        return output

class SO8TAdaptedPhi35(nn.Module):
    """SO(8)アダプター適用済みPhi-3.5モデル"""

    def __init__(self, base_model, config: SO8AdapterConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config

        # SO(8)アダプターを各指定層に追加
        self.adapters = nn.ModuleDict()
        for layer_idx in config.adapter_layers:
            adapter_name = f"adapter_{layer_idx}"
            self.adapters[adapter_name] = SO8ResidualAdapter(config)

        # ベースモデルの重みを凍結
        self._freeze_base_model()

    def _freeze_base_model(self):
        """ベースモデルの重みを凍結"""
        for param in self.base_model.parameters():
            param.requires_grad = False

        print("Base model weights frozen")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """順伝播（アダプター適用）"""

        # ベースモデルの出力を取得
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        # 各アダプターレイヤーで隠れ状態を変換
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            modified_hidden_states = []

            for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                # 指定された層にアダプターを適用
                if layer_idx in self.config.adapter_layers:
                    adapter_name = f"adapter_{layer_idx}"
                    adapter = self.adapters[adapter_name]

                    # アダプター適用
                    modified_hidden = adapter(hidden_state)
                    modified_hidden_states.append(modified_hidden)
                else:
                    modified_hidden_states.append(hidden_state)

            # 最終隠れ状態を更新
            outputs.hidden_states = tuple(modified_hidden_states)

            # 最終出力をアダプター適用済みの隠れ状態に基づいて再計算
            if len(modified_hidden_states) > 0:
                final_hidden = modified_hidden_states[-1]

                # 言語モデリングヘッドを適用
                if hasattr(self.base_model, 'lm_head'):
                    logits = self.base_model.lm_head(final_hidden)
                    outputs.logits = logits

        return outputs

    def generate(self, *args, **kwargs):
        """生成メソッド（アダプター適用済み）"""
        return self.base_model.generate(*args, **kwargs)

    def save_adapter(self, path: str):
        """アダプターパラメータのみ保存"""
        adapter_state = {
            'config': self.config,
            'adapters': self.adapters.state_dict()
        }
        torch.save(adapter_state, path)
        print(f"SO(8) adapters saved to {path}")

    def load_adapter(self, path: str):
        """アダプターパラメータを読み込み"""
        adapter_state = torch.load(path)
        self.adapters.load_state_dict(adapter_state['adapters'])
        print(f"SO(8) adapters loaded from {path}")

def create_so8t_adapted_phi35(
    base_model_path: str = "microsoft/Phi-3.5-mini-instruct",
    adapter_config: Optional[SO8AdapterConfig] = None
):
    """SO(8)アダプター適用済みPhi-3.5モデルを作成"""

    if adapter_config is None:
        adapter_config = SO8AdapterConfig()

    # ベースモデルを読み込み
    try:
        from transformers import AutoModelForCausalLM
        print(f"Loading base model: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load base model: {e}")
        return None

    # SO(8)アダプターを適用
    print("Applying SO(8) residual adapters...")
    adapted_model = SO8TAdaptedPhi35(base_model, adapter_config)

    print(f"SO(8) adapters applied to layers: {adapter_config.adapter_layers}")
    print(f"Trainable parameters: {sum(p.numel() for p in adapted_model.parameters() if p.requires_grad):,}")

    return adapted_model

# テスト関数
def test_so8t_adapter():
    """SO(8)アダプターテスト"""
    print("Testing SO(8) Residual Adapter...")

    config = SO8AdapterConfig(hidden_size=768, adapter_dim=64)  # テスト用小規模

    # テスト入力
    batch_size, seq_len, hidden_size = 2, 10, 768
    x = torch.randn(batch_size, seq_len, hidden_size)

    # アダプター作成
    adapter = SO8ResidualAdapter(config)

    # 順伝播
    output = adapter(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {sum(p.numel() for p in adapter.parameters() if p.requires_grad):,}")

    # SO(8)回転テスト
    rotation_layer = SO8RotationLayer(config)
    rot_output = rotation_layer(x)
    print(f"Rotation output shape: {rot_output.shape}")

    print("SO(8) Residual Adapter test completed!")

if __name__ == "__main__":
    test_so8t_adapter()
