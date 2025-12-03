#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8) Residual Adapter Implementation
SO(8)理論に基づく残差アダプター for Transformer中間層
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

@dataclass
class SO8Config:
    """SO(8)残差アダプター設定"""
    hidden_size: int = 3072  # Phi-3.5の隠れ層サイズ
    adapter_dim: int = 256   # アダプターディメンション
    num_layers: int = 32     # 適用する層数
    alpha_init: float = -0.5  # 初期アルファ値
    alpha_final: float = 0.382  # φ^(-2) ≈ 0.382
    annealing_steps: int = 10000  # アニーリングステップ数
    orthogonal_reg_weight: float = 0.1  # 直交正則化重み
    entropy_temp_init: float = 1.0  # エントロピー温度初期値
    entropy_temp_min: float = 0.1   # エントロピー温度最小値
    entropy_temp_max: float = 5.0   # エントロピー温度最大値

class SO8GammaMatrices:
    """SO(8)ガンマ行列生成"""

    @staticmethod
    def create_gamma_matrices() -> Dict[str, torch.Tensor]:
        """SO(8)のガンマ行列を生成"""
        # 8次元ガンマ行列の定義
        gamma = {}

        # 基本的なパウリ行列を拡張
        sigma1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        sigma2 = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.float32)
        sigma3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)

        # SO(8)用の拡張行列
        I4 = torch.eye(4, dtype=torch.complex64)
        gamma5 = torch.kron(sigma3, I4)  # γ5

        # γ1, γ2, γ3 (ベクトル表現)
        gamma['gamma1'] = torch.kron(sigma1, torch.eye(4, dtype=torch.complex64))
        gamma['gamma2'] = torch.kron(sigma2, torch.eye(4, dtype=torch.complex64))
        gamma['gamma3'] = torch.kron(sigma3, torch.eye(4, dtype=torch.complex64))

        # γ4 (時間的成分)
        gamma['gamma4'] = torch.kron(torch.eye(2, dtype=torch.complex64),
                                    torch.kron(sigma3, torch.eye(2, dtype=torch.complex64)))

        # γμν (残りの空間的成分)
        for i in range(5, 9):
            if i == 5:
                gamma[f'gamma{i}'] = gamma5
            else:
                # 追加のガンマ行列（簡易実装）
                gamma[f'gamma{i}'] = torch.randn(8, 8, dtype=torch.complex64)
                # 直交化
                U, _, Vh = torch.linalg.svd(gamma[f'gamma{i}'])
                gamma[f'gamma{i}'] = U @ Vh

        return gamma

class SO8ResidualAdapter(nn.Module):
    """SO(8)残差アダプター"""

    def __init__(self, config: SO8Config):
        super().__init__()
        self.config = config

        # SO(8)ガンマ行列
        self.gamma_matrices = SO8GammaMatrices.create_gamma_matrices()

        # アダプターパラメータ
        self.down_proj = nn.Linear(config.hidden_size, config.adapter_dim)
        self.up_proj = nn.Linear(config.adapter_dim, config.hidden_size)

        # SO(8)回転パラメータ
        self.rotation_params = nn.ParameterDict({
            f'omega_{i}': nn.Parameter(torch.randn(8, 8))
            for i in range(8)
        })

        # アルファゲート
        self.alpha_gate = nn.Parameter(torch.tensor(config.alpha_init))

        # 活性化関数
        self.activation = nn.GELU()

        # 直交正則化用のバッファ
        self.register_buffer('step_count', torch.tensor(0))

    def compute_orthogonal_error(self, weight: torch.Tensor) -> torch.Tensor:
        """直交誤差を計算"""
        # グラム行列
        gram = weight @ weight.T

        # 単位行列からの偏差
        identity = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
        error = torch.norm(gram - identity, p='fro') ** 2

        return error

    def apply_so8_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """SO(8)回転を適用"""
        batch_size, seq_len, hidden_size = x.shape

        # 隠れ層を8次元チャンクに分割
        if hidden_size % 8 != 0:
            # パディング
            pad_size = 8 - (hidden_size % 8)
            x_padded = F.pad(x, (0, pad_size))
        else:
            x_padded = x
            pad_size = 0

        # 8次元チャンクに分割
        chunks = x_padded.view(batch_size, seq_len, -1, 8)

        rotated_chunks = []
        for i, chunk in enumerate(chunks.split(1, dim=-2)):
            chunk = chunk.squeeze(-2)  # [batch, seq, 8]

            # SO(8)回転の適用
            rotated = chunk.clone()
            for j in range(8):
                omega = self.rotation_params[f'omega_{j}']
                # 簡易的な回転（実際のSO(8)回転は複雑）
                rotation_matrix = torch.matrix_exp(omega * self.alpha_gate)
                rotated = rotated @ rotation_matrix.T

            rotated_chunks.append(rotated)

        # 再結合
        rotated_tensor = torch.cat(rotated_chunks, dim=-1)

        # パディング除去
        if pad_size > 0:
            rotated_tensor = rotated_tensor[..., :hidden_size]

        return rotated_tensor

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """順伝播"""
        # 残差接続の準備
        residual = x

        # SO(8)回転適用
        rotated_x = self.apply_so8_rotation(x)

        # アダプター適用
        down = self.down_proj(rotated_x)
        down = self.activation(down)
        up = self.up_proj(down)

        # アルファゲート適用
        alpha = torch.sigmoid(self.alpha_gate)
        output = residual + alpha * up

        # 統計情報の収集
        stats = {
            'alpha_value': alpha.item(),
            'orthogonal_errors': {
                f'omega_{i}': self.compute_orthogonal_error(self.rotation_params[f'omega_{i}']).item()
                for i in range(8)
            },
            'adapter_norm': torch.norm(up).item(),
            'residual_norm': torch.norm(residual).item()
        }

        # ステップカウント更新
        self.step_count += 1

        return output, stats

    def anneal_alpha_gate(self, current_step: int):
        """アルファゲートのアニーリング"""
        if current_step >= self.config.annealing_steps:
            return

        # -0.5 から φ^(-2) ≈ 0.382 まで線形アニーリング
        progress = min(current_step / self.config.annealing_steps, 1.0)
        target_alpha = self.config.alpha_init + progress * (self.config.alpha_final - self.config.alpha_init)

        # ソフト更新
        current_alpha = self.alpha_gate.item()
        new_alpha = 0.9 * current_alpha + 0.1 * target_alpha

        self.alpha_gate.data = torch.tensor(new_alpha, device=self.alpha_gate.device)

    def adjust_entropy_temperature(self, entropy: float) -> float:
        """エントロピー制御による温度調整"""
        # 低エントロピー（確信度の高い推論）：加熱
        # 高エントロピー（不確実な推論）：冷却

        if entropy < 0.5:  # 低エントロピー
            # 加熱して探索を促進
            temperature = min(self.config.entropy_temp_max,
                            self.config.entropy_temp_init * (1 + (0.5 - entropy) * 2))
        else:  # 高エントロピー
            # 冷却して収束を促進
            temperature = max(self.config.entropy_temp_min,
                            self.config.entropy_temp_init * (1 - (entropy - 0.5) * 2))

        return temperature

class DynamicLayerSelector:
    """Transformer中間層の動的選択"""

    def __init__(self, num_layers: int = 32):
        self.num_layers = num_layers
        self.layer_scores = {}  # 層ごとの有効性スコア

    def select_layers(self, model_outputs: Dict[int, torch.Tensor],
                     target_distribution: torch.Tensor) -> List[int]:
        """有効性の高い層を選択"""
        layer_scores = {}

        for layer_idx, hidden_states in model_outputs.items():
            # KLダイバージェンスで層の有効性を評価
            with torch.no_grad():
                # 隠れ状態の分布を計算
                hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
                hidden_dist = torch.softmax(hidden_flat @ hidden_flat.T / hidden_states.shape[-1], dim=-1)

                # ターゲット分布とのKLダイバージェンス
                kl_div = F.kl_div(hidden_dist.log(), target_distribution.unsqueeze(0),
                                 reduction='batchmean')

                layer_scores[layer_idx] = -kl_div.item()  # 負のKL（低いほど良い）

        # 上位層を選択
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        selected_layers = [layer for layer, _ in sorted_layers[:8]]  # 上位8層

        return selected_layers

class SO8ThinkingModel(nn.Module):
    """SO(8) Thinkingモデル"""

    def __init__(self, base_model, config: SO8Config):
        super().__init__()
        self.base_model = base_model
        self.config = config

        # SO(8)アダプター
        self.so8_adapters = nn.ModuleDict({
            f'layer_{i}': SO8ResidualAdapter(config)
            for i in range(config.num_layers)
        })

        # 層選択器
        self.layer_selector = DynamicLayerSelector(config.num_layers)

        # /thinking 出力フォーマッタ
        self.thinking_formatter = ThinkingFormatter()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, Any]:

        # ベースモデルの出力を層ごとに収集
        layer_outputs = {}
        hook_handles = []

        def hook_fn(module, input, output, layer_idx):
            layer_outputs[layer_idx] = output[0] if isinstance(output, tuple) else output

        # 各層にフックを設定
        for i, layer in enumerate(self.base_model.model.layers):
            handle = layer.register_forward_hook(
                lambda mod, inp, out, idx=i: hook_fn(mod, inp, out, idx)
            )
            hook_handles.append(handle)

        try:
            # ベースモデル実行
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                **kwargs
            )

            # 動的層選択
            target_dist = torch.softmax(base_outputs.logits.mean(dim=1), dim=-1)
            selected_layers = self.layer_selector.select_layers(layer_outputs, target_dist)

            # SO(8)アダプター適用
            adapter_outputs = []
            total_stats = {}

            for layer_idx in selected_layers:
                if layer_idx in layer_outputs:
                    adapter_input = layer_outputs[layer_idx]
                    adapter_output, stats = self.so8_adapters[f'layer_{layer_idx}'](adapter_input)

                    adapter_outputs.append(adapter_output)
                    total_stats.update({f'layer_{layer_idx}_{k}': v for k, v in stats.items()})

            # アダプター出力を統合
            if adapter_outputs:
                combined_output = torch.stack(adapter_outputs, dim=0).mean(dim=0)
            else:
                combined_output = base_outputs.hidden_states[-1]

            # /thinking フォーマット適用
            thinking_output = self.thinking_formatter.format_output(
                combined_output, base_outputs.logits
            )

            # 最終出力を更新
            final_logits = self.base_model.lm_head(combined_output)

            return {
                'logits': final_logits,
                'thinking_output': thinking_output,
                'selected_layers': selected_layers,
                'so8_stats': total_stats,
                **base_outputs
            }

        finally:
            # フック解除
            for handle in hook_handles:
                handle.remove()

class ThinkingFormatter:
    """Thinking出力フォーマッタ"""

    def __init__(self):
        self.phi_ratio = (1 + math.sqrt(5)) / 2  # 黄金比

    def format_output(self, hidden_states: torch.Tensor,
                     logits: torch.Tensor) -> str:
        """隠れ状態から/thinking形式の出力を生成"""

        # エントロピー計算
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()

        # 思考の深さを決定
        thinking_depth = min(4, max(1, int(entropy.item() * 2)))

        thinking_parts = []

        # 観察部
        thinking_parts.append("<|observation|>")
        thinking_parts.append("量子幾何学的分析に基づく推論を開始")
        thinking_parts.append("<|end_observation|>")

        # 演繹部
        thinking_parts.append("<|deduction|>")
        thinking_parts.append(f"SO(8)多重推論: エントロピー {entropy:.3f}")
        thinking_parts.append("<|end_deduction|>")

        # 帰納部
        thinking_parts.append("<|abduction|>")
        thinking_parts.append(f"黄金比調和: φ = {self.phi_ratio:.3f}")
        thinking_parts.append("<|end_abduction|>")

        # 統合部
        thinking_parts.append("<|integration|>")
        thinking_parts.append("SO(8)幾何学的統合完了")
        thinking_parts.append("<|end_integration|>")

        thinking_text = "\n".join(thinking_parts)

        return f"<think>\n{thinking_text}\n</think>\n\n<final>\n[SO(8)推論完了]\n</final>"

# ユーティリティ関数
def create_so8_adapter_config(hidden_size: int = 3072) -> SO8Config:
    """SO(8)アダプター設定を作成"""
    return SO8Config(
        hidden_size=hidden_size,
        adapter_dim=max(256, hidden_size // 12),  # 適度なアダプターサイズ
        num_layers=32,  # Phi-3.5の層数
        alpha_init=-0.5,
        alpha_final=(1 + math.sqrt(5)) / 2 * 0.618,  # φ^(-2)
        annealing_steps=10000,
        orthogonal_reg_weight=0.1,
        entropy_temp_init=1.0,
        entropy_temp_min=0.1,
        entropy_temp_max=5.0
    )

if __name__ == "__main__":
    # テスト実行
    config = create_so8_adapter_config()
    print(f"SO(8)アダプター設定: {config}")

    # アダプターインスタンス作成テスト
    adapter = SO8ResidualAdapter(config)
    print(f"アダプター作成成功: {adapter}")

    # ダミーデータでテスト
    dummy_input = torch.randn(1, 10, config.hidden_size)
    output, stats = adapter(dummy_input)
    print(f"出力形状: {output.shape}")
    print(f"統計情報: {list(stats.keys())}")
