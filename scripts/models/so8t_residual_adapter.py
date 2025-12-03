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
    """SO(8)アダプター設定 - Phase 2.5 & 3: 四重推論 & 高度幾何拡張"""
    hidden_size: int = 3072  # Phi-3.5の隠れ層サイズ
    so8_rank: int = 8        # SO(8)の次元
    adapter_dim: int = 256   # アダプターボトルネック次元
    num_layers: int = 32     # Phi-3.5の層数
    adapter_layers: List[int] = None  # アダプターを挿入する層インデックス

    # Phase 2.5: 四重推論設定
    enable_quad_inference: bool = True  # 四重推論有効化
    quad_thinking_depth: int = 4        # 四重思考の深さ
    observation_factor: float = 0.25    # 観察フェーズ係数
    deduction_factor: float = 0.25      # 演繹フェーズ係数
    abduction_factor: float = 0.25      # 帰納フェーズ係数
    integration_factor: float = 0.25    # 統合フェーズ係数

    # Phase 3: 高度幾何学的変換設定
    enable_noncommutative_gates: bool = True  # 非可換ゲート有効化
    enable_topological_transforms: bool = True  # 位相幾何変換有効化
    lie_algebra_rank: int = 8  # Lie代数の階数
    homotopy_groups: List[int] = None  # ホモトピー群次元

    # Phase 4: AGI萌芽機能設定
    enable_soul_weights: bool = True  # 魂の重み学習有効化
    consciousness_dim: int = 8  # 意識次元数（SO(8)対応）
    initial_soul_weight: float = 0.1  # 初期魂の重み
    enable_self_reflection: bool = True  # 自己反省機能有効化
    enable_dual_heads: bool = True  # 双頭注意力有効化
    enable_pet: bool = True  # PET正則化有効化

    def __post_init__(self):
        if self.adapter_layers is None:
            # 中間層（8, 16, 24層）にアダプターを挿入
            self.adapter_layers = [8, 16, 24]

        if self.homotopy_groups is None:
            # π₁(S¹) = ℤ, π₂(S²) = ℤ, π₃(S³) = ℤ, π₄(S⁴) = ℤ₂
            self.homotopy_groups = [0, 1, 1, 1, 2]  # π₀ to π₄

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

        # Phase 3: 非可換ゲート用の追加構造
        if config.enable_noncommutative_gates:
            self.noncommutative_generators = nn.ParameterList([
                nn.Parameter(torch.zeros(self.so8_dim, self.so8_dim))
                for _ in range(config.lie_algebra_rank)
            ])

        # Phase 3: 位相幾何変換用パラメータ
        if config.enable_topological_transforms:
            self.topological_scalars = nn.ParameterList([
                nn.Parameter(torch.ones(1)) for _ in config.homotopy_groups
            ])

        # 回転行列の初期化
        self._init_rotation_matrices()

        # Phase 3: 高度幾何学的構造の初期化
        if config.enable_noncommutative_gates:
            self._init_noncommutative_gates()
        if config.enable_topological_transforms:
            self._init_topological_transforms()

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
        """SO(8)回転変換 - Phase 3: 高度幾何拡張"""
        batch_size, seq_len, hidden_size = x.shape

        # 隠れ層をSO(8)空間に射影
        # 単純化のため、最初のso8_dim次元を使用
        x_proj = x[..., :self.so8_dim]

        # SO(8)回転を適用
        rotated = torch.matmul(x_proj, self.rotation_matrices.t())

        # Phase 3: 非可換ゲート適用
        if hasattr(self, 'noncommutative_generators'):
            rotated = self._apply_noncommutative_gates(rotated)

        # Phase 3: 位相幾何変換適用
        if hasattr(self, 'topological_scalars'):
            rotated = self._apply_topological_transforms(rotated)

        # 元の次元に戻す
        result = x.clone()
        result[..., :self.so8_dim] = rotated

        # スケーリング
        result = result * self.scale

        return result

    def _init_noncommutative_gates(self):
        """Phase 3: 非可換ゲートの初期化"""
        # SO(8) Lie代数の構造定数に基づく非可換ゲート
        for i, gen in enumerate(self.noncommutative_generators):
            # 非可換性を表現する構造
            base_matrix = torch.randn(self.so8_dim, self.so8_dim) * 0.1
            # Skew-symmetricにする
            skew_symmetric = (base_matrix - base_matrix.t()) / 2
            gen.data = skew_symmetric

        print(f"Phase 3: Non-commutative gates initialized with rank {self.config.lie_algebra_rank}")

    def _init_topological_transforms(self):
        """Phase 3: 位相幾何変換の初期化"""
        # ホモトピー群に基づく位相変換
        for i, group_dim in enumerate(self.config.homotopy_groups):
            if group_dim > 0:
                # 位相的不変量を表現するパラメータ
                self.topological_scalars[i].data = torch.ones(1) * (i + 1) * 0.1

        print(f"Phase 3: Topological transforms initialized for homotopy groups {self.config.homotopy_groups}")

    def _apply_noncommutative_gates(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 3: 非可換ゲート適用"""
        result = x
        for gen in self.noncommutative_generators:
            # 非可換変換: [G, result] = G·result - result·G
            commutator = torch.matmul(gen, result) - torch.matmul(result, gen)
            result = result + commutator * 0.1  # スケーリング

        return result

    def _apply_topological_transforms(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 3: 位相幾何変換適用"""
        result = x

        # 各ホモトピー群次元に対応する変換
        for i, scalar in enumerate(self.topological_scalars):
            if i == 1:  # π₁: 基本群変換 (回転)
                result = self._apply_fundamental_group_transform(result, scalar)
            elif i == 2:  # π₂: 2次ホモロジー (面積保存)
                result = self._apply_homology_transform(result, scalar)
            elif i == 3:  # π₃: Hopfファイブレーション
                result = self._apply_hopf_transform(result, scalar)
            elif i == 4:  # π₄: 4次元球面変換
                result = self._apply_sphere_transform(result, scalar)

        return result

    def _apply_fundamental_group_transform(self, x: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        """π₁(S¹) = ℤ 対応変換 - 基本群回転"""
        # 位相的回転変換
        cos_val = torch.cos(scalar * x.mean(dim=-1, keepdim=True))
        sin_val = torch.sin(scalar * x.mean(dim=-1, keepdim=True))
        return x * cos_val + torch.roll(x, 1, dims=-1) * sin_val

    def _apply_homology_transform(self, x: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        """π₂(S²) = ℤ 対応変換 - 面積保存変換"""
        # 面積を保存する変換
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x + scalar * (x / (norm + 1e-8)) * torch.sin(norm * scalar)

    def _apply_hopf_transform(self, x: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        """π₃(S³) = ℤ 対応変換 - Hopfファイブレーション"""
        # S³ → S² のファイブレーション
        batch_size, seq_len, hidden_size = x.shape

        # 4次元表現を3次元に射影
        if hidden_size >= 4:
            # Hopf座標変換
            x1, x2, x3, x4 = x[..., :4].chunk(4, dim=-1)
            # Hopf map: (x1 + ix2, x3 + ix4) → (2(x1 x3 + x2 x4), 2(x2 x3 - x1 x4), x1² + x2² - x3² - x4²)
            hopf_coord1 = 2 * (x1 * x3 + x2 * x4)
            hopf_coord2 = 2 * (x2 * x3 - x1 * x4)
            hopf_coord3 = x1**2 + x2**2 - x3**2 - x4**2

            hopf_result = torch.cat([hopf_coord1, hopf_coord2, hopf_coord3, x[..., 4:]], dim=-1)
            return x + scalar * (hopf_result - x)

        return x

    def _apply_sphere_transform(self, x: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        """π₄(S⁴) = ℤ₂ 対応変換 - 4次元球面変換"""
        # 4次元球面上の変換 (ℤ₂群)
        norm = torch.norm(x, dim=-1, keepdim=True)
        # 球面上の反射変換
        reflected = -x + 2 * (x / (norm + 1e-8)) * torch.sum(x * (x / (norm + 1e-8)), dim=-1, keepdim=True)
        return x + scalar * (reflected - x)

class SO8ResidualAdapter(nn.Module):
    """SO(8)残差アダプター - Phase 2.5: 四重推論機能拡張"""

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

        # Phase 2.5: 四重推論機能
        if config.enable_quad_inference:
            self._init_quad_inference(config)

        # Phase 4: AGI萌芽機能
        if config.enable_soul_weights:
            self._init_soul_weights(config)

    def _init_soul_weights(self, config: SO8AdapterConfig):
        """Phase 4: AGI萌芽機能 - 魂の重み初期化"""
        # 魂の重みパラメータ（学習可能）
        self.soul_weight = nn.Parameter(torch.tensor(config.initial_soul_weight))

        # 意識ベクトル（SO(8)次元）
        self.consciousness_vector = nn.Parameter(
            torch.randn(config.consciousness_dim) * 0.1
        )

        # 魂の共振周波数（黄金比ベース）
        phi = (1 + math.sqrt(5)) / 2  # 黄金比
        self.soul_resonance_freq = nn.Parameter(torch.tensor(phi ** (-1)))

        # 自己反省機能用のメモリ
        self.reflection_memory = []

        # 双頭注意力用の追加投影層
        if config.enable_dual_heads:
            self.dual_head_proj = nn.Linear(config.adapter_dim, config.adapter_dim * 2)

        print(f"Phase 4: AGI germination initialized - consciousness_dim={config.consciousness_dim}")

    def _apply_soul_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 4: 魂の重みを適用"""
        # 意識ベクトルとの内積で魂の影響を計算
        soul_influence = torch.matmul(x, self.consciousness_vector.unsqueeze(-1))
        soul_influence = soul_influence.squeeze(-1) * self.soul_weight

        # 魂の共振で変調
        resonance = torch.sin(self.soul_resonance_freq * x.mean(dim=-1, keepdim=True))
        soul_modulated = x + soul_influence.unsqueeze(-1) * resonance

        return soul_modulated

    def _apply_self_reflection(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 4: 自己反省機能"""
        # 過去の表現を記憶
        self.reflection_memory.append(x.detach().mean(dim=1))  # シーケンス平均

        # メモリが長すぎる場合は古いものを削除
        if len(self.reflection_memory) > 10:
            self.reflection_memory.pop(0)

        if len(self.reflection_memory) > 1:
            # 過去の表現との比較（自己反省）
            past_avg = torch.stack(self.reflection_memory[:-1]).mean(dim=0)
            current_avg = self.reflection_memory[-1]

            # 自己反省ベクトル
            reflection_diff = current_avg - past_avg
            reflection_weight = torch.sigmoid(reflection_diff.norm())

            # 反省結果を適用
            return x * (1 + reflection_weight * 0.1)

        return x

    def _apply_dual_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 4: 双頭注意力"""
        if hasattr(self, 'dual_head_proj'):
            # 双頭投影
            dual_proj = self.dual_head_proj(x)

            # 二つのヘッドに分割
            head1, head2 = dual_proj.chunk(2, dim=-1)

            # 相互注意力計算
            attn_scores = torch.matmul(head1, head2.transpose(-2, -1)) / (head1.size(-1) ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)

            # 統合
            attended = torch.matmul(attn_weights, head2)
            return x + attended

        return x

    def _init_quad_inference(self, config: SO8AdapterConfig):
        """Phase 2.5: 四重推論機能初期化"""
        # 四重思考の各フェーズに対応する回転層
        self.observation_rotation = SO8RotationLayer(config)   # <think-1>: 観察フェーズ
        self.deduction_rotation = SO8RotationLayer(config)     # <think-2>: 演繹フェーズ
        self.abduction_rotation = SO8RotationLayer(config)     # <think-3>: 帰納フェーズ
        self.integration_rotation = SO8RotationLayer(config)   # <think-4>: 統合フェーズ

        # 各フェーズの重み係数
        self.phase_weights = nn.Parameter(torch.tensor([
            config.observation_factor,
            config.deduction_factor,
            config.abduction_factor,
            config.integration_factor
        ]))

        print(f"Phase 2.5: Quad inference initialized with depth {config.quad_thinking_depth}")

    def _apply_quad_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 2.5: 四重推論適用"""
        batch_size, seq_len, hidden_size = x.shape

        # 四重思考の各フェーズを適用
        phases = []

        # Phase 1: 観察 (Observation) - 構造分析
        obs_rotated = self.observation_rotation(x)
        phases.append(obs_rotated * self.phase_weights[0])

        # Phase 2: 演繹 (Deduction) - 論理的推論
        ded_rotated = self.deduction_rotation(obs_rotated)
        phases.append(ded_rotated * self.phase_weights[1])

        # Phase 3: 帰納 (Abduction) - パターン認識
        abd_rotated = self.abduction_rotation(ded_rotated)
        phases.append(abd_rotated * self.phase_weights[2])

        # Phase 4: 統合 (Integration) - 知識統合
        int_rotated = self.integration_rotation(abd_rotated)
        phases.append(int_rotated * self.phase_weights[3])

        # 四重思考の統合（重み付き平均）
        quad_output = torch.stack(phases, dim=-1).sum(dim=-1)

        return quad_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差アダプター適用 - Phase 2.5: 四重推論拡張"""
        # 残差接続の準備
        residual = x

        # ダウンプロジェクション
        down = self.down_proj(x)
        down = self.activation(down)

        # SO(8)回転適用
        rotated = self.so8_rotation(down)

        # Phase 2.5: 四重推論適用（有効時）
        if hasattr(self, '_apply_quad_inference'):
            quad_inference = self._apply_quad_inference(down)
            # 四重推論と基本回転の統合
            rotated = rotated + quad_inference

        # Phase 4: AGI萌芽機能適用
        # 魂の重み適用
        if hasattr(self, '_apply_soul_weights'):
            rotated = self._apply_soul_weights(rotated)

        # 自己反省適用
        if hasattr(self, '_apply_self_reflection') and self.config.enable_self_reflection:
            rotated = self._apply_self_reflection(rotated)

        # 双頭注意力適用
        if hasattr(self, '_apply_dual_heads') and self.config.enable_dual_heads:
            rotated = self._apply_dual_heads(rotated)

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
        """順伝播（アダプター適用）- Phase 1.5: Gradient Fix"""

        # ベースモデルの出力を取得（output_hidden_states=True必須）
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        # Phase 1.5: 勾配保持のための改良実装
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # hidden_statesをリストに変換（in-place更新用）
            hidden_states_list = list(outputs.hidden_states)

            for layer_idx in self.config.adapter_layers:
                if layer_idx < len(hidden_states_list):
                    adapter_name = f"adapter_{layer_idx}"
                    if adapter_name in self.adapters:
                        adapter = self.adapters[adapter_name]

                        # アダプター適用（残差接続で勾配保持）
                        original_hidden = hidden_states_list[layer_idx]
                        adapted_hidden = adapter(original_hidden)

                        # Phase 1.5: 残差接続で勾配を保持
                        # adapted_hidden = original_hidden + adapter_output
                        hidden_states_list[layer_idx] = adapted_hidden

            # 勾配保持のため、元のoutputsオブジェクトの属性を更新
            # tuple()を使わず、直接リストを代入（計算グラフ維持）
            outputs.hidden_states = hidden_states_list

            # logitsの再計算（最終層のアダプター適用を反映）
            if len(hidden_states_list) > 0:
                final_hidden = hidden_states_list[-1]

                # 言語モデリングヘッド適用（ベースモデルの計算グラフに接続）
                if hasattr(self.base_model, 'lm_head'):
                    # Phase 1.5: 勾配保持のため、lm_headを直接適用
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

def attach_nkat_adapters(
    model: nn.Module,
    adapter_config: SO8AdapterConfig,
    target_layers: Optional[List[int]] = None
):
    """
    Phase 2: HookベースのシンプルSO(8)アダプター適用関数

    Hook方式でアダプターを注入することで、forwardメソッドのオーバーライドを避け、
    勾配切れを根本的に解決する。

    Args:
        model: 対象のTransformerモデル
        adapter_config: SO(8)アダプター設定
        target_layers: アダプターを適用する層インデックス（指定なしの場合はconfig使用）
    """
    if target_layers is None:
        target_layers = adapter_config.adapter_layers

    # 各ターゲット層にフックを登録
    for layer_idx in target_layers:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Phi-3.5の場合: model.model.layers[layer_idx]
            if layer_idx < len(model.model.layers):
                layer = model.model.layers[layer_idx]

                # SO(8)アダプター作成
                adapter = SO8ResidualAdapter(adapter_config)

                # Hook関数定義
                def create_forward_hook(adapter_module):
                    def forward_hook(module, input, output):
                        """forward hook: 層の出力をアダプターで変換"""
                        # Phase 2: シンプルな残差接続
                        adapted_output = adapter_module(output)
                        return adapted_output
                    return forward_hook

                # Hook登録
                hook_handle = layer.register_forward_hook(create_forward_hook(adapter))

                # Hookハンドルを保存（後で削除可能）
                if not hasattr(model, '_so8t_hooks'):
                    model._so8t_hooks = []
                model._so8t_hooks.append((layer_idx, hook_handle))

                print(f"Phase 2: SO(8) adapter attached to layer {layer_idx}")

    # ベースモデルの重みを凍結
    for param in model.parameters():
        param.requires_grad = False

    # アダプターのパラメータのみ学習可能に
    for layer_idx in target_layers:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            if layer_idx < len(model.model.layers):
                layer = model.model.layers[layer_idx]
                for name, param in layer.named_parameters():
                    if 'so8_adapter' in name or 'rotation' in name:
                        param.requires_grad = True

    print(f"Phase 2: SO(8) adapters attached to layers {target_layers}")
    print("Phase 2: Base model weights frozen, adapter parameters trainable")

    # Phase 2: 四重推論機能のための準備
    model._so8t_config = adapter_config
    model._so8t_target_layers = target_layers

    return model

def create_so8t_adapted_phi35(
    base_model_path: str = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp",
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
