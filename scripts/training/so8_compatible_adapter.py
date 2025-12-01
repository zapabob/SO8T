#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8) Compatible LoRA Adapter
SO(8)回転残差アダプター - GGUF変換互換LoRA実装

このモジュールはSO(8)群のLie代数に基づく回転残差アダプターを実装し、
学習時は幾何学的制約下で動作し、保存時は標準LoRA形式に変換可能。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple
import numpy as np


class SO8CompatibleLoRA(nn.Module):
    """
    SO(8) Compatible LoRA Adapter

    学習時はSO(8)回転残差アダプターとして動作し、
    保存時は標準LoRA形式に変換可能なモジュール。
    """

    def __init__(self, hidden_size: int, rank: int = 8, alpha: float = 1.0):
        """
        SO(8) Compatible LoRA初期化

        Args:
            hidden_size: 隠れ層サイズ
            rank: アダプターランク (SO(8)なので8に固定)
            alpha: スケーリング係数
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.rank = rank  # SO(8)なので8に固定
        self.alpha = alpha

        # Lie代数パラメータ (8x8の歪対称行列)
        # SO(8)群の生成元として使用
        self.lie_algebra_param = nn.Parameter(
            torch.randn(rank, rank) * 0.01
        )

        # 標準LoRAパラメータ
        self.lora_A = nn.Parameter(torch.randn(rank, hidden_size) * 0.01)  # Down
        self.lora_B = nn.Parameter(torch.zeros(hidden_size, rank))         # Up

        # 回転行列のキャッシュ (学習時の高速化)
        self.register_buffer('rotation_matrix', torch.eye(rank))
        self._rotation_updated = False

    def _compute_rotation_matrix(self) -> torch.Tensor:
        """
        Lie代数パラメータからSO(8)回転行列を計算

        R = exp(A - A^T) where A is lie_algebra_param
        """
        # 歪対称行列を作成 (A - A^T)
        lie_algebra = self.lie_algebra_param - self.lie_algebra_param.transpose(-1, -2)

        # 行列指数関数で回転行列を生成
        rotation_matrix = torch.matrix_exp(lie_algebra)

        return rotation_matrix

    def _update_rotation_cache(self):
        """回転行列のキャッシュを更新"""
        if not self._rotation_updated:
            self.rotation_matrix.copy_(self._compute_rotation_matrix())
            self._rotation_updated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SO(8)回転残差アダプターのForward Pass

        y = x + α · Up(R_SO(8)(Down(x)))
        """
        # トレーニング時は動的に回転行列を計算
        if self.training:
            R = self._compute_rotation_matrix()
        else:
            # 推論時はキャッシュを使用
            self._update_rotation_cache()
            R = self.rotation_matrix

        # SO(8)回転残差計算
        # y = x + α · lora_B @ R @ lora_A @ x
        down_proj = F.linear(x, self.lora_A)  # Down(x)
        rotated = F.linear(down_proj, R)       # R(Down(x))
        up_proj = F.linear(rotated, self.lora_B)  # Up(R(Down(x)))

        return x + self.alpha * up_proj

    def merge_to_standard_lora(self) -> Dict[str, torch.Tensor]:
        """
        SO(8)アダプターを標準LoRA形式に変換

        学習完了後、このメソッドを呼び出してGGUF互換のLoRA重みに変換。

        Returns:
            標準LoRA形式のstate_dict
        """
        # 最終回転行列を計算
        R = self._compute_rotation_matrix()

        # SO(8)変換: W'_A = R @ W_A (Down)
        # SO(8)変換: W'_B = W_B (Up) - 変更なし
        standard_lora_A = torch.matmul(R, self.lora_A)  # R @ lora_A
        standard_lora_B = self.lora_B.clone()            # lora_Bはそのまま

        # 標準LoRA形式のstate_dict
        standard_state_dict = {
            'lora_A.weight': standard_lora_A,
            'lora_B.weight': standard_lora_B,
            'alpha': torch.tensor(self.alpha),
            'rank': torch.tensor(self.rank),
        }

        return standard_state_dict

    def get_memory_usage(self) -> Dict[str, int]:
        """メモリ使用量を取得"""
        return {
            'lie_algebra': self.lie_algebra_param.numel() * self.lie_algebra_param.element_size(),
            'lora_A': self.lora_A.numel() * self.lora_A.element_size(),
            'lora_B': self.lora_B.numel() * self.lora_B.element_size(),
            'rotation_cache': self.rotation_matrix.numel() * self.rotation_matrix.element_size(),
        }


def inject_so8_adapter_into_model(
    model: nn.Module,
    target_modules: list = None,
    rank: int = 8,
    alpha: float = 1.0
) -> Dict[str, SO8CompatibleLoRA]:
    """
    UnslothモデルにSO(8)アダプターを注入

    Args:
        model: 対象モデル
        target_modules: 注入対象のモジュール名リスト
        rank: アダプターランク
        alpha: スケーリング係数

    Returns:
        注入されたアダプターマッピング
    """
    if target_modules is None:
        # デフォルトでattentionとmlp層を対象
        target_modules = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',  # Attention
            'gate_proj', 'up_proj', 'down_proj'      # MLP
        ]

    injected_adapters = {}

    def replace_module(module: nn.Module, name: str):
        """再帰的にモジュールを置換"""
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Linear層をチェック
            if isinstance(child_module, nn.Linear):
                # 対象モジュールかチェック
                if any(target in full_name for target in target_modules):
                    # SO(8)アダプターを注入
                    adapter = SO8CompatibleLoRA(
                        hidden_size=child_module.out_features,
                        rank=rank,
                        alpha=alpha
                    )

                    # 元のLinear層を保持し、アダプターを追加
                    original_weight = child_module.weight.clone()
                    original_bias = child_module.bias.clone() if child_module.bias is not None else None

                    # アダプター適用後の出力を計算する新しいモジュール
                    class SO8AdaptedLinear(nn.Module):
                        def __init__(self, original_linear, adapter):
                            super().__init__()
                            self.original_linear = original_linear
                            self.adapter = adapter

                        def forward(self, x):
                            # 元の出力 + アダプター出力
                            original_out = self.original_linear(x)
                            adapter_out = self.adapter(x)
                            return original_out + adapter_out

                    # モジュールを置換
                    new_module = SO8AdaptedLinear(child_module, adapter)
                    setattr(module, child_name, new_module)

                    injected_adapters[full_name] = adapter
                    print(f"[SO8] Injected SO(8) adapter into {full_name}")

            # 子モジュールを再帰的に処理
            replace_module(child_module, full_name)

    replace_module(model, "")
    return injected_adapters


def extract_standard_lora_from_model(
    model: nn.Module,
    injected_adapters: Dict[str, SO8CompatibleLoRA]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    モデルから標準LoRA形式を抽出

    Args:
        model: アダプター注入済みモデル
        injected_adapters: 注入されたアダプターマッピング

    Returns:
        モジュールごとの標準LoRA state_dict
    """
    standard_loras = {}

    for module_name, adapter in injected_adapters.items():
        # SO(8)アダプターを標準LoRAに変換
        standard_state_dict = adapter.merge_to_standard_lora()
        standard_loras[module_name] = standard_state_dict

        print(f"[SO8] Converted {module_name} to standard LoRA format")

    return standard_loras


def save_as_standard_lora(
    model: nn.Module,
    injected_adapters: Dict[str, SO8CompatibleLoRA],
    output_path: str
):
    """
    モデルを標準LoRA形式で保存

    Args:
        model: アダプター注入済みモデル
        injected_adapters: 注入されたアダプターマッピング
        output_path: 保存先パス
    """
    import os
    from pathlib import Path

    # 出力ディレクトリ作成
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ベースモデルを保存
    print("[SO8] Saving base model...")
    model.save_pretrained(output_dir)

    # アダプターを標準LoRAに変換して保存
    standard_loras = extract_standard_lora_from_model(model, injected_adapters)

    # adapter_config.jsonを作成
    adapter_config = {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": None,
        "revision": None,
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": 8,
        "target_modules": list(standard_loras.keys()),
        "lora_alpha": 1.0,
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "use_rslora": False,
        "modules_to_save": [],
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {}
    }

    # adapter_config.jsonを保存
    import json
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    # adapter_model.binを保存
    adapter_state_dict = {}
    for module_name, lora_dict in standard_loras.items():
        # PEFT形式のキー名に変換
        peft_module_name = module_name.replace(".", ".lora_")
        adapter_state_dict[f"base_model.model.{peft_module_name}.lora_A.weight"] = lora_dict["lora_A.weight"]
        adapter_state_dict[f"base_model.model.{peft_module_name}.lora_B.weight"] = lora_dict["lora_B.weight"]

    torch.save(adapter_state_dict, output_dir / "adapter_model.safetensors")

    print(f"[SO8] Saved standard LoRA model to {output_path}")
    print(f"[SO8] Ready for llama.cpp GGUF conversion!")


# 使用例
if __name__ == "__main__":
    # テスト実行
    print("SO(8) Compatible LoRA Adapter Test")

    # アダプターインスタンス作成
    adapter = SO8CompatibleLoRA(hidden_size=3072, rank=8, alpha=1.0)

    # テスト入力
    x = torch.randn(1, 32, 3072)  # [batch, seq, hidden]

    # Forward pass
    with torch.no_grad():
        output = adapter(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")

    # 標準LoRA変換テスト
    standard_lora = adapter.merge_to_standard_lora()
    print(f"Standard LoRA keys: {list(standard_lora.keys())}")

    # メモリ使用量
    memory = adapter.get_memory_usage()
    total_memory = sum(memory.values())
    print(f"Total memory usage: {total_memory} bytes")

    print("SO(8) Compatible LoRA Adapter test completed!")
