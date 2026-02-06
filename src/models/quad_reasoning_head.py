#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quadruple Reasoning Head - Phase 2.5 準備

Observation / Deduction / Abduction / Integration
の4つのロールにhiddenを分解するヘッド。

Phase 2.5 (Quadruple Inference Integration) に向けて、
SO(8)回転エンジンの上で動く「思考の4ロール分解」器を準備。
"""

import torch
from torch import nn
from typing import Dict, Optional


class QuadReasoningHead(nn.Module):
    """
    NKAT理論に基づく4重推論ヘッド

    Observation / Deduction / Abduction / Integration
    の4つのロールにhidden_statesを分解する。

    特徴:
        - Phase 2.5 向けの思考ロール分離
        - SO(8)幾何変換との統合前提
        - Hookベース注入との互換性
        - 学習初期は線形分離のみ（安全策）
    """

    def __init__(self, hidden_dim: int, quad_method: str = "linear"):
        """
        Args:
            hidden_dim: 隠れ層次元数
            quad_method: 分解方法 ("linear", "so8_geometric", "topological")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.quad_method = quad_method

        if quad_method == "linear":
            # 線形分離（Phase 2.5 初期）
            self.proj = nn.Linear(hidden_dim, 4 * hidden_dim)
            self.act = nn.Tanh()  # ソフトな分離

        elif quad_method == "so8_geometric":
            # SO(8)幾何変換（Phase 3）
            # ここにSO(8)回転行列を使った分離を実装
            raise NotImplementedError("SO(8) geometric method not implemented yet")

        elif quad_method == "topological":
            # 位相幾何変換（Phase 3 advanced）
            # ホモトピー群やファイバー束理論ベース
            raise NotImplementedError("Topological method not implemented yet")

        else:
            raise ValueError(f"Unknown quad_method: {quad_method}")

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        隠れ状態を4つの思考ロールに分解

        Args:
            hidden: (batch, seq, hidden_dim)

        Returns:
            {
              "observation": (batch, seq, hidden_dim),
              "deduction":   (batch, seq, hidden_dim),
              "abduction":   (batch, seq, hidden_dim),
              "integration": (batch, seq, hidden_dim),
            }
        """
        if self.quad_method == "linear":
            # 線形射影 + Tanh活性化
            x = self.proj(hidden)  # (B, S, 4*H)
            x = self.act(x)        # ソフトな分離
            obs, ded, abd, integ = torch.chunk(x, 4, dim=-1)

        else:
            raise NotImplementedError(f"Method {self.quad_method} not implemented")

        return {
            "observation": obs,    # 観測・知覚
            "deduction":   ded,    # 演繹・論理推論
            "abduction":   abd,    # 仮説形成・帰納
            "integration": integ,  # 統合・合成
        }

    def get_quad_similarity(self) -> Dict[str, float]:
        """
        4ロール間の類似度を監視
        Phase 2.5 デバッグ用
        """
        if not hasattr(self, 'proj'):
            return {}

        W = self.proj.weight  # (4*H, H)
        quad_dim = self.hidden_dim

        similarities = {}
        for i, role1 in enumerate(['obs', 'ded', 'abd', 'integ']):
            for j, role2 in enumerate(['obs', 'ded', 'abd', 'integ']):
                if i < j:
                    w1 = W[i*quad_dim:(i+1)*quad_dim]  # (H, H)
                    w2 = W[j*quad_dim:(j+1)*quad_dim]  # (H, H)

                    # コサイン類似度
                    cos_sim = torch.nn.functional.cosine_similarity(
                        w1.flatten(), w2.flatten(), dim=0
                    ).item()

                    similarities[f'{role1}_{role2}'] = cos_sim

        return similarities


def attach_quad_reasoning_head(model, target_layers: Optional[list] = None):
    """
    モデルにQuadReasoningHeadを注入

    Phase 2.5 統合時に使用予定。
    現在はまだ学習損失に組み込まず、推論時のみ使用。

    Args:
        model: 対象モデル
        target_layers: 注入対象層（None=全層）
    """
    print("🧠 Attaching Quad Reasoning Head (Phase 2.5 preparation)...")

    # モデル構造解析
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        # LoRA適用後モデル
        layers = model.base_model.model.base_model.layers
        hidden_size = model.base_model.model.config.hidden_size
    elif hasattr(model, "base_model"):
        # 通常のPhi-3モデル
        layers = model.base_model.layers
        hidden_size = model.config.hidden_size
    else:
        raise ValueError("Unknown model structure")

    num_layers = len(layers)

    # ターゲット層決定
    if target_layers is None:
        target_layers = list(range(num_layers))  # 全層

    injected_count = 0
    for i in target_layers:
        if i >= num_layers:
            continue

        layer = layers[i]

        # QuadReasoningHead注入
        if not hasattr(layer, 'quad_reasoning_head'):
            quad_head = QuadReasoningHead(hidden_size)
            layer.add_module('quad_reasoning_head', quad_head)
            injected_count += 1

    print(f"[OK] Injected Quad Reasoning Heads into {injected_count} layers")

    # 推論専用なので勾配は切っておく（Phase 2.5 統合まで）
    for name, param in model.named_parameters():
        if 'quad_reasoning_head' in name:
            param.requires_grad = False

    return model


def extract_quad_reasoning(hidden_states: torch.Tensor,
                          quad_head: QuadReasoningHead) -> Dict[str, torch.Tensor]:
    """
    隠れ状態から4つの思考ロールを抽出

    Phase 2.5 の/think APIで使用予定

    Args:
        hidden_states: (batch, seq, hidden)
        quad_head: QuadReasoningHeadインスタンス

    Returns:
        4ロールの思考辞書
    """
    return quad_head(hidden_states)


# Phase 2.5 統合用のユーティリティ関数群
def create_quad_thinking_prompt(base_prompt: str) -> str:
    """
    Quadruple Inference 用のプロンプト生成

    Phase 2.5 で使用予定
    """
    quad_prompt = f"""
<think-observation>
観測フェーズ：{base_prompt}
現在の状況を観測し、利用可能な情報を整理する。
</think-observation>

<think-deduction>
演繹フェーズ：論理的推論に基づいて結論を導く。
</think-deduction>

<think-abduction>
仮説形成フェーズ：可能な説明や仮説を生成する。
</think-abduction>

<think-integration>
統合フェーズ：全ての思考を統合し、最終回答を生成する。
</think-integration>

<final>
"""

    return quad_prompt


def format_quad_response(quad_outputs: Dict[str, torch.Tensor],
                        tokenizer) -> str:
    """
    Quadruple Inference の出力を整形

    Phase 2.5 の/think APIレスポンス用
    """
    # 各ロールの出力をデコード（簡易実装）
    response_parts = []

    for role, hidden in quad_outputs.items():
        # 最後のトークンだけデコード（簡易版）
        last_token = hidden[:, -1:, :]  # (B, 1, H)
        # ここではダミーとしてロール名だけ返す
        response_parts.append(f"<think-{role}>[{role} thinking output]</think-{role}>")

    final_response = "\n".join(response_parts) + "\n<final>[final answer]</final>"

    return final_response


if __name__ == "__main__":
    # テスト実行
    print("🧠 Quad Reasoning Head Test")

    # ダミーデータ
    batch_size, seq_len, hidden_dim = 2, 10, 1024
    hidden = torch.randn(batch_size, seq_len, hidden_dim)

    # QuadReasoningHeadテスト
    quad_head = QuadReasoningHead(hidden_dim)

    quad_output = quad_head(hidden)
    print(f"Input shape: {hidden.shape}")
    print(f"Output keys: {list(quad_output.keys())}")
    for role, tensor in quad_output.items():
        print(f"  {role}: {tensor.shape}")

    # 類似度チェック
    similarities = quad_head.get_quad_similarity()
    print(f"Role similarities: {similarities}")

    print("[OK] Quad Reasoning Head test passed!")
# -*- coding: utf-8 -*-
"""
Quadruple Reasoning Head - Phase 2.5 準備

Observation / Deduction / Abduction / Integration
の4つのロールにhiddenを分解するヘッド。

Phase 2.5 (Quadruple Inference Integration) に向けて、
SO(8)回転エンジンの上で動く「思考の4ロール分解」器を準備。
"""

import torch
from torch import nn
from typing import Dict, Optional


class QuadReasoningHead(nn.Module):
    """
    NKAT理論に基づく4重推論ヘッド

    Observation / Deduction / Abduction / Integration
    の4つのロールにhidden_statesを分解する。

    特徴:
        - Phase 2.5 向けの思考ロール分離
        - SO(8)幾何変換との統合前提
        - Hookベース注入との互換性
        - 学習初期は線形分離のみ（安全策）
    """

    def __init__(self, hidden_dim: int, quad_method: str = "linear"):
        """
        Args:
            hidden_dim: 隠れ層次元数
            quad_method: 分解方法 ("linear", "so8_geometric", "topological")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.quad_method = quad_method

        if quad_method == "linear":
            # 線形分離（Phase 2.5 初期）
            self.proj = nn.Linear(hidden_dim, 4 * hidden_dim)
            self.act = nn.Tanh()  # ソフトな分離

        elif quad_method == "so8_geometric":
            # SO(8)幾何変換（Phase 3）
            # ここにSO(8)回転行列を使った分離を実装
            raise NotImplementedError("SO(8) geometric method not implemented yet")

        elif quad_method == "topological":
            # 位相幾何変換（Phase 3 advanced）
            # ホモトピー群やファイバー束理論ベース
            raise NotImplementedError("Topological method not implemented yet")

        else:
            raise ValueError(f"Unknown quad_method: {quad_method}")

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        隠れ状態を4つの思考ロールに分解

        Args:
            hidden: (batch, seq, hidden_dim)

        Returns:
            {
              "observation": (batch, seq, hidden_dim),
              "deduction":   (batch, seq, hidden_dim),
              "abduction":   (batch, seq, hidden_dim),
              "integration": (batch, seq, hidden_dim),
            }
        """
        if self.quad_method == "linear":
            # 線形射影 + Tanh活性化
            x = self.proj(hidden)  # (B, S, 4*H)
            x = self.act(x)        # ソフトな分離
            obs, ded, abd, integ = torch.chunk(x, 4, dim=-1)

        else:
            raise NotImplementedError(f"Method {self.quad_method} not implemented")

        return {
            "observation": obs,    # 観測・知覚
            "deduction":   ded,    # 演繹・論理推論
            "abduction":   abd,    # 仮説形成・帰納
            "integration": integ,  # 統合・合成
        }

    def get_quad_similarity(self) -> Dict[str, float]:
        """
        4ロール間の類似度を監視
        Phase 2.5 デバッグ用
        """
        if not hasattr(self, 'proj'):
            return {}

        W = self.proj.weight  # (4*H, H)
        quad_dim = self.hidden_dim

        similarities = {}
        for i, role1 in enumerate(['obs', 'ded', 'abd', 'integ']):
            for j, role2 in enumerate(['obs', 'ded', 'abd', 'integ']):
                if i < j:
                    w1 = W[i*quad_dim:(i+1)*quad_dim]  # (H, H)
                    w2 = W[j*quad_dim:(j+1)*quad_dim]  # (H, H)

                    # コサイン類似度
                    cos_sim = torch.nn.functional.cosine_similarity(
                        w1.flatten(), w2.flatten(), dim=0
                    ).item()

                    similarities[f'{role1}_{role2}'] = cos_sim

        return similarities


def attach_quad_reasoning_head(model, target_layers: Optional[list] = None):
    """
    モデルにQuadReasoningHeadを注入

    Phase 2.5 統合時に使用予定。
    現在はまだ学習損失に組み込まず、推論時のみ使用。

    Args:
        model: 対象モデル
        target_layers: 注入対象層（None=全層）
    """
    print("🧠 Attaching Quad Reasoning Head (Phase 2.5 preparation)...")

    # モデル構造解析
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        # LoRA適用後モデル
        layers = model.base_model.model.base_model.layers
        hidden_size = model.base_model.model.config.hidden_size
    elif hasattr(model, "base_model"):
        # 通常のPhi-3モデル
        layers = model.base_model.layers
        hidden_size = model.config.hidden_size
    else:
        raise ValueError("Unknown model structure")

    num_layers = len(layers)

    # ターゲット層決定
    if target_layers is None:
        target_layers = list(range(num_layers))  # 全層

    injected_count = 0
    for i in target_layers:
        if i >= num_layers:
            continue

        layer = layers[i]

        # QuadReasoningHead注入
        if not hasattr(layer, 'quad_reasoning_head'):
            quad_head = QuadReasoningHead(hidden_size)
            layer.add_module('quad_reasoning_head', quad_head)
            injected_count += 1

    print(f"[OK] Injected Quad Reasoning Heads into {injected_count} layers")

    # 推論専用なので勾配は切っておく（Phase 2.5 統合まで）
    for name, param in model.named_parameters():
        if 'quad_reasoning_head' in name:
            param.requires_grad = False

    return model


def extract_quad_reasoning(hidden_states: torch.Tensor,
                          quad_head: QuadReasoningHead) -> Dict[str, torch.Tensor]:
    """
    隠れ状態から4つの思考ロールを抽出

    Phase 2.5 の/think APIで使用予定

    Args:
        hidden_states: (batch, seq, hidden)
        quad_head: QuadReasoningHeadインスタンス

    Returns:
        4ロールの思考辞書
    """
    return quad_head(hidden_states)


# Phase 2.5 統合用のユーティリティ関数群
def create_quad_thinking_prompt(base_prompt: str) -> str:
    """
    Quadruple Inference 用のプロンプト生成

    Phase 2.5 で使用予定
    """
    quad_prompt = f"""
<think-observation>
観測フェーズ：{base_prompt}
現在の状況を観測し、利用可能な情報を整理する。
</think-observation>

<think-deduction>
演繹フェーズ：論理的推論に基づいて結論を導く。
</think-deduction>

<think-abduction>
仮説形成フェーズ：可能な説明や仮説を生成する。
</think-abduction>

<think-integration>
統合フェーズ：全ての思考を統合し、最終回答を生成する。
</think-integration>

<final>
"""

    return quad_prompt


def format_quad_response(quad_outputs: Dict[str, torch.Tensor],
                        tokenizer) -> str:
    """
    Quadruple Inference の出力を整形

    Phase 2.5 の/think APIレスポンス用
    """
    # 各ロールの出力をデコード（簡易実装）
    response_parts = []

    for role, hidden in quad_outputs.items():
        # 最後のトークンだけデコード（簡易版）
        last_token = hidden[:, -1:, :]  # (B, 1, H)
        # ここではダミーとしてロール名だけ返す
        response_parts.append(f"<think-{role}>[{role} thinking output]</think-{role}>")

    final_response = "\n".join(response_parts) + "\n<final>[final answer]</final>"

    return final_response


if __name__ == "__main__":
    # テスト実行
    print("🧠 Quad Reasoning Head Test")

    # ダミーデータ
    batch_size, seq_len, hidden_dim = 2, 10, 1024
    hidden = torch.randn(batch_size, seq_len, hidden_dim)

    # QuadReasoningHeadテスト
    quad_head = QuadReasoningHead(hidden_dim)

    quad_output = quad_head(hidden)
    print(f"Input shape: {hidden.shape}")
    print(f"Output keys: {list(quad_output.keys())}")
    for role, tensor in quad_output.items():
        print(f"  {role}: {tensor.shape}")

    # 類似度チェック
    similarities = quad_head.get_quad_similarity()
    print(f"Role similarities: {similarities}")

    print("[OK] Quad Reasoning Head test passed!")
# -*- coding: utf-8 -*-
"""
Quadruple Reasoning Head - Phase 2.5 準備

Observation / Deduction / Abduction / Integration
の4つのロールにhiddenを分解するヘッド。

Phase 2.5 (Quadruple Inference Integration) に向けて、
SO(8)回転エンジンの上で動く「思考の4ロール分解」器を準備。
"""

import torch
from torch import nn
from typing import Dict, Optional


class QuadReasoningHead(nn.Module):
    """
    NKAT理論に基づく4重推論ヘッド

    Observation / Deduction / Abduction / Integration
    の4つのロールにhidden_statesを分解する。

    特徴:
        - Phase 2.5 向けの思考ロール分離
        - SO(8)幾何変換との統合前提
        - Hookベース注入との互換性
        - 学習初期は線形分離のみ（安全策）
    """

    def __init__(self, hidden_dim: int, quad_method: str = "linear"):
        """
        Args:
            hidden_dim: 隠れ層次元数
            quad_method: 分解方法 ("linear", "so8_geometric", "topological")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.quad_method = quad_method

        if quad_method == "linear":
            # 線形分離（Phase 2.5 初期）
            self.proj = nn.Linear(hidden_dim, 4 * hidden_dim)
            self.act = nn.Tanh()  # ソフトな分離

        elif quad_method == "so8_geometric":
            # SO(8)幾何変換（Phase 3）
            # ここにSO(8)回転行列を使った分離を実装
            raise NotImplementedError("SO(8) geometric method not implemented yet")

        elif quad_method == "topological":
            # 位相幾何変換（Phase 3 advanced）
            # ホモトピー群やファイバー束理論ベース
            raise NotImplementedError("Topological method not implemented yet")

        else:
            raise ValueError(f"Unknown quad_method: {quad_method}")

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        隠れ状態を4つの思考ロールに分解

        Args:
            hidden: (batch, seq, hidden_dim)

        Returns:
            {
              "observation": (batch, seq, hidden_dim),
              "deduction":   (batch, seq, hidden_dim),
              "abduction":   (batch, seq, hidden_dim),
              "integration": (batch, seq, hidden_dim),
            }
        """
        if self.quad_method == "linear":
            # 線形射影 + Tanh活性化
            x = self.proj(hidden)  # (B, S, 4*H)
            x = self.act(x)        # ソフトな分離
            obs, ded, abd, integ = torch.chunk(x, 4, dim=-1)

        else:
            raise NotImplementedError(f"Method {self.quad_method} not implemented")

        return {
            "observation": obs,    # 観測・知覚
            "deduction":   ded,    # 演繹・論理推論
            "abduction":   abd,    # 仮説形成・帰納
            "integration": integ,  # 統合・合成
        }

    def get_quad_similarity(self) -> Dict[str, float]:
        """
        4ロール間の類似度を監視
        Phase 2.5 デバッグ用
        """
        if not hasattr(self, 'proj'):
            return {}

        W = self.proj.weight  # (4*H, H)
        quad_dim = self.hidden_dim

        similarities = {}
        for i, role1 in enumerate(['obs', 'ded', 'abd', 'integ']):
            for j, role2 in enumerate(['obs', 'ded', 'abd', 'integ']):
                if i < j:
                    w1 = W[i*quad_dim:(i+1)*quad_dim]  # (H, H)
                    w2 = W[j*quad_dim:(j+1)*quad_dim]  # (H, H)

                    # コサイン類似度
                    cos_sim = torch.nn.functional.cosine_similarity(
                        w1.flatten(), w2.flatten(), dim=0
                    ).item()

                    similarities[f'{role1}_{role2}'] = cos_sim

        return similarities


def attach_quad_reasoning_head(model, target_layers: Optional[list] = None):
    """
    モデルにQuadReasoningHeadを注入

    Phase 2.5 統合時に使用予定。
    現在はまだ学習損失に組み込まず、推論時のみ使用。

    Args:
        model: 対象モデル
        target_layers: 注入対象層（None=全層）
    """
    print("🧠 Attaching Quad Reasoning Head (Phase 2.5 preparation)...")

    # モデル構造解析
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        # LoRA適用後モデル
        layers = model.base_model.model.base_model.layers
        hidden_size = model.base_model.model.config.hidden_size
    elif hasattr(model, "base_model"):
        # 通常のPhi-3モデル
        layers = model.base_model.layers
        hidden_size = model.config.hidden_size
    else:
        raise ValueError("Unknown model structure")

    num_layers = len(layers)

    # ターゲット層決定
    if target_layers is None:
        target_layers = list(range(num_layers))  # 全層

    injected_count = 0
    for i in target_layers:
        if i >= num_layers:
            continue

        layer = layers[i]

        # QuadReasoningHead注入
        if not hasattr(layer, 'quad_reasoning_head'):
            quad_head = QuadReasoningHead(hidden_size)
            layer.add_module('quad_reasoning_head', quad_head)
            injected_count += 1

    print(f"[OK] Injected Quad Reasoning Heads into {injected_count} layers")

    # 推論専用なので勾配は切っておく（Phase 2.5 統合まで）
    for name, param in model.named_parameters():
        if 'quad_reasoning_head' in name:
            param.requires_grad = False

    return model


def extract_quad_reasoning(hidden_states: torch.Tensor,
                          quad_head: QuadReasoningHead) -> Dict[str, torch.Tensor]:
    """
    隠れ状態から4つの思考ロールを抽出

    Phase 2.5 の/think APIで使用予定

    Args:
        hidden_states: (batch, seq, hidden)
        quad_head: QuadReasoningHeadインスタンス

    Returns:
        4ロールの思考辞書
    """
    return quad_head(hidden_states)


# Phase 2.5 統合用のユーティリティ関数群
def create_quad_thinking_prompt(base_prompt: str) -> str:
    """
    Quadruple Inference 用のプロンプト生成

    Phase 2.5 で使用予定
    """
    quad_prompt = f"""
<think-observation>
観測フェーズ：{base_prompt}
現在の状況を観測し、利用可能な情報を整理する。
</think-observation>

<think-deduction>
演繹フェーズ：論理的推論に基づいて結論を導く。
</think-deduction>

<think-abduction>
仮説形成フェーズ：可能な説明や仮説を生成する。
</think-abduction>

<think-integration>
統合フェーズ：全ての思考を統合し、最終回答を生成する。
</think-integration>

<final>
"""

    return quad_prompt


def format_quad_response(quad_outputs: Dict[str, torch.Tensor],
                        tokenizer) -> str:
    """
    Quadruple Inference の出力を整形

    Phase 2.5 の/think APIレスポンス用
    """
    # 各ロールの出力をデコード（簡易実装）
    response_parts = []

    for role, hidden in quad_outputs.items():
        # 最後のトークンだけデコード（簡易版）
        last_token = hidden[:, -1:, :]  # (B, 1, H)
        # ここではダミーとしてロール名だけ返す
        response_parts.append(f"<think-{role}>[{role} thinking output]</think-{role}>")

    final_response = "\n".join(response_parts) + "\n<final>[final answer]</final>"

    return final_response


if __name__ == "__main__":
    # テスト実行
    print("🧠 Quad Reasoning Head Test")

    # ダミーデータ
    batch_size, seq_len, hidden_dim = 2, 10, 1024
    hidden = torch.randn(batch_size, seq_len, hidden_dim)

    # QuadReasoningHeadテスト
    quad_head = QuadReasoningHead(hidden_dim)

    quad_output = quad_head(hidden)
    print(f"Input shape: {hidden.shape}")
    print(f"Output keys: {list(quad_output.keys())}")
    for role, tensor in quad_output.items():
        print(f"  {role}: {tensor.shape}")

    # 類似度チェック
    similarities = quad_head.get_quad_similarity()
    print(f"Role similarities: {similarities}")

    print("[OK] Quad Reasoning Head test passed!")
# -*- coding: utf-8 -*-
"""
Quadruple Reasoning Head - Phase 2.5 準備

Observation / Deduction / Abduction / Integration
の4つのロールにhiddenを分解するヘッド。

Phase 2.5 (Quadruple Inference Integration) に向けて、
SO(8)回転エンジンの上で動く「思考の4ロール分解」器を準備。
"""

import torch
from torch import nn
from typing import Dict, Optional


class QuadReasoningHead(nn.Module):
    """
    NKAT理論に基づく4重推論ヘッド

    Observation / Deduction / Abduction / Integration
    の4つのロールにhidden_statesを分解する。

    特徴:
        - Phase 2.5 向けの思考ロール分離
        - SO(8)幾何変換との統合前提
        - Hookベース注入との互換性
        - 学習初期は線形分離のみ（安全策）
    """

    def __init__(self, hidden_dim: int, quad_method: str = "linear"):
        """
        Args:
            hidden_dim: 隠れ層次元数
            quad_method: 分解方法 ("linear", "so8_geometric", "topological")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.quad_method = quad_method

        if quad_method == "linear":
            # 線形分離（Phase 2.5 初期）
            self.proj = nn.Linear(hidden_dim, 4 * hidden_dim)
            self.act = nn.Tanh()  # ソフトな分離

        elif quad_method == "so8_geometric":
            # SO(8)幾何変換（Phase 3）
            # ここにSO(8)回転行列を使った分離を実装
            raise NotImplementedError("SO(8) geometric method not implemented yet")

        elif quad_method == "topological":
            # 位相幾何変換（Phase 3 advanced）
            # ホモトピー群やファイバー束理論ベース
            raise NotImplementedError("Topological method not implemented yet")

        else:
            raise ValueError(f"Unknown quad_method: {quad_method}")

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        隠れ状態を4つの思考ロールに分解

        Args:
            hidden: (batch, seq, hidden_dim)

        Returns:
            {
              "observation": (batch, seq, hidden_dim),
              "deduction":   (batch, seq, hidden_dim),
              "abduction":   (batch, seq, hidden_dim),
              "integration": (batch, seq, hidden_dim),
            }
        """
        if self.quad_method == "linear":
            # 線形射影 + Tanh活性化
            x = self.proj(hidden)  # (B, S, 4*H)
            x = self.act(x)        # ソフトな分離
            obs, ded, abd, integ = torch.chunk(x, 4, dim=-1)

        else:
            raise NotImplementedError(f"Method {self.quad_method} not implemented")

        return {
            "observation": obs,    # 観測・知覚
            "deduction":   ded,    # 演繹・論理推論
            "abduction":   abd,    # 仮説形成・帰納
            "integration": integ,  # 統合・合成
        }

    def get_quad_similarity(self) -> Dict[str, float]:
        """
        4ロール間の類似度を監視
        Phase 2.5 デバッグ用
        """
        if not hasattr(self, 'proj'):
            return {}

        W = self.proj.weight  # (4*H, H)
        quad_dim = self.hidden_dim

        similarities = {}
        for i, role1 in enumerate(['obs', 'ded', 'abd', 'integ']):
            for j, role2 in enumerate(['obs', 'ded', 'abd', 'integ']):
                if i < j:
                    w1 = W[i*quad_dim:(i+1)*quad_dim]  # (H, H)
                    w2 = W[j*quad_dim:(j+1)*quad_dim]  # (H, H)

                    # コサイン類似度
                    cos_sim = torch.nn.functional.cosine_similarity(
                        w1.flatten(), w2.flatten(), dim=0
                    ).item()

                    similarities[f'{role1}_{role2}'] = cos_sim

        return similarities


def attach_quad_reasoning_head(model, target_layers: Optional[list] = None):
    """
    モデルにQuadReasoningHeadを注入

    Phase 2.5 統合時に使用予定。
    現在はまだ学習損失に組み込まず、推論時のみ使用。

    Args:
        model: 対象モデル
        target_layers: 注入対象層（None=全層）
    """
    print("🧠 Attaching Quad Reasoning Head (Phase 2.5 preparation)...")

    # モデル構造解析
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        # LoRA適用後モデル
        layers = model.base_model.model.base_model.layers
        hidden_size = model.base_model.model.config.hidden_size
    elif hasattr(model, "base_model"):
        # 通常のPhi-3モデル
        layers = model.base_model.layers
        hidden_size = model.config.hidden_size
    else:
        raise ValueError("Unknown model structure")

    num_layers = len(layers)

    # ターゲット層決定
    if target_layers is None:
        target_layers = list(range(num_layers))  # 全層

    injected_count = 0
    for i in target_layers:
        if i >= num_layers:
            continue

        layer = layers[i]

        # QuadReasoningHead注入
        if not hasattr(layer, 'quad_reasoning_head'):
            quad_head = QuadReasoningHead(hidden_size)
            layer.add_module('quad_reasoning_head', quad_head)
            injected_count += 1

    print(f"[OK] Injected Quad Reasoning Heads into {injected_count} layers")

    # 推論専用なので勾配は切っておく（Phase 2.5 統合まで）
    for name, param in model.named_parameters():
        if 'quad_reasoning_head' in name:
            param.requires_grad = False

    return model


def extract_quad_reasoning(hidden_states: torch.Tensor,
                          quad_head: QuadReasoningHead) -> Dict[str, torch.Tensor]:
    """
    隠れ状態から4つの思考ロールを抽出

    Phase 2.5 の/think APIで使用予定

    Args:
        hidden_states: (batch, seq, hidden)
        quad_head: QuadReasoningHeadインスタンス

    Returns:
        4ロールの思考辞書
    """
    return quad_head(hidden_states)


# Phase 2.5 統合用のユーティリティ関数群
def create_quad_thinking_prompt(base_prompt: str) -> str:
    """
    Quadruple Inference 用のプロンプト生成

    Phase 2.5 で使用予定
    """
    quad_prompt = f"""
<think-observation>
観測フェーズ：{base_prompt}
現在の状況を観測し、利用可能な情報を整理する。
</think-observation>

<think-deduction>
演繹フェーズ：論理的推論に基づいて結論を導く。
</think-deduction>

<think-abduction>
仮説形成フェーズ：可能な説明や仮説を生成する。
</think-abduction>

<think-integration>
統合フェーズ：全ての思考を統合し、最終回答を生成する。
</think-integration>

<final>
"""

    return quad_prompt


def format_quad_response(quad_outputs: Dict[str, torch.Tensor],
                        tokenizer) -> str:
    """
    Quadruple Inference の出力を整形

    Phase 2.5 の/think APIレスポンス用
    """
    # 各ロールの出力をデコード（簡易実装）
    response_parts = []

    for role, hidden in quad_outputs.items():
        # 最後のトークンだけデコード（簡易版）
        last_token = hidden[:, -1:, :]  # (B, 1, H)
        # ここではダミーとしてロール名だけ返す
        response_parts.append(f"<think-{role}>[{role} thinking output]</think-{role}>")

    final_response = "\n".join(response_parts) + "\n<final>[final answer]</final>"

    return final_response


if __name__ == "__main__":
    # テスト実行
    print("🧠 Quad Reasoning Head Test")

    # ダミーデータ
    batch_size, seq_len, hidden_dim = 2, 10, 1024
    hidden = torch.randn(batch_size, seq_len, hidden_dim)

    # QuadReasoningHeadテスト
    quad_head = QuadReasoningHead(hidden_dim)

    quad_output = quad_head(hidden)
    print(f"Input shape: {hidden.shape}")
    print(f"Output keys: {list(quad_output.keys())}")
    for role, tensor in quad_output.items():
        print(f"  {role}: {tensor.shape}")

    # 類似度チェック
    similarities = quad_head.get_quad_similarity()
    print(f"Role similarities: {similarities}")

    print("[OK] Quad Reasoning Head test passed!")
# -*- coding: utf-8 -*-
"""
Quadruple Reasoning Head - Phase 2.5 準備

Observation / Deduction / Abduction / Integration
の4つのロールにhiddenを分解するヘッド。

Phase 2.5 (Quadruple Inference Integration) に向けて、
SO(8)回転エンジンの上で動く「思考の4ロール分解」器を準備。
"""

import torch
from torch import nn
from typing import Dict, Optional


class QuadReasoningHead(nn.Module):
    """
    NKAT理論に基づく4重推論ヘッド

    Observation / Deduction / Abduction / Integration
    の4つのロールにhidden_statesを分解する。

    特徴:
        - Phase 2.5 向けの思考ロール分離
        - SO(8)幾何変換との統合前提
        - Hookベース注入との互換性
        - 学習初期は線形分離のみ（安全策）
    """

    def __init__(self, hidden_dim: int, quad_method: str = "linear"):
        """
        Args:
            hidden_dim: 隠れ層次元数
            quad_method: 分解方法 ("linear", "so8_geometric", "topological")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.quad_method = quad_method

        if quad_method == "linear":
            # 線形分離（Phase 2.5 初期）
            self.proj = nn.Linear(hidden_dim, 4 * hidden_dim)
            self.act = nn.Tanh()  # ソフトな分離

        elif quad_method == "so8_geometric":
            # SO(8)幾何変換（Phase 3）
            # ここにSO(8)回転行列を使った分離を実装
            raise NotImplementedError("SO(8) geometric method not implemented yet")

        elif quad_method == "topological":
            # 位相幾何変換（Phase 3 advanced）
            # ホモトピー群やファイバー束理論ベース
            raise NotImplementedError("Topological method not implemented yet")

        else:
            raise ValueError(f"Unknown quad_method: {quad_method}")

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        隠れ状態を4つの思考ロールに分解

        Args:
            hidden: (batch, seq, hidden_dim)

        Returns:
            {
              "observation": (batch, seq, hidden_dim),
              "deduction":   (batch, seq, hidden_dim),
              "abduction":   (batch, seq, hidden_dim),
              "integration": (batch, seq, hidden_dim),
            }
        """
        if self.quad_method == "linear":
            # 線形射影 + Tanh活性化
            x = self.proj(hidden)  # (B, S, 4*H)
            x = self.act(x)        # ソフトな分離
            obs, ded, abd, integ = torch.chunk(x, 4, dim=-1)

        else:
            raise NotImplementedError(f"Method {self.quad_method} not implemented")

        return {
            "observation": obs,    # 観測・知覚
            "deduction":   ded,    # 演繹・論理推論
            "abduction":   abd,    # 仮説形成・帰納
            "integration": integ,  # 統合・合成
        }

    def get_quad_similarity(self) -> Dict[str, float]:
        """
        4ロール間の類似度を監視
        Phase 2.5 デバッグ用
        """
        if not hasattr(self, 'proj'):
            return {}

        W = self.proj.weight  # (4*H, H)
        quad_dim = self.hidden_dim

        similarities = {}
        for i, role1 in enumerate(['obs', 'ded', 'abd', 'integ']):
            for j, role2 in enumerate(['obs', 'ded', 'abd', 'integ']):
                if i < j:
                    w1 = W[i*quad_dim:(i+1)*quad_dim]  # (H, H)
                    w2 = W[j*quad_dim:(j+1)*quad_dim]  # (H, H)

                    # コサイン類似度
                    cos_sim = torch.nn.functional.cosine_similarity(
                        w1.flatten(), w2.flatten(), dim=0
                    ).item()

                    similarities[f'{role1}_{role2}'] = cos_sim

        return similarities


def attach_quad_reasoning_head(model, target_layers: Optional[list] = None):
    """
    モデルにQuadReasoningHeadを注入

    Phase 2.5 統合時に使用予定。
    現在はまだ学習損失に組み込まず、推論時のみ使用。

    Args:
        model: 対象モデル
        target_layers: 注入対象層（None=全層）
    """
    print("🧠 Attaching Quad Reasoning Head (Phase 2.5 preparation)...")

    # モデル構造解析
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        # LoRA適用後モデル
        layers = model.base_model.model.base_model.layers
        hidden_size = model.base_model.model.config.hidden_size
    elif hasattr(model, "base_model"):
        # 通常のPhi-3モデル
        layers = model.base_model.layers
        hidden_size = model.config.hidden_size
    else:
        raise ValueError("Unknown model structure")

    num_layers = len(layers)

    # ターゲット層決定
    if target_layers is None:
        target_layers = list(range(num_layers))  # 全層

    injected_count = 0
    for i in target_layers:
        if i >= num_layers:
            continue

        layer = layers[i]

        # QuadReasoningHead注入
        if not hasattr(layer, 'quad_reasoning_head'):
            quad_head = QuadReasoningHead(hidden_size)
            layer.add_module('quad_reasoning_head', quad_head)
            injected_count += 1

    print(f"[OK] Injected Quad Reasoning Heads into {injected_count} layers")

    # 推論専用なので勾配は切っておく（Phase 2.5 統合まで）
    for name, param in model.named_parameters():
        if 'quad_reasoning_head' in name:
            param.requires_grad = False

    return model


def extract_quad_reasoning(hidden_states: torch.Tensor,
                          quad_head: QuadReasoningHead) -> Dict[str, torch.Tensor]:
    """
    隠れ状態から4つの思考ロールを抽出

    Phase 2.5 の/think APIで使用予定

    Args:
        hidden_states: (batch, seq, hidden)
        quad_head: QuadReasoningHeadインスタンス

    Returns:
        4ロールの思考辞書
    """
    return quad_head(hidden_states)


# Phase 2.5 統合用のユーティリティ関数群
def create_quad_thinking_prompt(base_prompt: str) -> str:
    """
    Quadruple Inference 用のプロンプト生成

    Phase 2.5 で使用予定
    """
    quad_prompt = f"""
<think-observation>
観測フェーズ：{base_prompt}
現在の状況を観測し、利用可能な情報を整理する。
</think-observation>

<think-deduction>
演繹フェーズ：論理的推論に基づいて結論を導く。
</think-deduction>

<think-abduction>
仮説形成フェーズ：可能な説明や仮説を生成する。
</think-abduction>

<think-integration>
統合フェーズ：全ての思考を統合し、最終回答を生成する。
</think-integration>

<final>
"""

    return quad_prompt


def format_quad_response(quad_outputs: Dict[str, torch.Tensor],
                        tokenizer) -> str:
    """
    Quadruple Inference の出力を整形

    Phase 2.5 の/think APIレスポンス用
    """
    # 各ロールの出力をデコード（簡易実装）
    response_parts = []

    for role, hidden in quad_outputs.items():
        # 最後のトークンだけデコード（簡易版）
        last_token = hidden[:, -1:, :]  # (B, 1, H)
        # ここではダミーとしてロール名だけ返す
        response_parts.append(f"<think-{role}>[{role} thinking output]</think-{role}>")

    final_response = "\n".join(response_parts) + "\n<final>[final answer]</final>"

    return final_response


if __name__ == "__main__":
    # テスト実行
    print("🧠 Quad Reasoning Head Test")

    # ダミーデータ
    batch_size, seq_len, hidden_dim = 2, 10, 1024
    hidden = torch.randn(batch_size, seq_len, hidden_dim)

    # QuadReasoningHeadテスト
    quad_head = QuadReasoningHead(hidden_dim)

    quad_output = quad_head(hidden)
    print(f"Input shape: {hidden.shape}")
    print(f"Output keys: {list(quad_output.keys())}")
    for role, tensor in quad_output.items():
        print(f"  {role}: {tensor.shape}")

    # 類似度チェック
    similarities = quad_head.get_quad_similarity()
    print(f"Role similarities: {similarities}")

    print("[OK] Quad Reasoning Head test passed!")
# -*- coding: utf-8 -*-
"""
Quadruple Reasoning Head - Phase 2.5 準備

Observation / Deduction / Abduction / Integration
の4つのロールにhiddenを分解するヘッド。

Phase 2.5 (Quadruple Inference Integration) に向けて、
SO(8)回転エンジンの上で動く「思考の4ロール分解」器を準備。
"""

import torch
from torch import nn
from typing import Dict, Optional


class QuadReasoningHead(nn.Module):
    """
    NKAT理論に基づく4重推論ヘッド

    Observation / Deduction / Abduction / Integration
    の4つのロールにhidden_statesを分解する。

    特徴:
        - Phase 2.5 向けの思考ロール分離
        - SO(8)幾何変換との統合前提
        - Hookベース注入との互換性
        - 学習初期は線形分離のみ（安全策）
    """

    def __init__(self, hidden_dim: int, quad_method: str = "linear"):
        """
        Args:
            hidden_dim: 隠れ層次元数
            quad_method: 分解方法 ("linear", "so8_geometric", "topological")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.quad_method = quad_method

        if quad_method == "linear":
            # 線形分離（Phase 2.5 初期）
            self.proj = nn.Linear(hidden_dim, 4 * hidden_dim)
            self.act = nn.Tanh()  # ソフトな分離

        elif quad_method == "so8_geometric":
            # SO(8)幾何変換（Phase 3）
            # ここにSO(8)回転行列を使った分離を実装
            raise NotImplementedError("SO(8) geometric method not implemented yet")

        elif quad_method == "topological":
            # 位相幾何変換（Phase 3 advanced）
            # ホモトピー群やファイバー束理論ベース
            raise NotImplementedError("Topological method not implemented yet")

        else:
            raise ValueError(f"Unknown quad_method: {quad_method}")

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        隠れ状態を4つの思考ロールに分解

        Args:
            hidden: (batch, seq, hidden_dim)

        Returns:
            {
              "observation": (batch, seq, hidden_dim),
              "deduction":   (batch, seq, hidden_dim),
              "abduction":   (batch, seq, hidden_dim),
              "integration": (batch, seq, hidden_dim),
            }
        """
        if self.quad_method == "linear":
            # 線形射影 + Tanh活性化
            x = self.proj(hidden)  # (B, S, 4*H)
            x = self.act(x)        # ソフトな分離
            obs, ded, abd, integ = torch.chunk(x, 4, dim=-1)

        else:
            raise NotImplementedError(f"Method {self.quad_method} not implemented")

        return {
            "observation": obs,    # 観測・知覚
            "deduction":   ded,    # 演繹・論理推論
            "abduction":   abd,    # 仮説形成・帰納
            "integration": integ,  # 統合・合成
        }

    def get_quad_similarity(self) -> Dict[str, float]:
        """
        4ロール間の類似度を監視
        Phase 2.5 デバッグ用
        """
        if not hasattr(self, 'proj'):
            return {}

        W = self.proj.weight  # (4*H, H)
        quad_dim = self.hidden_dim

        similarities = {}
        for i, role1 in enumerate(['obs', 'ded', 'abd', 'integ']):
            for j, role2 in enumerate(['obs', 'ded', 'abd', 'integ']):
                if i < j:
                    w1 = W[i*quad_dim:(i+1)*quad_dim]  # (H, H)
                    w2 = W[j*quad_dim:(j+1)*quad_dim]  # (H, H)

                    # コサイン類似度
                    cos_sim = torch.nn.functional.cosine_similarity(
                        w1.flatten(), w2.flatten(), dim=0
                    ).item()

                    similarities[f'{role1}_{role2}'] = cos_sim

        return similarities


def attach_quad_reasoning_head(model, target_layers: Optional[list] = None):
    """
    モデルにQuadReasoningHeadを注入

    Phase 2.5 統合時に使用予定。
    現在はまだ学習損失に組み込まず、推論時のみ使用。

    Args:
        model: 対象モデル
        target_layers: 注入対象層（None=全層）
    """
    print("🧠 Attaching Quad Reasoning Head (Phase 2.5 preparation)...")

    # モデル構造解析
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        # LoRA適用後モデル
        layers = model.base_model.model.base_model.layers
        hidden_size = model.base_model.model.config.hidden_size
    elif hasattr(model, "base_model"):
        # 通常のPhi-3モデル
        layers = model.base_model.layers
        hidden_size = model.config.hidden_size
    else:
        raise ValueError("Unknown model structure")

    num_layers = len(layers)

    # ターゲット層決定
    if target_layers is None:
        target_layers = list(range(num_layers))  # 全層

    injected_count = 0
    for i in target_layers:
        if i >= num_layers:
            continue

        layer = layers[i]

        # QuadReasoningHead注入
        if not hasattr(layer, 'quad_reasoning_head'):
            quad_head = QuadReasoningHead(hidden_size)
            layer.add_module('quad_reasoning_head', quad_head)
            injected_count += 1

    print(f"[OK] Injected Quad Reasoning Heads into {injected_count} layers")

    # 推論専用なので勾配は切っておく（Phase 2.5 統合まで）
    for name, param in model.named_parameters():
        if 'quad_reasoning_head' in name:
            param.requires_grad = False

    return model


def extract_quad_reasoning(hidden_states: torch.Tensor,
                          quad_head: QuadReasoningHead) -> Dict[str, torch.Tensor]:
    """
    隠れ状態から4つの思考ロールを抽出

    Phase 2.5 の/think APIで使用予定

    Args:
        hidden_states: (batch, seq, hidden)
        quad_head: QuadReasoningHeadインスタンス

    Returns:
        4ロールの思考辞書
    """
    return quad_head(hidden_states)


# Phase 2.5 統合用のユーティリティ関数群
def create_quad_thinking_prompt(base_prompt: str) -> str:
    """
    Quadruple Inference 用のプロンプト生成

    Phase 2.5 で使用予定
    """
    quad_prompt = f"""
<think-observation>
観測フェーズ：{base_prompt}
現在の状況を観測し、利用可能な情報を整理する。
</think-observation>

<think-deduction>
演繹フェーズ：論理的推論に基づいて結論を導く。
</think-deduction>

<think-abduction>
仮説形成フェーズ：可能な説明や仮説を生成する。
</think-abduction>

<think-integration>
統合フェーズ：全ての思考を統合し、最終回答を生成する。
</think-integration>

<final>
"""

    return quad_prompt


def format_quad_response(quad_outputs: Dict[str, torch.Tensor],
                        tokenizer) -> str:
    """
    Quadruple Inference の出力を整形

    Phase 2.5 の/think APIレスポンス用
    """
    # 各ロールの出力をデコード（簡易実装）
    response_parts = []

    for role, hidden in quad_outputs.items():
        # 最後のトークンだけデコード（簡易版）
        last_token = hidden[:, -1:, :]  # (B, 1, H)
        # ここではダミーとしてロール名だけ返す
        response_parts.append(f"<think-{role}>[{role} thinking output]</think-{role}>")

    final_response = "\n".join(response_parts) + "\n<final>[final answer]</final>"

    return final_response


if __name__ == "__main__":
    # テスト実行
    print("🧠 Quad Reasoning Head Test")

    # ダミーデータ
    batch_size, seq_len, hidden_dim = 2, 10, 1024
    hidden = torch.randn(batch_size, seq_len, hidden_dim)

    # QuadReasoningHeadテスト
    quad_head = QuadReasoningHead(hidden_dim)

    quad_output = quad_head(hidden)
    print(f"Input shape: {hidden.shape}")
    print(f"Output keys: {list(quad_output.keys())}")
    for role, tensor in quad_output.items():
        print(f"  {role}: {tensor.shape}")

    # 類似度チェック
    similarities = quad_head.get_quad_similarity()
    print(f"Role similarities: {similarities}")

    print("[OK] Quad Reasoning Head test passed!")
# -*- coding: utf-8 -*-
"""
Quadruple Reasoning Head - Phase 2.5 準備

Observation / Deduction / Abduction / Integration
の4つのロールにhiddenを分解するヘッド。

Phase 2.5 (Quadruple Inference Integration) に向けて、
SO(8)回転エンジンの上で動く「思考の4ロール分解」器を準備。
"""

import torch
from torch import nn
from typing import Dict, Optional


class QuadReasoningHead(nn.Module):
    """
    NKAT理論に基づく4重推論ヘッド

    Observation / Deduction / Abduction / Integration
    の4つのロールにhidden_statesを分解する。

    特徴:
        - Phase 2.5 向けの思考ロール分離
        - SO(8)幾何変換との統合前提
        - Hookベース注入との互換性
        - 学習初期は線形分離のみ（安全策）
    """

    def __init__(self, hidden_dim: int, quad_method: str = "linear"):
        """
        Args:
            hidden_dim: 隠れ層次元数
            quad_method: 分解方法 ("linear", "so8_geometric", "topological")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.quad_method = quad_method

        if quad_method == "linear":
            # 線形分離（Phase 2.5 初期）
            self.proj = nn.Linear(hidden_dim, 4 * hidden_dim)
            self.act = nn.Tanh()  # ソフトな分離

        elif quad_method == "so8_geometric":
            # SO(8)幾何変換（Phase 3）
            # ここにSO(8)回転行列を使った分離を実装
            raise NotImplementedError("SO(8) geometric method not implemented yet")

        elif quad_method == "topological":
            # 位相幾何変換（Phase 3 advanced）
            # ホモトピー群やファイバー束理論ベース
            raise NotImplementedError("Topological method not implemented yet")

        else:
            raise ValueError(f"Unknown quad_method: {quad_method}")

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        隠れ状態を4つの思考ロールに分解

        Args:
            hidden: (batch, seq, hidden_dim)

        Returns:
            {
              "observation": (batch, seq, hidden_dim),
              "deduction":   (batch, seq, hidden_dim),
              "abduction":   (batch, seq, hidden_dim),
              "integration": (batch, seq, hidden_dim),
            }
        """
        if self.quad_method == "linear":
            # 線形射影 + Tanh活性化
            x = self.proj(hidden)  # (B, S, 4*H)
            x = self.act(x)        # ソフトな分離
            obs, ded, abd, integ = torch.chunk(x, 4, dim=-1)

        else:
            raise NotImplementedError(f"Method {self.quad_method} not implemented")

        return {
            "observation": obs,    # 観測・知覚
            "deduction":   ded,    # 演繹・論理推論
            "abduction":   abd,    # 仮説形成・帰納
            "integration": integ,  # 統合・合成
        }

    def get_quad_similarity(self) -> Dict[str, float]:
        """
        4ロール間の類似度を監視
        Phase 2.5 デバッグ用
        """
        if not hasattr(self, 'proj'):
            return {}

        W = self.proj.weight  # (4*H, H)
        quad_dim = self.hidden_dim

        similarities = {}
        for i, role1 in enumerate(['obs', 'ded', 'abd', 'integ']):
            for j, role2 in enumerate(['obs', 'ded', 'abd', 'integ']):
                if i < j:
                    w1 = W[i*quad_dim:(i+1)*quad_dim]  # (H, H)
                    w2 = W[j*quad_dim:(j+1)*quad_dim]  # (H, H)

                    # コサイン類似度
                    cos_sim = torch.nn.functional.cosine_similarity(
                        w1.flatten(), w2.flatten(), dim=0
                    ).item()

                    similarities[f'{role1}_{role2}'] = cos_sim

        return similarities


def attach_quad_reasoning_head(model, target_layers: Optional[list] = None):
    """
    モデルにQuadReasoningHeadを注入

    Phase 2.5 統合時に使用予定。
    現在はまだ学習損失に組み込まず、推論時のみ使用。

    Args:
        model: 対象モデル
        target_layers: 注入対象層（None=全層）
    """
    print("🧠 Attaching Quad Reasoning Head (Phase 2.5 preparation)...")

    # モデル構造解析
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        # LoRA適用後モデル
        layers = model.base_model.model.base_model.layers
        hidden_size = model.base_model.model.config.hidden_size
    elif hasattr(model, "base_model"):
        # 通常のPhi-3モデル
        layers = model.base_model.layers
        hidden_size = model.config.hidden_size
    else:
        raise ValueError("Unknown model structure")

    num_layers = len(layers)

    # ターゲット層決定
    if target_layers is None:
        target_layers = list(range(num_layers))  # 全層

    injected_count = 0
    for i in target_layers:
        if i >= num_layers:
            continue

        layer = layers[i]

        # QuadReasoningHead注入
        if not hasattr(layer, 'quad_reasoning_head'):
            quad_head = QuadReasoningHead(hidden_size)
            layer.add_module('quad_reasoning_head', quad_head)
            injected_count += 1

    print(f"[OK] Injected Quad Reasoning Heads into {injected_count} layers")

    # 推論専用なので勾配は切っておく（Phase 2.5 統合まで）
    for name, param in model.named_parameters():
        if 'quad_reasoning_head' in name:
            param.requires_grad = False

    return model


def extract_quad_reasoning(hidden_states: torch.Tensor,
                          quad_head: QuadReasoningHead) -> Dict[str, torch.Tensor]:
    """
    隠れ状態から4つの思考ロールを抽出

    Phase 2.5 の/think APIで使用予定

    Args:
        hidden_states: (batch, seq, hidden)
        quad_head: QuadReasoningHeadインスタンス

    Returns:
        4ロールの思考辞書
    """
    return quad_head(hidden_states)


# Phase 2.5 統合用のユーティリティ関数群
def create_quad_thinking_prompt(base_prompt: str) -> str:
    """
    Quadruple Inference 用のプロンプト生成

    Phase 2.5 で使用予定
    """
    quad_prompt = f"""
<think-observation>
観測フェーズ：{base_prompt}
現在の状況を観測し、利用可能な情報を整理する。
</think-observation>

<think-deduction>
演繹フェーズ：論理的推論に基づいて結論を導く。
</think-deduction>

<think-abduction>
仮説形成フェーズ：可能な説明や仮説を生成する。
</think-abduction>

<think-integration>
統合フェーズ：全ての思考を統合し、最終回答を生成する。
</think-integration>

<final>
"""

    return quad_prompt


def format_quad_response(quad_outputs: Dict[str, torch.Tensor],
                        tokenizer) -> str:
    """
    Quadruple Inference の出力を整形

    Phase 2.5 の/think APIレスポンス用
    """
    # 各ロールの出力をデコード（簡易実装）
    response_parts = []

    for role, hidden in quad_outputs.items():
        # 最後のトークンだけデコード（簡易版）
        last_token = hidden[:, -1:, :]  # (B, 1, H)
        # ここではダミーとしてロール名だけ返す
        response_parts.append(f"<think-{role}>[{role} thinking output]</think-{role}>")

    final_response = "\n".join(response_parts) + "\n<final>[final answer]</final>"

    return final_response


if __name__ == "__main__":
    # テスト実行
    print("🧠 Quad Reasoning Head Test")

    # ダミーデータ
    batch_size, seq_len, hidden_dim = 2, 10, 1024
    hidden = torch.randn(batch_size, seq_len, hidden_dim)

    # QuadReasoningHeadテスト
    quad_head = QuadReasoningHead(hidden_dim)

    quad_output = quad_head(hidden)
    print(f"Input shape: {hidden.shape}")
    print(f"Output keys: {list(quad_output.keys())}")
    for role, tensor in quad_output.items():
        print(f"  {role}: {tensor.shape}")

    # 類似度チェック
    similarities = quad_head.get_quad_similarity()
    print(f"Role similarities: {similarities}")

    print("[OK] Quad Reasoning Head test passed!")
# -*- coding: utf-8 -*-
"""
Quadruple Reasoning Head - Phase 2.5 準備

Observation / Deduction / Abduction / Integration
の4つのロールにhiddenを分解するヘッド。

Phase 2.5 (Quadruple Inference Integration) に向けて、
SO(8)回転エンジンの上で動く「思考の4ロール分解」器を準備。
"""

import torch
from torch import nn
from typing import Dict, Optional


class QuadReasoningHead(nn.Module):
    """
    NKAT理論に基づく4重推論ヘッド

    Observation / Deduction / Abduction / Integration
    の4つのロールにhidden_statesを分解する。

    特徴:
        - Phase 2.5 向けの思考ロール分離
        - SO(8)幾何変換との統合前提
        - Hookベース注入との互換性
        - 学習初期は線形分離のみ（安全策）
    """

    def __init__(self, hidden_dim: int, quad_method: str = "linear"):
        """
        Args:
            hidden_dim: 隠れ層次元数
            quad_method: 分解方法 ("linear", "so8_geometric", "topological")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.quad_method = quad_method

        if quad_method == "linear":
            # 線形分離（Phase 2.5 初期）
            self.proj = nn.Linear(hidden_dim, 4 * hidden_dim)
            self.act = nn.Tanh()  # ソフトな分離

        elif quad_method == "so8_geometric":
            # SO(8)幾何変換（Phase 3）
            # ここにSO(8)回転行列を使った分離を実装
            raise NotImplementedError("SO(8) geometric method not implemented yet")

        elif quad_method == "topological":
            # 位相幾何変換（Phase 3 advanced）
            # ホモトピー群やファイバー束理論ベース
            raise NotImplementedError("Topological method not implemented yet")

        else:
            raise ValueError(f"Unknown quad_method: {quad_method}")

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        隠れ状態を4つの思考ロールに分解

        Args:
            hidden: (batch, seq, hidden_dim)

        Returns:
            {
              "observation": (batch, seq, hidden_dim),
              "deduction":   (batch, seq, hidden_dim),
              "abduction":   (batch, seq, hidden_dim),
              "integration": (batch, seq, hidden_dim),
            }
        """
        if self.quad_method == "linear":
            # 線形射影 + Tanh活性化
            x = self.proj(hidden)  # (B, S, 4*H)
            x = self.act(x)        # ソフトな分離
            obs, ded, abd, integ = torch.chunk(x, 4, dim=-1)

        else:
            raise NotImplementedError(f"Method {self.quad_method} not implemented")

        return {
            "observation": obs,    # 観測・知覚
            "deduction":   ded,    # 演繹・論理推論
            "abduction":   abd,    # 仮説形成・帰納
            "integration": integ,  # 統合・合成
        }

    def get_quad_similarity(self) -> Dict[str, float]:
        """
        4ロール間の類似度を監視
        Phase 2.5 デバッグ用
        """
        if not hasattr(self, 'proj'):
            return {}

        W = self.proj.weight  # (4*H, H)
        quad_dim = self.hidden_dim

        similarities = {}
        for i, role1 in enumerate(['obs', 'ded', 'abd', 'integ']):
            for j, role2 in enumerate(['obs', 'ded', 'abd', 'integ']):
                if i < j:
                    w1 = W[i*quad_dim:(i+1)*quad_dim]  # (H, H)
                    w2 = W[j*quad_dim:(j+1)*quad_dim]  # (H, H)

                    # コサイン類似度
                    cos_sim = torch.nn.functional.cosine_similarity(
                        w1.flatten(), w2.flatten(), dim=0
                    ).item()

                    similarities[f'{role1}_{role2}'] = cos_sim

        return similarities


def attach_quad_reasoning_head(model, target_layers: Optional[list] = None):
    """
    モデルにQuadReasoningHeadを注入

    Phase 2.5 統合時に使用予定。
    現在はまだ学習損失に組み込まず、推論時のみ使用。

    Args:
        model: 対象モデル
        target_layers: 注入対象層（None=全層）
    """
    print("🧠 Attaching Quad Reasoning Head (Phase 2.5 preparation)...")

    # モデル構造解析
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        # LoRA適用後モデル
        layers = model.base_model.model.base_model.layers
        hidden_size = model.base_model.model.config.hidden_size
    elif hasattr(model, "base_model"):
        # 通常のPhi-3モデル
        layers = model.base_model.layers
        hidden_size = model.config.hidden_size
    else:
        raise ValueError("Unknown model structure")

    num_layers = len(layers)

    # ターゲット層決定
    if target_layers is None:
        target_layers = list(range(num_layers))  # 全層

    injected_count = 0
    for i in target_layers:
        if i >= num_layers:
            continue

        layer = layers[i]

        # QuadReasoningHead注入
        if not hasattr(layer, 'quad_reasoning_head'):
            quad_head = QuadReasoningHead(hidden_size)
            layer.add_module('quad_reasoning_head', quad_head)
            injected_count += 1

    print(f"[OK] Injected Quad Reasoning Heads into {injected_count} layers")

    # 推論専用なので勾配は切っておく（Phase 2.5 統合まで）
    for name, param in model.named_parameters():
        if 'quad_reasoning_head' in name:
            param.requires_grad = False

    return model


def extract_quad_reasoning(hidden_states: torch.Tensor,
                          quad_head: QuadReasoningHead) -> Dict[str, torch.Tensor]:
    """
    隠れ状態から4つの思考ロールを抽出

    Phase 2.5 の/think APIで使用予定

    Args:
        hidden_states: (batch, seq, hidden)
        quad_head: QuadReasoningHeadインスタンス

    Returns:
        4ロールの思考辞書
    """
    return quad_head(hidden_states)


# Phase 2.5 統合用のユーティリティ関数群
def create_quad_thinking_prompt(base_prompt: str) -> str:
    """
    Quadruple Inference 用のプロンプト生成

    Phase 2.5 で使用予定
    """
    quad_prompt = f"""
<think-observation>
観測フェーズ：{base_prompt}
現在の状況を観測し、利用可能な情報を整理する。
</think-observation>

<think-deduction>
演繹フェーズ：論理的推論に基づいて結論を導く。
</think-deduction>

<think-abduction>
仮説形成フェーズ：可能な説明や仮説を生成する。
</think-abduction>

<think-integration>
統合フェーズ：全ての思考を統合し、最終回答を生成する。
</think-integration>

<final>
"""

    return quad_prompt


def format_quad_response(quad_outputs: Dict[str, torch.Tensor],
                        tokenizer) -> str:
    """
    Quadruple Inference の出力を整形

    Phase 2.5 の/think APIレスポンス用
    """
    # 各ロールの出力をデコード（簡易実装）
    response_parts = []

    for role, hidden in quad_outputs.items():
        # 最後のトークンだけデコード（簡易版）
        last_token = hidden[:, -1:, :]  # (B, 1, H)
        # ここではダミーとしてロール名だけ返す
        response_parts.append(f"<think-{role}>[{role} thinking output]</think-{role}>")

    final_response = "\n".join(response_parts) + "\n<final>[final answer]</final>"

    return final_response


if __name__ == "__main__":
    # テスト実行
    print("🧠 Quad Reasoning Head Test")

    # ダミーデータ
    batch_size, seq_len, hidden_dim = 2, 10, 1024
    hidden = torch.randn(batch_size, seq_len, hidden_dim)

    # QuadReasoningHeadテスト
    quad_head = QuadReasoningHead(hidden_dim)

    quad_output = quad_head(hidden)
    print(f"Input shape: {hidden.shape}")
    print(f"Output keys: {list(quad_output.keys())}")
    for role, tensor in quad_output.items():
        print(f"  {role}: {tensor.shape}")

    # 類似度チェック
    similarities = quad_head.get_quad_similarity()
    print(f"Role similarities: {similarities}")

    print("[OK] Quad Reasoning Head test passed!")
# -*- coding: utf-8 -*-
"""
Quadruple Reasoning Head - Phase 2.5 準備

Observation / Deduction / Abduction / Integration
の4つのロールにhiddenを分解するヘッド。

Phase 2.5 (Quadruple Inference Integration) に向けて、
SO(8)回転エンジンの上で動く「思考の4ロール分解」器を準備。
"""

import torch
from torch import nn
from typing import Dict, Optional


class QuadReasoningHead(nn.Module):
    """
    NKAT理論に基づく4重推論ヘッド

    Observation / Deduction / Abduction / Integration
    の4つのロールにhidden_statesを分解する。

    特徴:
        - Phase 2.5 向けの思考ロール分離
        - SO(8)幾何変換との統合前提
        - Hookベース注入との互換性
        - 学習初期は線形分離のみ（安全策）
    """

    def __init__(self, hidden_dim: int, quad_method: str = "linear"):
        """
        Args:
            hidden_dim: 隠れ層次元数
            quad_method: 分解方法 ("linear", "so8_geometric", "topological")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.quad_method = quad_method

        if quad_method == "linear":
            # 線形分離（Phase 2.5 初期）
            self.proj = nn.Linear(hidden_dim, 4 * hidden_dim)
            self.act = nn.Tanh()  # ソフトな分離

        elif quad_method == "so8_geometric":
            # SO(8)幾何変換（Phase 3）
            # ここにSO(8)回転行列を使った分離を実装
            raise NotImplementedError("SO(8) geometric method not implemented yet")

        elif quad_method == "topological":
            # 位相幾何変換（Phase 3 advanced）
            # ホモトピー群やファイバー束理論ベース
            raise NotImplementedError("Topological method not implemented yet")

        else:
            raise ValueError(f"Unknown quad_method: {quad_method}")

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        隠れ状態を4つの思考ロールに分解

        Args:
            hidden: (batch, seq, hidden_dim)

        Returns:
            {
              "observation": (batch, seq, hidden_dim),
              "deduction":   (batch, seq, hidden_dim),
              "abduction":   (batch, seq, hidden_dim),
              "integration": (batch, seq, hidden_dim),
            }
        """
        if self.quad_method == "linear":
            # 線形射影 + Tanh活性化
            x = self.proj(hidden)  # (B, S, 4*H)
            x = self.act(x)        # ソフトな分離
            obs, ded, abd, integ = torch.chunk(x, 4, dim=-1)

        else:
            raise NotImplementedError(f"Method {self.quad_method} not implemented")

        return {
            "observation": obs,    # 観測・知覚
            "deduction":   ded,    # 演繹・論理推論
            "abduction":   abd,    # 仮説形成・帰納
            "integration": integ,  # 統合・合成
        }

    def get_quad_similarity(self) -> Dict[str, float]:
        """
        4ロール間の類似度を監視
        Phase 2.5 デバッグ用
        """
        if not hasattr(self, 'proj'):
            return {}

        W = self.proj.weight  # (4*H, H)
        quad_dim = self.hidden_dim

        similarities = {}
        for i, role1 in enumerate(['obs', 'ded', 'abd', 'integ']):
            for j, role2 in enumerate(['obs', 'ded', 'abd', 'integ']):
                if i < j:
                    w1 = W[i*quad_dim:(i+1)*quad_dim]  # (H, H)
                    w2 = W[j*quad_dim:(j+1)*quad_dim]  # (H, H)

                    # コサイン類似度
                    cos_sim = torch.nn.functional.cosine_similarity(
                        w1.flatten(), w2.flatten(), dim=0
                    ).item()

                    similarities[f'{role1}_{role2}'] = cos_sim

        return similarities


def attach_quad_reasoning_head(model, target_layers: Optional[list] = None):
    """
    モデルにQuadReasoningHeadを注入

    Phase 2.5 統合時に使用予定。
    現在はまだ学習損失に組み込まず、推論時のみ使用。

    Args:
        model: 対象モデル
        target_layers: 注入対象層（None=全層）
    """
    print("🧠 Attaching Quad Reasoning Head (Phase 2.5 preparation)...")

    # モデル構造解析
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        # LoRA適用後モデル
        layers = model.base_model.model.base_model.layers
        hidden_size = model.base_model.model.config.hidden_size
    elif hasattr(model, "base_model"):
        # 通常のPhi-3モデル
        layers = model.base_model.layers
        hidden_size = model.config.hidden_size
    else:
        raise ValueError("Unknown model structure")

    num_layers = len(layers)

    # ターゲット層決定
    if target_layers is None:
        target_layers = list(range(num_layers))  # 全層

    injected_count = 0
    for i in target_layers:
        if i >= num_layers:
            continue

        layer = layers[i]

        # QuadReasoningHead注入
        if not hasattr(layer, 'quad_reasoning_head'):
            quad_head = QuadReasoningHead(hidden_size)
            layer.add_module('quad_reasoning_head', quad_head)
            injected_count += 1

    print(f"[OK] Injected Quad Reasoning Heads into {injected_count} layers")

    # 推論専用なので勾配は切っておく（Phase 2.5 統合まで）
    for name, param in model.named_parameters():
        if 'quad_reasoning_head' in name:
            param.requires_grad = False

    return model


def extract_quad_reasoning(hidden_states: torch.Tensor,
                          quad_head: QuadReasoningHead) -> Dict[str, torch.Tensor]:
    """
    隠れ状態から4つの思考ロールを抽出

    Phase 2.5 の/think APIで使用予定

    Args:
        hidden_states: (batch, seq, hidden)
        quad_head: QuadReasoningHeadインスタンス

    Returns:
        4ロールの思考辞書
    """
    return quad_head(hidden_states)


# Phase 2.5 統合用のユーティリティ関数群
def create_quad_thinking_prompt(base_prompt: str) -> str:
    """
    Quadruple Inference 用のプロンプト生成

    Phase 2.5 で使用予定
    """
    quad_prompt = f"""
<think-observation>
観測フェーズ：{base_prompt}
現在の状況を観測し、利用可能な情報を整理する。
</think-observation>

<think-deduction>
演繹フェーズ：論理的推論に基づいて結論を導く。
</think-deduction>

<think-abduction>
仮説形成フェーズ：可能な説明や仮説を生成する。
</think-abduction>

<think-integration>
統合フェーズ：全ての思考を統合し、最終回答を生成する。
</think-integration>

<final>
"""

    return quad_prompt


def format_quad_response(quad_outputs: Dict[str, torch.Tensor],
                        tokenizer) -> str:
    """
    Quadruple Inference の出力を整形

    Phase 2.5 の/think APIレスポンス用
    """
    # 各ロールの出力をデコード（簡易実装）
    response_parts = []

    for role, hidden in quad_outputs.items():
        # 最後のトークンだけデコード（簡易版）
        last_token = hidden[:, -1:, :]  # (B, 1, H)
        # ここではダミーとしてロール名だけ返す
        response_parts.append(f"<think-{role}>[{role} thinking output]</think-{role}>")

    final_response = "\n".join(response_parts) + "\n<final>[final answer]</final>"

    return final_response


if __name__ == "__main__":
    # テスト実行
    print("🧠 Quad Reasoning Head Test")

    # ダミーデータ
    batch_size, seq_len, hidden_dim = 2, 10, 1024
    hidden = torch.randn(batch_size, seq_len, hidden_dim)

    # QuadReasoningHeadテスト
    quad_head = QuadReasoningHead(hidden_dim)

    quad_output = quad_head(hidden)
    print(f"Input shape: {hidden.shape}")
    print(f"Output keys: {list(quad_output.keys())}")
    for role, tensor in quad_output.items():
        print(f"  {role}: {tensor.shape}")

    # 類似度チェック
    similarities = quad_head.get_quad_similarity()
    print(f"Role similarities: {similarities}")

    print("[OK] Quad Reasoning Head test passed!")
# -*- coding: utf-8 -*-
"""
Quadruple Reasoning Head - Phase 2.5 準備

Observation / Deduction / Abduction / Integration
の4つのロールにhiddenを分解するヘッド。

Phase 2.5 (Quadruple Inference Integration) に向けて、
SO(8)回転エンジンの上で動く「思考の4ロール分解」器を準備。
"""

import torch
from torch import nn
from typing import Dict, Optional


class QuadReasoningHead(nn.Module):
    """
    NKAT理論に基づく4重推論ヘッド

    Observation / Deduction / Abduction / Integration
    の4つのロールにhidden_statesを分解する。

    特徴:
        - Phase 2.5 向けの思考ロール分離
        - SO(8)幾何変換との統合前提
        - Hookベース注入との互換性
        - 学習初期は線形分離のみ（安全策）
    """

    def __init__(self, hidden_dim: int, quad_method: str = "linear"):
        """
        Args:
            hidden_dim: 隠れ層次元数
            quad_method: 分解方法 ("linear", "so8_geometric", "topological")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.quad_method = quad_method

        if quad_method == "linear":
            # 線形分離（Phase 2.5 初期）
            self.proj = nn.Linear(hidden_dim, 4 * hidden_dim)
            self.act = nn.Tanh()  # ソフトな分離

        elif quad_method == "so8_geometric":
            # SO(8)幾何変換（Phase 3）
            # ここにSO(8)回転行列を使った分離を実装
            raise NotImplementedError("SO(8) geometric method not implemented yet")

        elif quad_method == "topological":
            # 位相幾何変換（Phase 3 advanced）
            # ホモトピー群やファイバー束理論ベース
            raise NotImplementedError("Topological method not implemented yet")

        else:
            raise ValueError(f"Unknown quad_method: {quad_method}")

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        隠れ状態を4つの思考ロールに分解

        Args:
            hidden: (batch, seq, hidden_dim)

        Returns:
            {
              "observation": (batch, seq, hidden_dim),
              "deduction":   (batch, seq, hidden_dim),
              "abduction":   (batch, seq, hidden_dim),
              "integration": (batch, seq, hidden_dim),
            }
        """
        if self.quad_method == "linear":
            # 線形射影 + Tanh活性化
            x = self.proj(hidden)  # (B, S, 4*H)
            x = self.act(x)        # ソフトな分離
            obs, ded, abd, integ = torch.chunk(x, 4, dim=-1)

        else:
            raise NotImplementedError(f"Method {self.quad_method} not implemented")

        return {
            "observation": obs,    # 観測・知覚
            "deduction":   ded,    # 演繹・論理推論
            "abduction":   abd,    # 仮説形成・帰納
            "integration": integ,  # 統合・合成
        }

    def get_quad_similarity(self) -> Dict[str, float]:
        """
        4ロール間の類似度を監視
        Phase 2.5 デバッグ用
        """
        if not hasattr(self, 'proj'):
            return {}

        W = self.proj.weight  # (4*H, H)
        quad_dim = self.hidden_dim

        similarities = {}
        for i, role1 in enumerate(['obs', 'ded', 'abd', 'integ']):
            for j, role2 in enumerate(['obs', 'ded', 'abd', 'integ']):
                if i < j:
                    w1 = W[i*quad_dim:(i+1)*quad_dim]  # (H, H)
                    w2 = W[j*quad_dim:(j+1)*quad_dim]  # (H, H)

                    # コサイン類似度
                    cos_sim = torch.nn.functional.cosine_similarity(
                        w1.flatten(), w2.flatten(), dim=0
                    ).item()

                    similarities[f'{role1}_{role2}'] = cos_sim

        return similarities


def attach_quad_reasoning_head(model, target_layers: Optional[list] = None):
    """
    モデルにQuadReasoningHeadを注入

    Phase 2.5 統合時に使用予定。
    現在はまだ学習損失に組み込まず、推論時のみ使用。

    Args:
        model: 対象モデル
        target_layers: 注入対象層（None=全層）
    """
    print("🧠 Attaching Quad Reasoning Head (Phase 2.5 preparation)...")

    # モデル構造解析
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        # LoRA適用後モデル
        layers = model.base_model.model.base_model.layers
        hidden_size = model.base_model.model.config.hidden_size
    elif hasattr(model, "base_model"):
        # 通常のPhi-3モデル
        layers = model.base_model.layers
        hidden_size = model.config.hidden_size
    else:
        raise ValueError("Unknown model structure")

    num_layers = len(layers)

    # ターゲット層決定
    if target_layers is None:
        target_layers = list(range(num_layers))  # 全層

    injected_count = 0
    for i in target_layers:
        if i >= num_layers:
            continue

        layer = layers[i]

        # QuadReasoningHead注入
        if not hasattr(layer, 'quad_reasoning_head'):
            quad_head = QuadReasoningHead(hidden_size)
            layer.add_module('quad_reasoning_head', quad_head)
            injected_count += 1

    print(f"[OK] Injected Quad Reasoning Heads into {injected_count} layers")

    # 推論専用なので勾配は切っておく（Phase 2.5 統合まで）
    for name, param in model.named_parameters():
        if 'quad_reasoning_head' in name:
            param.requires_grad = False

    return model


def extract_quad_reasoning(hidden_states: torch.Tensor,
                          quad_head: QuadReasoningHead) -> Dict[str, torch.Tensor]:
    """
    隠れ状態から4つの思考ロールを抽出

    Phase 2.5 の/think APIで使用予定

    Args:
        hidden_states: (batch, seq, hidden)
        quad_head: QuadReasoningHeadインスタンス

    Returns:
        4ロールの思考辞書
    """
    return quad_head(hidden_states)


# Phase 2.5 統合用のユーティリティ関数群
def create_quad_thinking_prompt(base_prompt: str) -> str:
    """
    Quadruple Inference 用のプロンプト生成

    Phase 2.5 で使用予定
    """
    quad_prompt = f"""
<think-observation>
観測フェーズ：{base_prompt}
現在の状況を観測し、利用可能な情報を整理する。
</think-observation>

<think-deduction>
演繹フェーズ：論理的推論に基づいて結論を導く。
</think-deduction>

<think-abduction>
仮説形成フェーズ：可能な説明や仮説を生成する。
</think-abduction>

<think-integration>
統合フェーズ：全ての思考を統合し、最終回答を生成する。
</think-integration>

<final>
"""

    return quad_prompt


def format_quad_response(quad_outputs: Dict[str, torch.Tensor],
                        tokenizer) -> str:
    """
    Quadruple Inference の出力を整形

    Phase 2.5 の/think APIレスポンス用
    """
    # 各ロールの出力をデコード（簡易実装）
    response_parts = []

    for role, hidden in quad_outputs.items():
        # 最後のトークンだけデコード（簡易版）
        last_token = hidden[:, -1:, :]  # (B, 1, H)
        # ここではダミーとしてロール名だけ返す
        response_parts.append(f"<think-{role}>[{role} thinking output]</think-{role}>")

    final_response = "\n".join(response_parts) + "\n<final>[final answer]</final>"

    return final_response


if __name__ == "__main__":
    # テスト実行
    print("🧠 Quad Reasoning Head Test")

    # ダミーデータ
    batch_size, seq_len, hidden_dim = 2, 10, 1024
    hidden = torch.randn(batch_size, seq_len, hidden_dim)

    # QuadReasoningHeadテスト
    quad_head = QuadReasoningHead(hidden_dim)

    quad_output = quad_head(hidden)
    print(f"Input shape: {hidden.shape}")
    print(f"Output keys: {list(quad_output.keys())}")
    for role, tensor in quad_output.items():
        print(f"  {role}: {tensor.shape}")

    # 類似度チェック
    similarities = quad_head.get_quad_similarity()
    print(f"Role similarities: {similarities}")

    print("[OK] Quad Reasoning Head test passed!")
# -*- coding: utf-8 -*-
"""
Quadruple Reasoning Head - Phase 2.5 準備

Observation / Deduction / Abduction / Integration
の4つのロールにhiddenを分解するヘッド。

Phase 2.5 (Quadruple Inference Integration) に向けて、
SO(8)回転エンジンの上で動く「思考の4ロール分解」器を準備。
"""

import torch
from torch import nn
from typing import Dict, Optional


class QuadReasoningHead(nn.Module):
    """
    NKAT理論に基づく4重推論ヘッド

    Observation / Deduction / Abduction / Integration
    の4つのロールにhidden_statesを分解する。

    特徴:
        - Phase 2.5 向けの思考ロール分離
        - SO(8)幾何変換との統合前提
        - Hookベース注入との互換性
        - 学習初期は線形分離のみ（安全策）
    """

    def __init__(self, hidden_dim: int, quad_method: str = "linear"):
        """
        Args:
            hidden_dim: 隠れ層次元数
            quad_method: 分解方法 ("linear", "so8_geometric", "topological")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.quad_method = quad_method

        if quad_method == "linear":
            # 線形分離（Phase 2.5 初期）
            self.proj = nn.Linear(hidden_dim, 4 * hidden_dim)
            self.act = nn.Tanh()  # ソフトな分離

        elif quad_method == "so8_geometric":
            # SO(8)幾何変換（Phase 3）
            # ここにSO(8)回転行列を使った分離を実装
            raise NotImplementedError("SO(8) geometric method not implemented yet")

        elif quad_method == "topological":
            # 位相幾何変換（Phase 3 advanced）
            # ホモトピー群やファイバー束理論ベース
            raise NotImplementedError("Topological method not implemented yet")

        else:
            raise ValueError(f"Unknown quad_method: {quad_method}")

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        隠れ状態を4つの思考ロールに分解

        Args:
            hidden: (batch, seq, hidden_dim)

        Returns:
            {
              "observation": (batch, seq, hidden_dim),
              "deduction":   (batch, seq, hidden_dim),
              "abduction":   (batch, seq, hidden_dim),
              "integration": (batch, seq, hidden_dim),
            }
        """
        if self.quad_method == "linear":
            # 線形射影 + Tanh活性化
            x = self.proj(hidden)  # (B, S, 4*H)
            x = self.act(x)        # ソフトな分離
            obs, ded, abd, integ = torch.chunk(x, 4, dim=-1)

        else:
            raise NotImplementedError(f"Method {self.quad_method} not implemented")

        return {
            "observation": obs,    # 観測・知覚
            "deduction":   ded,    # 演繹・論理推論
            "abduction":   abd,    # 仮説形成・帰納
            "integration": integ,  # 統合・合成
        }

    def get_quad_similarity(self) -> Dict[str, float]:
        """
        4ロール間の類似度を監視
        Phase 2.5 デバッグ用
        """
        if not hasattr(self, 'proj'):
            return {}

        W = self.proj.weight  # (4*H, H)
        quad_dim = self.hidden_dim

        similarities = {}
        for i, role1 in enumerate(['obs', 'ded', 'abd', 'integ']):
            for j, role2 in enumerate(['obs', 'ded', 'abd', 'integ']):
                if i < j:
                    w1 = W[i*quad_dim:(i+1)*quad_dim]  # (H, H)
                    w2 = W[j*quad_dim:(j+1)*quad_dim]  # (H, H)

                    # コサイン類似度
                    cos_sim = torch.nn.functional.cosine_similarity(
                        w1.flatten(), w2.flatten(), dim=0
                    ).item()

                    similarities[f'{role1}_{role2}'] = cos_sim

        return similarities


def attach_quad_reasoning_head(model, target_layers: Optional[list] = None):
    """
    モデルにQuadReasoningHeadを注入

    Phase 2.5 統合時に使用予定。
    現在はまだ学習損失に組み込まず、推論時のみ使用。

    Args:
        model: 対象モデル
        target_layers: 注入対象層（None=全層）
    """
    print("🧠 Attaching Quad Reasoning Head (Phase 2.5 preparation)...")

    # モデル構造解析
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        # LoRA適用後モデル
        layers = model.base_model.model.base_model.layers
        hidden_size = model.base_model.model.config.hidden_size
    elif hasattr(model, "base_model"):
        # 通常のPhi-3モデル
        layers = model.base_model.layers
        hidden_size = model.config.hidden_size
    else:
        raise ValueError("Unknown model structure")

    num_layers = len(layers)

    # ターゲット層決定
    if target_layers is None:
        target_layers = list(range(num_layers))  # 全層

    injected_count = 0
    for i in target_layers:
        if i >= num_layers:
            continue

        layer = layers[i]

        # QuadReasoningHead注入
        if not hasattr(layer, 'quad_reasoning_head'):
            quad_head = QuadReasoningHead(hidden_size)
            layer.add_module('quad_reasoning_head', quad_head)
            injected_count += 1

    print(f"[OK] Injected Quad Reasoning Heads into {injected_count} layers")

    # 推論専用なので勾配は切っておく（Phase 2.5 統合まで）
    for name, param in model.named_parameters():
        if 'quad_reasoning_head' in name:
            param.requires_grad = False

    return model


def extract_quad_reasoning(hidden_states: torch.Tensor,
                          quad_head: QuadReasoningHead) -> Dict[str, torch.Tensor]:
    """
    隠れ状態から4つの思考ロールを抽出

    Phase 2.5 の/think APIで使用予定

    Args:
        hidden_states: (batch, seq, hidden)
        quad_head: QuadReasoningHeadインスタンス

    Returns:
        4ロールの思考辞書
    """
    return quad_head(hidden_states)


# Phase 2.5 統合用のユーティリティ関数群
def create_quad_thinking_prompt(base_prompt: str) -> str:
    """
    Quadruple Inference 用のプロンプト生成

    Phase 2.5 で使用予定
    """
    quad_prompt = f"""
<think-observation>
観測フェーズ：{base_prompt}
現在の状況を観測し、利用可能な情報を整理する。
</think-observation>

<think-deduction>
演繹フェーズ：論理的推論に基づいて結論を導く。
</think-deduction>

<think-abduction>
仮説形成フェーズ：可能な説明や仮説を生成する。
</think-abduction>

<think-integration>
統合フェーズ：全ての思考を統合し、最終回答を生成する。
</think-integration>

<final>
"""

    return quad_prompt


def format_quad_response(quad_outputs: Dict[str, torch.Tensor],
                        tokenizer) -> str:
    """
    Quadruple Inference の出力を整形

    Phase 2.5 の/think APIレスポンス用
    """
    # 各ロールの出力をデコード（簡易実装）
    response_parts = []

    for role, hidden in quad_outputs.items():
        # 最後のトークンだけデコード（簡易版）
        last_token = hidden[:, -1:, :]  # (B, 1, H)
        # ここではダミーとしてロール名だけ返す
        response_parts.append(f"<think-{role}>[{role} thinking output]</think-{role}>")

    final_response = "\n".join(response_parts) + "\n<final>[final answer]</final>"

    return final_response


if __name__ == "__main__":
    # テスト実行
    print("🧠 Quad Reasoning Head Test")

    # ダミーデータ
    batch_size, seq_len, hidden_dim = 2, 10, 1024
    hidden = torch.randn(batch_size, seq_len, hidden_dim)

    # QuadReasoningHeadテスト
    quad_head = QuadReasoningHead(hidden_dim)

    quad_output = quad_head(hidden)
    print(f"Input shape: {hidden.shape}")
    print(f"Output keys: {list(quad_output.keys())}")
    for role, tensor in quad_output.items():
        print(f"  {role}: {tensor.shape}")

    # 類似度チェック
    similarities = quad_head.get_quad_similarity()
    print(f"Role similarities: {similarities}")

    print("[OK] Quad Reasoning Head test passed!")
