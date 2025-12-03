# so8t_residual_adapter.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple

class SO8ResidualAdapter(nn.Module):
    """
    NKAT理論に基づく SO(8) 残差回転アダプター

    Inputs: [batch, seq_len, hidden_size]
    Outputs: inputs + alpha * Up( R( Down(inputs) ) )

    特徴:
        - Lie Algebra (so(8)) による厳密な直交回転行列の生成 (Matrix Exponential)
        - Log-space Alpha による学習安定化
        - Hookベース注入に最適化
    """
    def __init__(self, hidden_size: int, so8_dim: int = 8, alpha_init: float = 0.01):
        super().__init__()

        self.hidden_size = hidden_size
        self.so8_dim = so8_dim

        # 1. 次元圧縮 (Down-projection)
        # バイアスなし（原点中心の回転対称性を保つため）
        self.down_proj = nn.Linear(hidden_size, so8_dim, bias=False)

        # 2. SO(8) 回転生成元 (Lie Algebra generator)
        # 反対称行列 A (A^T = -A) をパラメータとして持つ
        self.lie_algebra = nn.Parameter(torch.zeros(so8_dim, so8_dim))

        # 3. 次元復元 (Up-projection)
        self.up_proj = nn.Linear(so8_dim, hidden_size, bias=False)

        # 4. 混合率 alpha (学習可能)
        # 負の値にならないよう、実体は log(alpha) で持つ
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha_init)))

        # 初期化
        self._init_weights()

    def _init_weights(self):
        # Down: Kaiming初期化 (分散を保つ)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))

        # Up: ゼロ初期化 (学習初期はベースモデルの挙動を阻害しない -> 重要！)
        nn.init.zeros_(self.up_proj.weight)

        # Lie Algebra: 小さなランダム値 (初期はほぼ恒等回転)
        nn.init.normal_(self.lie_algebra, std=0.001)

    def get_rotation_matrix(self):
        """リー代数から回転行列 R = exp(A - A^T) を生成"""
        # A を反対称化 (Skew-symmetric)
        A = self.lie_algebra
        skew = A - A.T
        # 行列指数関数 (Matrix Exponential)
        # これで数学的に厳密な回転行列が得られる
        R = torch.matrix_exp(skew)
        return R

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection
        x: [batch_size, seq_len, hidden_size]
        """
        # 型変換 (FP16/BF16対応)
        # 回転精度確保のため一時的にFP32推奨だが、VRAM節約のため入力型に合わせる
        dtype = x.dtype
        x_in = x

        # 1. Down: [B, S, H] -> [B, S, 8]
        z = self.down_proj(x_in)

        # 2. Rotate: z . R^T
        R = self.get_rotation_matrix().to(dtype)
        z_rot = F.linear(z, R)

        # 3. Up: [B, S, 8] -> [B, S, H]
        delta = self.up_proj(z_rot)

        # 4. Residual Add
        alpha = torch.exp(self.log_alpha)

        # ★★★ ここが重要：In-place加算 (+=) は勾配計算でエラーになりやすいので避ける ★★★
        out = x_in + alpha * delta

        return out

    def get_orthogonality_error(self):
        """直交性誤差モニタリング (R^T R - I)"""
        with torch.no_grad():
            R = self.get_rotation_matrix()
            I = torch.eye(self.so8_dim, device=R.device)
            err = torch.norm(R.T @ R - I)
        return err.item()

def attach_nkat_adapters(model, target_layers: Optional[Union[List[int], str]] = "middle"):
    """
    Unsloth/HFモデルにSO(8)アダプターをHookとして注入する関数
    """
    print("🧬 Injecting NKAT SO(8) Adapters (Hook Mode)...")

    # モデル構造の解析
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "base_model") and hasattr(model.base_model.model.base_model, "layers"):
        # LoRA適用後のPhi-3モデル (PeftModel -> LoraModel -> Phi3ForCausalLM -> Phi3Model -> layers)
        layers = model.base_model.model.base_model.layers
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
        # LoRA適用後のUnslothモデルなど
        layers = model.base_model.model.layers
    elif hasattr(model, "base_model") and hasattr(model.base_model, "layers"):
        # Phi-3 モデル (base_model が Phi3Model の場合)
        layers = model.base_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # 通常のHFモデル
        layers = model.model.layers
    else:
        # 特殊構造対応
        if hasattr(model, "layers"):
             layers = model.layers
        else:
             raise ValueError("Unknown model structure: Cannot find 'layers' attribute.")

    hidden_size = model.config.hidden_size
    num_layers = len(layers)

    # ターゲット層の決定
    if target_layers == "all":
        target_indices = range(num_layers)
    elif target_layers == "middle":
        start = num_layers // 4
        end = num_layers * 3 // 4
        target_indices = range(start, end)
    elif isinstance(target_layers, list):
        target_indices = target_layers
    else:
        target_indices = range(num_layers)

    print(f"Targeting layers: {list(target_indices)}")

    injected_count = 0
    for i in target_indices:
        layer = layers[i]

        # 既に注入済みならスキップ
        if hasattr(layer, "nkat_adapter"):
            print(f"Layer {i} already has NKAT adapter.")
            continue

        # アダプター作成
        # デバイスと型をモデルに合わせる
        # 注意: Unslothのレイヤーは独自クラスの場合があるので、パラメータから取得
        sample_param = next(layer.parameters())
        device = sample_param.device
        dtype = sample_param.dtype

        adapter = SO8ResidualAdapter(hidden_size).to(device).to(dtype)

        # パラメータとして登録
        layer.add_module("nkat_adapter", adapter)

        # Forward Hook の定義
        # ★★★ 勾配問題を解決する安全なHook ★★★
        def nkat_hook(module, input, output):
            # output は通常 (hidden_states, ...) のタプル
            # hidden_states は計算グラフの一部である必要がある
            if isinstance(output, tuple):
                hidden_states = output[0]
                # アダプター適用 (新たなTensorが生成され、グラフが分岐・合流する)
                new_hidden = module.nkat_adapter(hidden_states)
                return (new_hidden,) + output[1:]
            elif isinstance(output, torch.Tensor):
                return module.nkat_adapter(output)
            else:
                return output

        # Hook登録
        layer.register_forward_hook(nkat_hook)
        injected_count += 1

    print(f"✅ Successfully injected NKAT adapters into {injected_count} layers.")

    # 勾配有効化 (重要！)
    for name, param in model.named_parameters():
        if "nkat_adapter" in name:
            param.requires_grad = True

    return model