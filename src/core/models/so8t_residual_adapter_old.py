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
        - Phase 2.5: 四重推論統合
    """
    def __init__(self, hidden_size: int, so8_dim: int = 8, alpha_init: float = 0.1,
                 enable_quad_inference: bool = False):
        super().__init__()

        self.hidden_size = hidden_size
        self.so8_dim = so8_dim
        self.enable_quad_inference = enable_quad_inference

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

        # Phase 2.5: Quadruple inference components
        if enable_quad_inference:
            from src.models.quad_reasoning_head import QuadReasoningHead
            self.quad_reasoning_head = QuadReasoningHead(hidden_size)
            self.quad_integration_proj = nn.Linear(4 * hidden_size, hidden_size)

        # 学習しやすいよう、alphaパラメータに特別な初期化
        torch.nn.init.normal_(self.log_alpha, mean=0.0, std=0.1)

        # 初期化
        self._init_weights()

    def _init_weights(self):
        # Down: Kaiming初期化 (分散を保つ)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))

        # Up: 小さなランダム初期化 (学習初期に適度な影響を与える)
        nn.init.normal_(self.up_proj.weight, std=0.01)

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

        # Phase 2.5: Apply quadruple inference
        if self.enable_quad_inference and hasattr(self, 'quad_reasoning_head'):
            quad_outputs = self.quad_reasoning_head(x)
            # Integrate the four reasoning outputs
            integrated_quad_output = torch.cat(
                [quad_outputs["observation"], quad_outputs["deduction"],
                 quad_outputs["abduction"], quad_outputs["integration"]], dim=-1
            )
            integrated_quad_output = self.quad_integration_proj(integrated_quad_output)
            delta = delta + integrated_quad_output  # Add to the SO(8) rotated output

        # ★★★ ここが重要：In-place加算 (+=) は勾配計算でエラーになりやすいので避ける ★★★
        out = x_in + alpha * delta

        return out

    def get_orthogonality_error(self):
        """直交性誤差モニタリング (R^T R - I)"""
        with torch.no_grad():
            R = self.get_rotation_matrix()
            I = torch.eye(self.so8_dim, device=R.device, dtype=R.dtype)
            err = torch.norm(R.T @ R - I)
        return err.item()

    def get_adapter_stats(self):
        """アダプター統計情報取得（デバッグ用）"""
        with torch.no_grad():
            alpha = torch.exp(self.log_alpha).item()
            ortho_err = self.get_orthogonality_error()
            down_norm = torch.norm(self.down_proj.weight).item()
            up_norm = torch.norm(self.up_proj.weight).item()
            lie_norm = torch.norm(self.lie_algebra).item()

        return {
            'alpha': alpha,
            'orthogonality_error': ortho_err,
            'down_proj_norm': down_norm,
            'up_proj_norm': up_norm,
            'lie_algebra_norm': lie_norm
        }

def attach_nkat_adapters(model, target_layers: Optional[Union[List[int], str]] = "middle"):
    """
    Unsloth/HFモデルにSO(8)アダプターをHookとして注入する関数
    """
    print("[SO8T] Injecting NKAT SO(8) Adapters (Hook Mode)...")

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

    print(f"[OK] Successfully injected NKAT adapters into {injected_count} layers.")

    # 勾配有効化 (重要！)
    for name, param in model.named_parameters():
        if "nkat_adapter" in name:
            param.requires_grad = True

    return model
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

    print(f"[OK] Successfully injected NKAT adapters into {injected_count} layers.")

    # 勾配有効化 (重要！)
    for name, param in model.named_parameters():
        if "nkat_adapter" in name:
            param.requires_grad = True

    return model
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

    print(f"[OK] Successfully injected NKAT adapters into {injected_count} layers.")

    # 勾配有効化 (重要！)
    for name, param in model.named_parameters():
        if "nkat_adapter" in name:
            param.requires_grad = True

    return model
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

    print(f"[OK] Successfully injected NKAT adapters into {injected_count} layers.")

    # 勾配有効化 (重要！)
    for name, param in model.named_parameters():
        if "nkat_adapter" in name:
            param.requires_grad = True

    return model
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

    print(f"[OK] Successfully injected NKAT adapters into {injected_count} layers.")

    # 勾配有効化 (重要！)
    for name, param in model.named_parameters():
        if "nkat_adapter" in name:
            param.requires_grad = True

    return model
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

    print(f"[OK] Successfully injected NKAT adapters into {injected_count} layers.")

    # 勾配有効化 (重要！)
    for name, param in model.named_parameters():
        if "nkat_adapter" in name:
            param.requires_grad = True

    return model
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

    print(f"[OK] Successfully injected NKAT adapters into {injected_count} layers.")

    # 勾配有効化 (重要！)
    for name, param in model.named_parameters():
        if "nkat_adapter" in name:
            param.requires_grad = True

    return model
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

    print(f"[OK] Successfully injected NKAT adapters into {injected_count} layers.")

    # 勾配有効化 (重要！)
    for name, param in model.named_parameters():
        if "nkat_adapter" in name:
            param.requires_grad = True

    return model
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

    print(f"[OK] Successfully injected NKAT adapters into {injected_count} layers.")

    # 勾配有効化 (重要！)
    for name, param in model.named_parameters():
        if "nkat_adapter" in name:
            param.requires_grad = True

    return model
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

    print(f"[OK] Successfully injected NKAT adapters into {injected_count} layers.")

    # 勾配有効化 (重要！)
    for name, param in model.named_parameters():
        if "nkat_adapter" in name:
            param.requires_grad = True

    return model
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

    print(f"[OK] Successfully injected NKAT adapters into {injected_count} layers.")

    # 勾配有効化 (重要！)
    for name, param in model.named_parameters():
        if "nkat_adapter" in name:
            param.requires_grad = True

    return model