import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
import types

class SO8ResidualAdapter(nn.Module):
    """
    NKAT理論に基づく SO(8) 残差回転アダプター (Mixed Precision Robust Ver.)
    重要パラメータ(Lie Algebra, Alpha)をFP32で保持し、NaNを防ぐ。
    """

    def __init__(self, hidden_size: int, so8_dim: int = 8, alpha_init: float = 1e-4):
        super().__init__()
        self.hidden_size = hidden_size
        self.so8_dim = so8_dim

        # 射影層 (ここはモデル本体と同じ精度 FP16/BF16 でOK)
        self.down_proj = nn.Linear(hidden_size, so8_dim, bias=False)
        self.up_proj = nn.Linear(so8_dim, hidden_size, bias=False)

        # ★ 重要: 計算精度が必要なパラメータは FP32 で定義 ★
        self.lie_algebra = nn.Parameter(torch.zeros(so8_dim, so8_dim, dtype=torch.float32))
        self.alpha_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # 初期化
        self.alpha_init_val = alpha_init
        self.reset_parameters()  # PyTorch標準メソッド名に合わせる

    def _init_weights(self):
        # Linear層の初期化
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

        # Lie Algebra (FP32): ゼロ初期化で安全に (NaNを防ぐ)
        with torch.no_grad():
            self.lie_algebra.zero_()

        # Alpha Logit (FP32)
        # alpha = 1.5 * sigmoid(x) - 0.5
        # target = -0.4 (安全な範囲で始める)
        target = -0.4  # -0.5だとp=0になってlog(0)でエラー
        p = max(1e-7, min(1-1e-7, (target + 0.5) / 1.5))  # 安全にclamp
        init_logit = math.log(p / (1.0 - p))
        with torch.no_grad():
            self.alpha_logit.fill_(init_logit)

    # ★ Pytorchの .to() や .half() で FP32 が壊れないようにガード ★
    def _apply(self, fn):
        super()._apply(fn)
        # 強制的に FP32 に戻す (NaNチェックも追加)
        if self.lie_algebra.dtype != torch.float32:
            self.lie_algebra.data = self.lie_algebra.data.to(dtype=torch.float32)
        # NaNが発生していたらゼロにリセット
        if torch.isnan(self.lie_algebra).any():
            self.lie_algebra.data.zero_()

        if self.alpha_logit.dtype != torch.float32:
            self.alpha_logit.data = self.alpha_logit.data.to(dtype=torch.float32)
        # NaNが発生していたら安全値にリセット
        if torch.isnan(self.alpha_logit):
            init_val = math.log(0.5 / 0.5)  # sigmoid(0) = 0.5, alpha = 0
            self.alpha_logit.data.fill_(init_val)

        return self

    @property
    def log_alpha(self):
        return self.alpha_logit

    def get_rotation_matrix(self):
        # lie_algebra は FP32 なのでそのまま計算
        A = self.lie_algebra

        # 完全にゼロの場合は単位行列を返す (初期状態)
        if torch.allclose(A, torch.zeros_like(A), atol=1e-8):
            return torch.eye(self.so8_dim, device=A.device, dtype=torch.float32)

        skew = A - A.T

        # 安全装置
        skew = torch.nan_to_num(skew, nan=0.0)

        # ノルムが大きすぎる場合はクリッピング
        norm = torch.norm(skew)
        if norm > 1.0:  # より保守的な閾値
            skew = skew * (1.0 / norm)

        # 行列指数関数 (FP32)
        R = torch.matrix_exp(skew)

        if torch.isnan(R).any():
            return torch.eye(self.so8_dim, device=skew.device, dtype=torch.float32)
        return R

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 入力 x は FP16 かもしれない
        orig_dtype = x.dtype

        # 1. Down (FP16演算)
        z = self.down_proj(x) # [B, S, 8]

        # 2. Rotate (FP32で回して精度良く回すのが理想だが、速度重視なら R をキャスト)
        # z を FP32 にして精度良く回すのが理想だが、速度重視なら R をキャスト
        # ここでは精度重視で z を FP32 にする
        z_fp32 = z.to(torch.float32)
        R = self.get_rotation_matrix() # [8, 8] FP32
        z_rot = F.linear(z_fp32, R) # [B, S, 8] FP32

        # 3. Up (元の型に戻して演算)
        z_rot_cast = z_rot.to(orig_dtype)
        delta = self.up_proj(z_rot_cast) # [B, S, H]

        # 4. Alpha (FP32 -> キャスト)
        alpha_raw = torch.sigmoid(self.alpha_logit)
        alpha = (1.5 * alpha_raw - 0.5).to(orig_dtype)

        # Residual Add
        return x + alpha * delta

    def get_orthogonality_error(self):
        with torch.no_grad():
            R = self.get_rotation_matrix()
            I = torch.eye(self.so8_dim, device=R.device)
            err = torch.norm(R.T @ R - I)
        return err.item()

    def get_adapter_stats(self):
        with torch.no_grad():
            alpha_raw = torch.sigmoid(self.alpha_logit)
            alpha = 1.5 * alpha_raw - 0.5
            ortho_err = self.get_orthogonality_error()
            lie_norm = torch.norm(self.lie_algebra).item()

            # NaN安全策
            if math.isnan(lie_norm):
                lie_norm = 0.0
            if math.isnan(alpha.item()):
                alpha = torch.tensor(0.0)

        return {
            'alpha': alpha.item(),
            'orthogonality_error': ortho_err,
            'lie_norm': lie_norm,
            'lie_algebra_norm': lie_norm
        }

def monkey_patch_unsloth_layers(model, target_layers="middle"):
    print("🧬 Injecting NKAT SO(8) Adapters (Robust Mixed-Precision Mode)...")

    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base_entity = model.base_model.model
        if hasattr(base_entity, "model") and hasattr(base_entity.model, "layers"):
            layers = base_entity.model.layers
        elif hasattr(base_entity, "layers"):
            layers = base_entity.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        # 最後の手段：名前で検索
        print("⚠️ Layer attribute not found standardly. Searching by name...")
        layers = None
        for name, module in model.named_modules():
            if name.endswith("layers"):
                layers = module
                break
        if layers is None:
             raise ValueError("Could not find layers.")



    hidden_size = model.config.hidden_size
    num_layers = len(layers)

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
        if not hasattr(layer, "mlp"): continue
        target_module = layer.mlp
        if hasattr(target_module, "nkat_adapter"): continue

        sample_param = next(target_module.parameters())

        # ★ 修正: deviceだけ指定し、dtypeキャストは adapter 内の _apply で制御される
        adapter = SO8ResidualAdapter(hidden_size).to(sample_param.device)
        # Linear層だけキャスト
        adapter.down_proj.to(sample_param.dtype)
        adapter.up_proj.to(sample_param.dtype)

        target_module.nkat_adapter = adapter
        target_module.add_module("nkat_adapter", adapter)

        original_forward = target_module.forward

        def new_mlp_forward(self, x):
            output = original_forward(x)
            if output.requires_grad is False and torch.is_grad_enabled():
                output.requires_grad_(True)
            nkat_out = self.nkat_adapter(output)
            return output + nkat_out

        target_module.forward = types.MethodType(new_mlp_forward, target_module)
        injected_count += 1

    print(f"✅ Monkey Patched {injected_count} MLPs.")

    # 勾配有効化
    enabled_count = 0
    for name, param in model.named_parameters():
        if "nkat_adapter" in name or "lie_algebra" in name or "alpha_logit" in name:
            param.requires_grad = True
            enabled_count += 1

    print(f"🔥 Force-enabled gradients for {enabled_count} SO(8) parameters")

    return model


class NKATMLPWrapper(nn.Module):
    """
    既存のMLPをラップし、NKATアダプタを追加する正規のnn.Module
    Monkey PatchよりもPyTorchとの親和性が高く、確実に勾配を通す。
    """
    def __init__(self, original_mlp, adapter):
        super().__init__()
        self.original_mlp = original_mlp
        self.nkat_adapter = adapter

    def forward(self, x):
        # 1. 元のMLP実行 (Unslothの最適化カーネルが走る)
        output = self.original_mlp(x)

        # 2. 勾配の呼び水 (Checkpointing対策)
        if output.requires_grad is False and torch.is_grad_enabled():
            output.requires_grad_(True)

        # 3. アダプタ適用 (残差)
        nkat_out = self.nkat_adapter(output)

        return output + nkat_out


def replace_mlp_with_nkat(model, target_layers="middle"):
    print("🧬 Injecting NKAT SO(8) Adapters (Module Replacement Mode)...")

    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base_entity = model.base_model.model
        if hasattr(base_entity, "model") and hasattr(base_entity.model, "layers"):
            layers = base_entity.model.layers
        elif hasattr(base_entity, "layers"):
            layers = base_entity.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        # 最後の手段：名前で検索
        print("⚠️ Layer attribute not found standardly. Searching by name...")
        layers = None
        for name, module in model.named_modules():
            if name.endswith("layers"):
                layers = module
                break
        if layers is None:
             raise ValueError("Could not find layers.")

    hidden_size = model.config.hidden_size
    num_layers = len(layers)

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
        if not hasattr(layer, "mlp"): continue

        # 既にラップ済みならスキップ
        if isinstance(layer.mlp, NKATMLPWrapper): continue

        original_mlp = layer.mlp

        # アダプタ作成
        sample_param = next(original_mlp.parameters())
        adapter = SO8ResidualAdapter(hidden_size).to(sample_param.device)
        # Linear層の型合わせ
        adapter.down_proj.to(sample_param.dtype)
        adapter.up_proj.to(sample_param.dtype)

        # ★★★ ここが変更点: ラッパーで物理的に置き換える ★★★
        wrapper = NKATMLPWrapper(original_mlp, adapter)

        # レイヤーの属性を上書き
        layer.mlp = wrapper

        injected_count += 1

    print(f"✅ Replaced {injected_count} MLPs with NKAT Wrappers.")

    # 勾配有効化
    trainable_count = 0
    for name, param in model.named_parameters():
        if "nkat_adapter" in name or "lie_algebra" in name or "alpha_logit" in name:
            param.requires_grad = True
            trainable_count += 1

    print(f"🔥 Force-enabled gradients for {trainable_count} SO(8) parameters")

    return model