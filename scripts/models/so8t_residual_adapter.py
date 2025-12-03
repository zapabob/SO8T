import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
import types

class SO8ResidualAdapter(nn.Module):
    """
    NKAT理論に基づく SO(8) 残差回転アダプター (Sigmoid Alpha + 互換性維持Ver.)
    """
    def __init__(self, hidden_size: int, so8_dim: int = 8, alpha_init: float = 1e-4):
        super().__init__()
        self.hidden_size = hidden_size
        self.so8_dim = so8_dim

        self.down_proj = nn.Linear(hidden_size, so8_dim, bias=False)
        self.up_proj = nn.Linear(so8_dim, hidden_size, bias=False)
        self.lie_algebra = nn.Parameter(torch.zeros(so8_dim, so8_dim))

        # ★ Sigmoid用 Logit (初期値計算)
        # alpha = 1.5 * sigmoid(x) - 0.5 の逆関数で初期値を決める
        # alpha_init = -0.1 くらいから始めたい
        target_alpha = -0.1 
        # p = (target_alpha + 0.5) / 1.5 = 0.4 / 1.5 = 0.266...
        p = (target_alpha + 0.5) / 1.5
        init_logit = math.log(p / (1.0 - p))
        
        self.alpha_logit = nn.Parameter(torch.tensor(float(init_logit)))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.normal_(self.lie_algebra, std=1e-3)  # 少し大きくしてNaNを防ぐ
        # alpha_logit は計算済みなのでそのままでOK

    # ★★★ 互換性レイヤー (ここがエラー回避の鍵！) ★★★
    @property
    def log_alpha(self):
        """
        古いコード(logger)が log_alpha を参照した時にエラーにならないようにする。
        本来は log(alpha) を返すべきだが、ログ出力用なので alpha_logit をそのまま返す。
        """
        return self.alpha_logit

    def get_rotation_matrix(self):
        A = self.lie_algebra
        skew = A - A.T
        skew = torch.nan_to_num(skew, nan=0.0)
        skew_fp32 = skew.to(torch.float32)
        
        norm = torch.norm(skew_fp32)
        if norm > 5.0:
            skew_fp32 = skew_fp32 * (5.0 / norm)
            
        R_fp32 = torch.matrix_exp(skew_fp32)
        
        if torch.isnan(R_fp32).any():
            return torch.eye(self.so8_dim, device=skew.device, dtype=torch.float32)
        return R_fp32.to(A.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        
        if torch.isnan(x_fp32).any():
            return x

        z = self.down_proj(x_fp32)
        R = self.get_rotation_matrix()
        z_rot = F.linear(z, R)
        delta = self.up_proj(z_rot)

        # ★ アニーリング付き Alpha (-0.5 ~ 1.0)
        alpha_raw = torch.sigmoid(self.alpha_logit)
        alpha = 1.5 * alpha_raw - 0.5

        if torch.isnan(delta).any():
            delta = torch.zeros_like(delta)

        out = x_fp32 + alpha * delta
        
        if torch.isnan(out).any():
            return x

        return out.to(orig_dtype)

    def get_orthogonality_error(self):
        with torch.no_grad():
            R = self.get_rotation_matrix()
            I = torch.eye(self.so8_dim, device=R.device)
            err = torch.norm(R.T @ R - I)
        return err.item()

    def get_adapter_stats(self):
        """デバッグ用統計情報"""
        with torch.no_grad():
            # Alpha計算
            alpha_raw = torch.sigmoid(self.alpha_logit)
            alpha = 1.5 * alpha_raw - 0.5

            ortho_err = self.get_orthogonality_error()
            lie_norm = torch.norm(self.lie_algebra).item()

        return {
            'alpha': alpha.item(),
            'orthogonality_error': ortho_err,
            'lie_norm': lie_norm,          # ★これが必要やったんや！
            'lie_algebra_norm': lie_norm   # こっちも念のため残す
        }


# ==========================================
# Monkey Patch Function (これも必須！)
# ==========================================
def monkey_patch_unsloth_layers(model, target_layers="middle"):
    print("🧬 Injecting NKAT SO(8) Adapters (Safe MLP Patch Mode)...")

    # ★★★ 修正ポイント: 堅牢なレイヤー探索ロジック ★★★
    layers = None

    # Case 1: Unsloth / PEFT Wrapped Model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        # ここで model.base_model.model は 'Phi3ForCausalLM' 等の実体
        base_entity = model.base_model.model

        if hasattr(base_entity, "model") and hasattr(base_entity.model, "layers"):
            # Phi-3, Llama, Mistral (Deep structure)
            # Peft -> Base -> ForCausalLM -> Model -> Layers
            layers = base_entity.model.layers
        elif hasattr(base_entity, "layers"):
            # GPT-NeoX など (Shallow structure)
            layers = base_entity.layers

    # Case 2: Standard HF Model (No PEFT)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers

    # Case 3: Bare Model
    elif hasattr(model, "layers"):
        layers = model.layers

    # エラーハンドリング
    if layers is None:
        print("❌ [ERROR] Could not find 'layers' attribute in the model!")
        print(f"Model structure: {type(model)}")
        if hasattr(model, "base_model"):
            print(f"Base model: {type(model.base_model)}")
        raise ValueError("Unknown model structure: Cannot inject adapters.")

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
        
        if not hasattr(layer, "mlp"):
            continue
            
        target_module = layer.mlp
        
        if hasattr(target_module, "nkat_adapter"):
            continue

        sample_param = next(target_module.parameters())
        adapter = SO8ResidualAdapter(hidden_size).to(sample_param.device).to(sample_param.dtype)
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