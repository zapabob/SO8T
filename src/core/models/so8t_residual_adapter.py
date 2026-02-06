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

    def reset_parameters(self):
        # Linear層の初期化
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

        # Lie Algebra: 少し大きめにして「0じゃない」ことを確認しやすくする
        nn.init.normal_(self.lie_algebra, std=1e-4)

        # Alpha Logit
        target = -0.1
        p = (target + 0.5) / 1.5
        init_logit = math.log(p / (1.0 - p))
        with torch.no_grad():
            self.alpha_logit.fill_(init_logit)

        # ★念押しでデータ型強制★
        self.lie_algebra.data = self.lie_algebra.data.float()
        self.alpha_logit.data = self.alpha_logit.data.float()

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
    print("[SO8T] Injecting NKAT SO(8) Adapters (Robust Mixed-Precision Mode)...")

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
        print("[WARN] Layer attribute not found standardly. Searching by name...")
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

    print(f"[OK] Monkey Patched {injected_count} MLPs.")

    # 勾配有効化
    enabled_count = 0
    for name, param in model.named_parameters():
        if "nkat_adapter" in name or "lie_algebra" in name or "alpha_logit" in name:
            param.requires_grad = True
            enabled_count += 1

    print(f"[HOT] Force-enabled gradients for {enabled_count} SO(8) parameters")

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


class NKATLayerWrapper(nn.Module):
    """
    完全なTransformer層をラップし、すべてのサブコンポーネントにNKATアダプターを適用
    Attention + MLP + 残差接続のすべてにSO(8)変換を適用
    """
    def __init__(self, original_layer, adapters):
        super().__init__()
        self.original_layer = original_layer
        self.adapters = adapters  # {'attention': adapter, 'mlp': adapter, 'residual': adapter}

        # 元のコンポーネントを保存
        self.input_layernorm = original_layer.input_layernorm
        self.self_attn = original_layer.self_attn
        self.post_attention_layernorm = getattr(original_layer, 'post_attention_layernorm', None)
        self.mlp = original_layer.mlp
        self.resid_attn_dropout = getattr(original_layer, 'resid_attn_dropout', None)
        self.resid_mlp_dropout = getattr(original_layer, 'resid_mlp_dropout', None)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        residual = hidden_states

        # 1. Input LayerNorm + Attention
        if self.input_layernorm is not None:
            hidden_states = self.input_layernorm(hidden_states)

        # Attention with NKAT adapter
        attn_output = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        if isinstance(attn_output, tuple):
            hidden_states, attn_weights = attn_output
        else:
            hidden_states = attn_output
            attn_weights = None

        # NKAT adapter on attention output
        if 'attention' in self.adapters and self.adapters['attention'] is not None:
            if hidden_states.requires_grad is False and torch.is_grad_enabled():
                hidden_states.requires_grad_(True)
            attn_nkat = self.adapters['attention'](hidden_states)
            hidden_states = hidden_states + attn_nkat

        # Residual connection + dropout
        if self.resid_attn_dropout is not None:
            hidden_states = self.resid_attn_dropout(hidden_states)
        hidden_states = residual + hidden_states

        # 2. Post-Attention LayerNorm + MLP
        residual = hidden_states

        if self.post_attention_layernorm is not None:
            hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP with NKAT adapter
        mlp_output = self.mlp(hidden_states)

        # NKAT adapter on MLP output
        if 'mlp' in self.adapters and self.adapters['mlp'] is not None:
            if mlp_output.requires_grad is False and torch.is_grad_enabled():
                mlp_output.requires_grad_(True)
            mlp_nkat = self.adapters['mlp'](mlp_output)
            mlp_output = mlp_output + mlp_nkat

        # Residual connection + dropout
        if self.resid_mlp_dropout is not None:
            mlp_output = self.resid_mlp_dropout(mlp_output)
        hidden_states = residual + mlp_output

        # 3. Final residual NKAT adapter (layer全体の出力に適用)
        if 'residual' in self.adapters and self.adapters['residual'] is not None:
            if hidden_states.requires_grad is False and torch.is_grad_enabled():
                hidden_states.requires_grad_(True)
            residual_nkat = self.adapters['residual'](hidden_states)
            hidden_states = hidden_states + residual_nkat

        if output_attentions and attn_weights is not None:
            return (hidden_states, attn_weights)
        return hidden_states


def replace_mlp_with_nkat(model, target_layers="middle"):
    """後方互換性のための関数 - MLPのみ適用"""
    print("[SO8T] Injecting NKAT SO(8) Adapters (MLP Only Mode - Legacy)...")
    return inject_nkat_to_all_layers(model, target_layers, mode="mlp_only")


def inject_nkat_to_all_layers(model, target_layers="all", mode="full_layer"):
    """
    Transformerのすべての層にNKATアダプターを注入
    mode: "mlp_only" - MLPのみ, "full_layer" - すべてのコンポーネント
    """
    print(f"[SO8T] Injecting NKAT SO(8) Adapters (Mode: {mode})...")

    # モデル構造の探索
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
        print("[WARN] Layer attribute not found standardly. Searching by name...")
        layers = None
        for name, module in model.named_modules():
            if name.endswith("layers"):
                layers = module
                break
        if layers is None:
            raise ValueError("Could not find layers.")

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

    print(f"Targeting layers: {list(target_indices)} (Total: {num_layers})")

    injected_count = 0
    adapter_count = 0

    for i in target_indices:
        layer = layers[i]

        if mode == "mlp_only":
            # 従来のMLPのみモード
            if not hasattr(layer, "mlp"): continue
            if isinstance(layer.mlp, NKATMLPWrapper): continue

            original_mlp = layer.mlp
            sample_param = next(original_mlp.parameters())
            adapter = SO8ResidualAdapter(hidden_size).to(sample_param.device)
            adapter.down_proj.to(sample_param.dtype)
            adapter.up_proj.to(sample_param.dtype)

            wrapper = NKATMLPWrapper(original_mlp, adapter)
            layer.mlp = wrapper
            injected_count += 1
            adapter_count += 1

        elif mode == "full_layer":
            # 完全層モード - Attention + MLP + Residualすべてに適用
            if isinstance(layer, NKATLayerWrapper): continue

            # 各コンポーネント用のアダプター作成
            adapters = {}
            sample_param = next(layer.parameters()) if list(layer.parameters()) else None
            if sample_param is None:
                continue

            # Attention出力用アダプター
            if hasattr(layer, 'self_attn'):
                adapters['attention'] = SO8ResidualAdapter(hidden_size).to(sample_param.device)
                adapters['attention'].down_proj.to(sample_param.dtype)
                adapters['attention'].up_proj.to(sample_param.dtype)
                adapter_count += 1

            # MLP出力用アダプター
            if hasattr(layer, 'mlp'):
                adapters['mlp'] = SO8ResidualAdapter(hidden_size).to(sample_param.device)
                adapters['mlp'].down_proj.to(sample_param.dtype)
                adapters['mlp'].up_proj.to(sample_param.dtype)
                adapter_count += 1

            # 層全体のResidual用アダプター
            adapters['residual'] = SO8ResidualAdapter(hidden_size).to(sample_param.device)
            adapters['residual'].down_proj.to(sample_param.dtype)
            adapters['residual'].up_proj.to(sample_param.dtype)
            adapter_count += 1

            # 完全ラッパーで層を置き換え
            wrapper = NKATLayerWrapper(layer, adapters)
            layers[i] = wrapper
            injected_count += 1

    print(f"[OK] Injected NKAT adapters to {injected_count} layers ({adapter_count} total adapters)")

    # 勾配有効化
    trainable_count = 0
    for name, param in model.named_parameters():
        if "nkat_adapter" in name or "lie_algebra" in name or "alpha_logit" in name:
            param.requires_grad = True
            trainable_count += 1

    print(f"[HOT] Force-enabled gradients for {trainable_count} SO(8) parameters")
    print(f"[TARGET] Mode: {mode} - {'MLP only' if mode == 'mlp_only' else 'Full layer coverage'}")

    return model