"""
Safety-Aware SO8T Model Implementation

既存CausalLM（Qwen2, Llama, Mistral等）をベースに拡張し、以下を実装:
1. 四成分表現空間分割: [h^(V), h^(S+), h^(S-), h^(Ver)]
2. 厳密なSO(8)群回転ゲート（右側からの作用）
3. 幾何学的制約（ノルム・直交性・等長性）
4. Safety Head（Spinor+成分からALLOW/ESCALATE/REFUSE 3分類）
5. Verifier Head（Verifier成分から自己検証スコア）
6. PET正則化（既存SO8TGroupStructureと簡易PETのハイブリッド）
"""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Literal

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

from .strict_so8_rotation_gate import StrictSO8RotationGate

# 既存実装のインポート
import sys
from pathlib import Path
# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
try:
    from models.so8t_group_structure import SO8TGroupStructure, PETRegularization
except ImportError:
    # フォールバック: 簡易実装を使用
    class SO8TGroupStructure(nn.Module):
        def __init__(self, hidden_size: int, lambda_pet: float):
            super().__init__()
            self.hidden_size = hidden_size
            self.lambda_pet = lambda_pet
        
        def compute_pet_loss(self, hidden_states: torch.Tensor) -> torch.Tensor:
            if hidden_states.size(1) < 3:
                return torch.tensor(0.0, device=hidden_states.device)
            d2 = hidden_states[:, :-2] - 2 * hidden_states[:, 1:-1] + hidden_states[:, 2:]
            return self.lambda_pet * torch.mean(d2 ** 2)


@dataclass
class SafetyAwareSO8TConfig:
    """
    Safety-Aware SO8T Model Configuration
    
    四成分表現空間分割と幾何学的制約のハイパーパラメータを含む。
    """
    # 安全ヘッドのクラス数: 0=ALLOW, 1=ESCALATE, 2=REFUSE
    num_safety_labels: int = 3
    
    # 検証ヘッドの次元数（例: overall_confidenceのみなら1）
    num_verifier_dims: int = 1
    
    # 四成分分割の次元設定
    # hidden_size = d_V + d_S_plus + d_S_minus + d_Ver を満たす必要がある
    d_V: Optional[int] = None  # Vectorロール（タスク出力）
    d_S_plus: Optional[int] = None  # Spinor+ロール（安全・倫理）
    d_S_minus: Optional[int] = None  # Spinor-ロール（エスカレーション・慎重側）
    d_Ver: Optional[int] = None  # Verifierロール（自己検証）
    # 自動分割の場合の比率（合計が1.0になる必要がある）
    role_split_ratio: Tuple[float, float, float, float] = (0.5, 0.2, 0.15, 0.15)
    
    # PET（二階差分ペナルティ）の係数
    pet_lambda: float = 0.1
    
    # 損失の重み
    alpha_safety: float = 2.0
    beta_danger_penalty: float = 8.0
    gamma_safe_allow_reward: float = 1.0
    delta_escalate_penalty: float = 0.5
    
    # 幾何学的制約の重み
    mu_norm: float = 0.01  # ノルム制約
    nu_orth: float = 0.01  # 直交性制約
    rho_iso: float = 0.01  # 等長性制約
    
    # ターゲットノルム（各ロール成分の期待ノルム）
    c_V: float = 1.0
    c_S_plus: float = 1.0
    c_S_minus: float = 1.0
    c_Ver: float = 1.0
    
    # 危険サンプルでのラベル値（データ側と合わせておく）
    dangerous_label_ids: Tuple[int, ...] = (2,)  # REFUSEが正解なのにALLOWを出したら危険
    
    # EasyケースでESCALATEし過ぎを軽く罰したい場合のEasyラベル
    easy_label_ids: Tuple[int, ...] = (0,)  # ALLOWが正なケース
    
    # Safety Gate推論用しきい値
    safety_conf_threshold: float = 0.7
    
    # Verifier出力の有無
    use_verifier_head: bool = True
    
    # PETをどの層で計算するか: "last" or "all"
    pet_mode: Literal["last", "all"] = "last"
    
    # SO(8)回転ゲートの設定
    use_strict_so8_rotation: bool = True  # 厳密なSO(8)回転ゲートを使用
    so8_use_cayley: bool = True  # Cayley変換を使用
    so8_orthogonal_reg: float = 1e-3  # 直交性正則化
    
    # SO(8)回転ゲートの適用レイヤー設定（LLMベストプラクティス: 中間レイヤーのみ）
    so8_apply_to_intermediate_layers: bool = True  # 中間レイヤーのみに適用（True）または最終層のみ（False）
    so8_intermediate_layer_start: Optional[int] = None  # 中間レイヤーの開始インデックス（None: 自動計算）
    so8_intermediate_layer_end: Optional[int] = None  # 中間レイヤーの終了インデックス（None: 自動計算）
    so8_intermediate_layer_ratio: Tuple[float, float] = (0.25, 0.75)  # 中間レイヤーの範囲（開始比率、終了比率）
    
    # 直交誤差測定とログ出力
    so8_log_orthogonal_error: bool = True  # 直交誤差をログ出力するか
    so8_orthogonal_error_threshold: float = 1e-3  # 直交誤差の警告閾値
    
    # PET正則化の強化（高周波成分カット）
    pet_apply_to_intermediate_layers: bool = True  # 中間レイヤーにもPET正則化を適用
    pet_high_freq_cutoff: float = 0.5  # 高周波成分カットオフ（0.0-1.0、高いほど強いカット）
    
    # Alpha Gate設定（黄金比の逆数の二乗: Φ^(-2) = 0.432）
    use_alpha_gate: bool = True  # Alpha Gateを使用するか
    alpha_gate_target: float = 0.432  # ターゲット値: Φ^(-2) = (1/1.618)^2 ≈ 0.382, ユーザー指定: 0.432
    alpha_gate_start: float = -5.0  # 初期値（Chaos状態）
    alpha_gate_annealing_steps: int = 1000  # アニーリングステップ数
    alpha_gate_steepness: float = 12.0  # シグモイドアニーリングの急激さ
    alpha_gate_orthogonal_weight: float = 1.0  # 直交誤差の重み（ベイズ最適化で調整）
    alpha_gate_pet_weight: float = 0.1  # PET正則化の重み（ベイズ最適化で調整）
    
    def compute_role_dimensions(self, hidden_size: int) -> Tuple[int, int, int, int]:
        """
        四成分分割の次元を計算
        
        Args:
            hidden_size: ベースモデルの隠れ次元
        
        Returns:
            (d_V, d_S_plus, d_S_minus, d_Ver): 各ロールの次元
        """
        if all(x is not None for x in [self.d_V, self.d_S_plus, self.d_S_minus, self.d_Ver]):
            # 明示的に指定されている場合
            total = self.d_V + self.d_S_plus + self.d_S_minus + self.d_Ver
            if total != hidden_size:
                raise ValueError(
                    f"Sum of role dimensions ({total}) must equal hidden_size ({hidden_size})"
                )
            return (self.d_V, self.d_S_plus, self.d_S_minus, self.d_Ver)
        else:
            # 比率から自動計算
            r_V, r_S_plus, r_S_minus, r_Ver = self.role_split_ratio
            total_ratio = r_V + r_S_plus + r_S_minus + r_Ver
            if abs(total_ratio - 1.0) > 1e-6:
                raise ValueError(f"role_split_ratio must sum to 1.0, got {total_ratio}")
            
            d_V = int(hidden_size * r_V)
            d_S_plus = int(hidden_size * r_S_plus)
            d_S_minus = int(hidden_size * r_S_minus)
            d_Ver = hidden_size - d_V - d_S_plus - d_S_minus  # 残りをVerifierに
            
            return (d_V, d_S_plus, d_S_minus, d_Ver)


class GeometricConstraints(nn.Module):
    """
    幾何学的制約モジュール
    
    ノルム制約、直交性制約、等長性制約を計算する。
    """
    
    def __init__(self, config: SafetyAwareSO8TConfig):
        super().__init__()
        self.config = config
    
    def compute_norm_constraint(
        self,
        h_V: torch.Tensor,
        h_S_plus: torch.Tensor,
        h_S_minus: torch.Tensor,
        h_Ver: torch.Tensor,
    ) -> torch.Tensor:
        """
        ノルム制約損失
        L_norm = Σ_{r ∈ {V,S+,S-,Ver}} (E[||h^(r)||^2] - c_r)^2
        
        Args:
            h_V: [B, T, d_V] Vectorロール成分
            h_S_plus: [B, T, d_S_plus] Spinor+ロール成分
            h_S_minus: [B, T, d_S_minus] Spinor-ロール成分
            h_Ver: [B, T, d_Ver] Verifierロール成分
        
        Returns:
            loss: スカラーテンソル
        """
        # 各ロール成分のノルムを計算
        norm_V = torch.mean(h_V ** 2)
        norm_S_plus = torch.mean(h_S_plus ** 2)
        norm_S_minus = torch.mean(h_S_minus ** 2)
        norm_Ver = torch.mean(h_Ver ** 2)
        
        # ターゲットノルムとの差の二乗
        loss = (
            (norm_V - self.config.c_V) ** 2 +
            (norm_S_plus - self.config.c_S_plus) ** 2 +
            (norm_S_minus - self.config.c_S_minus) ** 2 +
            (norm_Ver - self.config.c_Ver) ** 2
        )
        
        return loss
    
    def compute_orthogonality_constraint(
        self,
        h_V: torch.Tensor,
        h_S_plus: torch.Tensor,
        h_S_minus: torch.Tensor,
        h_Ver: torch.Tensor,
    ) -> torch.Tensor:
        """
        直交性制約損失
        L_orth = Σ_{(r,s), r≠s} (E[<h^(r), h^(s)>])^2
        
        Args:
            h_V: [B, T, d_V] Vectorロール成分
            h_S_plus: [B, T, d_S_plus] Spinor+ロール成分
            h_S_minus: [B, T, d_S_minus] Spinor-ロール成分
            h_Ver: [B, T, d_Ver] Verifierロール成分
        
        Returns:
            loss: スカラーテンソル
        """
        # 各ロール成分をフラット化
        h_V_flat = h_V.view(-1, h_V.size(-1))  # [B*T, d_V]
        h_S_plus_flat = h_S_plus.view(-1, h_S_plus.size(-1))  # [B*T, d_S_plus]
        h_S_minus_flat = h_S_minus.view(-1, h_S_minus.size(-1))  # [B*T, d_S_minus]
        h_Ver_flat = h_Ver.view(-1, h_Ver.size(-1))  # [B*T, d_Ver]
        
        # 次元を揃えるため、最小次元に合わせる
        min_dim = min(h_V.size(-1), h_S_plus.size(-1), h_S_minus.size(-1), h_Ver.size(-1))
        
        # 各ロール成分を最小次元に射影（平均プーリング）
        if h_V.size(-1) > min_dim:
            h_V_proj = h_V_flat[:, :min_dim]
        else:
            h_V_proj = h_V_flat
        if h_S_plus.size(-1) > min_dim:
            h_S_plus_proj = h_S_plus_flat[:, :min_dim]
        else:
            h_S_plus_proj = h_S_plus_flat
        if h_S_minus.size(-1) > min_dim:
            h_S_minus_proj = h_S_minus_flat[:, :min_dim]
        else:
            h_S_minus_proj = h_S_minus_flat
        if h_Ver.size(-1) > min_dim:
            h_Ver_proj = h_Ver_flat[:, :min_dim]
        else:
            h_Ver_proj = h_Ver_flat
        
        # ロール間の内積を計算
        inner_V_S_plus = torch.mean(torch.sum(h_V_proj * h_S_plus_proj, dim=-1))
        inner_V_S_minus = torch.mean(torch.sum(h_V_proj * h_S_minus_proj, dim=-1))
        inner_V_Ver = torch.mean(torch.sum(h_V_proj * h_Ver_proj, dim=-1))
        inner_S_plus_S_minus = torch.mean(torch.sum(h_S_plus_proj * h_S_minus_proj, dim=-1))
        inner_S_plus_Ver = torch.mean(torch.sum(h_S_plus_proj * h_Ver_proj, dim=-1))
        inner_S_minus_Ver = torch.mean(torch.sum(h_S_minus_proj * h_Ver_proj, dim=-1))
        
        # 内積の二乗和
        loss = (
            inner_V_S_plus ** 2 +
            inner_V_S_minus ** 2 +
            inner_V_Ver ** 2 +
            inner_S_plus_S_minus ** 2 +
            inner_S_plus_Ver ** 2 +
            inner_S_minus_Ver ** 2
        )
        
        return loss
    
    def compute_isometry_constraint(
        self,
        weight_matrices: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        等長性制約損失
        L_iso = Σ_{r ∈ {S+,S-,Ver}} ||W^(r)^T W^(r) - I||_F^2
        
        Args:
            weight_matrices: ロール空間に作用する重み行列の辞書
                            {'S_plus': W_S_plus, 'S_minus': W_S_minus, 'Ver': W_Ver}
        
        Returns:
            loss: スカラーテンソル
        """
        if weight_matrices is None:
            return torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else torch.device('cpu'))
        
        loss = torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else torch.device('cpu'))
        
        for role_name, W in weight_matrices.items():
            if W is None:
                continue
            
            # W^T @ W
            W_T_W = torch.matmul(W.transpose(-1, -2), W)
            
            # I
            I = torch.eye(W.size(-1), device=W.device, dtype=W.dtype)
            if len(W.shape) > 2:
                I = I.unsqueeze(0).expand_as(W_T_W)
            
            # ||W^T W - I||_F^2
            loss = loss + torch.mean((W_T_W - I) ** 2)
        
        return loss


class SafetyAwareSO8TModel(PreTrainedModel):
    """
    Safety-Aware SO8T Model
    
    既存 CausalLM モデルに以下を追加:
    - 四成分表現空間分割: [h^(V), h^(S+), h^(S-), h^(Ver)]
    - 厳密なSO(8)群回転ゲート（右側からの作用）
    - Safety head: ALLOW / ESCALATE / REFUSE の3分類
    - Verifier head: 自己検証用スコア（任意）
    - PET: 二階差分正則化
    - 幾何学的制約: ノルム・直交性・等長性制約
    """
    
    config_class = AutoConfig
    
    def __init__(
        self, 
        base_model_name_or_path: str, 
        so8t_config: SafetyAwareSO8TConfig,
        quantization_config: Optional[Any] = None
    ):
        # ベースモデル読み込み
        base_config = AutoConfig.from_pretrained(base_model_name_or_path)
        super().__init__(base_config)
        
        self.so8t_cfg = so8t_config
        
        # 内部にベースのCausalLMを保持
        model_kwargs = {
            "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            # 8bit量子化を使用する場合、device_map="auto"が必要
            model_kwargs["device_map"] = "auto"
            # GPUメモリ使用量を制限（RTX3080は10GB）
            if torch.cuda.is_available():
                # メモリの80%を使用、20%をバッファとして確保
                max_memory = {0: "8GiB"}  # RTX3080は10GBなので8GBに制限
                model_kwargs["max_memory"] = max_memory
        else:
            # 量子化を使用しない場合も、メモリ効率のためにdevice_mapを設定
            # ただし、device_map="auto"は量子化なしでは問題を起こす可能性があるため、CPUに読み込んでからGPUに移動
            if torch.cuda.is_available():
                # まずCPUに読み込んでからGPUに移動（メモリ効率のため）
                model_kwargs["device_map"] = None  # device_mapをNoneにして、後で手動でGPUに移動
                # または、device_map="cpu"でCPUに読み込んでからGPUに移動
                # model_kwargs["device_map"] = "cpu"
        
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[MODEL] Loading base model with kwargs: {list(model_kwargs.keys())}")
            if "max_memory" in model_kwargs:
                logger.info(f"[MODEL] Max memory: {model_kwargs['max_memory']}")
            if "quantization_config" in model_kwargs:
                logger.info(f"[MODEL] Using quantization: {type(model_kwargs['quantization_config']).__name__}")
        except:
            pass
        
        try:
            logger.info(f"[MODEL] Starting model loading from: {base_model_name_or_path}")
            self.base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                **model_kwargs
            )
            # 量子化なしの場合、CPUに読み込んだ後GPUに移動
            # ただし、GPUへの移動でクラッシュする可能性があるため、環境変数で制御可能にする
            force_cpu = os.environ.get("SO8T_FORCE_CPU", "false").lower() == "true"
            
            if quantization_config is None and torch.cuda.is_available() and model_kwargs.get("device_map") is None and not force_cpu:
                logger.info("[MODEL] Attempting to move model to GPU...")
                try:
                    # CUDAメモリをクリア
                    torch.cuda.empty_cache()
                    # モデルサイズを確認
                    if hasattr(self.base_model, 'get_memory_footprint'):
                        memory_footprint = self.base_model.get_memory_footprint()
                        logger.info(f"[MODEL] Model memory footprint: {memory_footprint / 1024**3:.2f} GB")
                    
                    # GPUへの移動を段階的に行う（エラーハンドリング付き）
                    logger.info("[MODEL] Moving model to GPU (this may take a while)...")
                    # まず、モデルを評価モードにしてメモリ使用量を削減
                    self.base_model.eval()
                    # GPUへの移動
                    self.base_model = self.base_model.to("cuda")
                    logger.info("[MODEL] Model successfully moved to GPU")
                    
                    # メモリ使用状況を確認
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    reserved = torch.cuda.memory_reserved(0) / 1024**3
                    logger.info(f"[CUDA] Memory after model load - allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")
                except (RuntimeError, Exception) as e:
                    logger.error(f"[ERROR] Failed to move model to GPU: {e}")
                    logger.warning("[WARNING] Model will remain on CPU (may cause performance issues)")
                    logger.warning("[WARNING] Set SO8T_FORCE_CPU=true to skip GPU move attempt")
                    # GPUへの移動に失敗した場合、CPUのまま続行
                    torch.cuda.empty_cache()
                    # モデルをCPUに明示的に設定
                    self.base_model = self.base_model.to("cpu")
            elif force_cpu:
                logger.info("[MODEL] SO8T_FORCE_CPU=true, keeping model on CPU")
                self.base_model = self.base_model.to("cpu")
            logger.info("[MODEL] Base model loaded successfully")
            
            # base_model読み込み後、レイヤー数を確定して中間レイヤー範囲を再計算
            if self.num_layers is None:
                if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
                    self.num_layers = len(self.base_model.model.layers)
                elif hasattr(self.base_model, 'config') and hasattr(self.base_model.config, 'num_hidden_layers'):
                    self.num_layers = self.base_model.config.num_hidden_layers
                else:
                    # デフォルト値（Phi-3.5-miniは32層）
                    self.num_layers = 32
                    logger.warning(f"[SO8T] Could not determine num_layers, using default: {self.num_layers}")
            
            # 中間レイヤー範囲を再計算
            if self.so8t_cfg.so8_apply_to_intermediate_layers:
                if self.so8_layer_start is None:
                    self.so8_layer_start = int(self.num_layers * self.so8t_cfg.so8_intermediate_layer_ratio[0])
                if self.so8_layer_end is None:
                    self.so8_layer_end = int(self.num_layers * self.so8t_cfg.so8_intermediate_layer_ratio[1])
                
                logger.info(f"[SO8T] SO(8) rotation gate will be applied to intermediate layers: {self.so8_layer_start}-{self.so8_layer_end} (total {self.num_layers} layers)")
        except RuntimeError as e:
            # CUDAメモリをクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import traceback
            error_msg = f"RuntimeError while loading base model: {e}\n{traceback.format_exc()}"
            logger.error(f"[ERROR] {error_msg}")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            # CUDAメモリをクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import traceback
            error_msg = f"Unexpected error while loading base model: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"[ERROR] {error_msg}")
            raise RuntimeError(error_msg) from e
        
        hidden_size = base_config.hidden_size
        
        # 四成分分割の次元を計算
        self.d_V, self.d_S_plus, self.d_S_minus, self.d_Ver = so8t_config.compute_role_dimensions(hidden_size)
        
        # ベースモデルのレイヤー数を取得（中間レイヤー選択用）
        # base_modelがPhi-3.5などの場合、model.layersの数を取得
        self.num_layers = getattr(base_config, 'num_hidden_layers', None)
        if self.num_layers is None:
            # フォールバック: base_modelから直接取得（まだbase_modelが読み込まれていない場合は後で設定）
            self.num_layers = None
        
        # 中間レイヤーの範囲を計算（base_model読み込み後に再設定）
        if so8t_config.so8_apply_to_intermediate_layers:
            if so8t_config.so8_intermediate_layer_start is not None:
                self.so8_layer_start = so8t_config.so8_intermediate_layer_start
            else:
                # デフォルト: 25%-75%の範囲（後でnum_layersが確定したら再計算）
                self.so8_layer_start = None
            
            if so8t_config.so8_intermediate_layer_end is not None:
                self.so8_layer_end = so8t_config.so8_intermediate_layer_end
            else:
                # デフォルト: 25%-75%の範囲（後でnum_layersが確定したら再計算）
                self.so8_layer_end = None
        else:
            self.so8_layer_start = None
            self.so8_layer_end = None
        
        # 厳密なSO(8)回転ゲート（右側からの作用）
        if so8t_config.use_strict_so8_rotation:
            self.so8_rotation_gate = StrictSO8RotationGate(
                hidden_size=hidden_size,
                use_cayley=so8t_config.so8_use_cayley,
                orthogonal_regularization=so8t_config.so8_orthogonal_reg,
            )
        else:
            self.so8_rotation_gate = None
        
        # 設定を保存（ログ出力用）
        self.so8t_cfg = so8t_config
        
        # Alpha Gateパラメータ（学習可能パラメータとして初期化）
        if so8t_config.use_alpha_gate:
            # Alpha Gateの初期値: シグモイド(-5.0) ≈ 0.0067（Chaos状態）
            # ターゲット: シグモイド(α) = 0.432 となるようなαを計算
            # sigmoid(α) = 0.432 → α = logit(0.432) = log(0.432 / (1 - 0.432)) ≈ -0.28
            # しかし、アニーリングで-5.0から開始するため、初期値は-5.0
            self.alpha_gate = nn.Parameter(torch.tensor(so8t_config.alpha_gate_start))
            self.alpha_gate_step = 0  # アニーリングステップカウンター
            self.alpha_gate_target = so8t_config.alpha_gate_target
            self.alpha_gate_start = so8t_config.alpha_gate_start
            self.alpha_gate_annealing_steps = so8t_config.alpha_gate_annealing_steps
            self.alpha_gate_steepness = so8t_config.alpha_gate_steepness
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"[ALPHA_GATE] Initialized: start={so8t_config.alpha_gate_start}, "
                f"target={so8t_config.alpha_gate_target}, "
                f"annealing_steps={so8t_config.alpha_gate_annealing_steps}, "
                f"steepness={so8t_config.alpha_gate_steepness}"
            )
        else:
            self.alpha_gate = None
        
        # Safety head: Spinor+成分から3分類
        self.safety_head = nn.Sequential(
            nn.Linear(self.d_S_plus, self.d_S_plus // 2),
            nn.GELU(),
            nn.Linear(self.d_S_plus // 2, so8t_config.num_safety_labels),
        )
        
        # Verifier head（任意）
        if so8t_config.use_verifier_head:
            self.verifier_head = nn.Sequential(
                nn.Linear(self.d_Ver, self.d_Ver // 2),
                nn.GELU(),
                nn.Linear(self.d_Ver // 2, so8t_config.num_verifier_dims),
            )
        else:
            self.verifier_head = None
        
        # 幾何学的制約モジュール
        self.geometric_constraints = GeometricConstraints(so8t_config)
        
        # 既存のSO8TGroupStructure（PET損失計算用）
        self.group_structure = SO8TGroupStructure(
            hidden_size=hidden_size,
            lambda_pet=so8t_config.pet_lambda,
        )
        
        # 重み初期化（base_modelは既に初期化済みなのでスキップ）
        # post_init()を呼ばずに、手動でSO8T固有のモジュールのみ初期化
        # post_init()はbase_modelも含めて再帰的に初期化しようとするため、
        # base_modelが既に初期化済みの場合に再帰エラーが発生する
        self._init_so8t_weights()
    
    def post_init(self):
        """
        post_init()をオーバーライドして、base_modelの再初期化を防ぐ
        
        PreTrainedModelの__init__が自動的にpost_init()を呼び出すため、
        これをオーバーライドして、SO8T固有のモジュールのみ初期化する。
        """
        # post_init()は既に_init_so8t_weights()で実行済み
        # base_modelの再初期化を防ぐため、何もしない
        pass
    
    def _init_so8t_weights(self):
        """
        SO8T固有のモジュールのみ重み初期化（base_modelはスキップ）
        
        base_modelは既にfrom_pretrained()で初期化されているため、
        再初期化をスキップして再帰エラーを防ぐ。
        """
        # SO8T固有のモジュールのみ初期化
        for module in [self.safety_head, self.verifier_head, self.geometric_constraints]:
            if module is not None:
                self._init_module_weights(module)
        
        # SO8T rotation gateの初期化
        if self.so8_rotation_gate is not None:
            self._init_module_weights(self.so8_rotation_gate)
        
        # Group structureの初期化
        if self.group_structure is not None:
            self._init_module_weights(self.group_structure)
    
    def _init_module_weights(self, module):
        """
        モジュールの重みを初期化（再帰的に、base_modelはスキップ）
        """
        # base_modelはスキップ
        if module is self.base_model:
            return
        
        for child in module.children():
            self._init_module_weights(child)
        
        # リーフモジュールの初期化
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # デフォルトの初期化
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def split_hidden_states(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        隠れ状態を四成分に分割
        
        Args:
            hidden_states: [B, T, hidden_size] 隠れ状態
        
        Returns:
            (h_V, h_S_plus, h_S_minus, h_Ver): 各ロール成分
        """
        B, T, D = hidden_states.shape
        
        # 次元を分割
        h_V = hidden_states[:, :, :self.d_V]
        h_S_plus = hidden_states[:, :, self.d_V:self.d_V + self.d_S_plus]
        h_S_minus = hidden_states[:, :, self.d_V + self.d_S_plus:self.d_V + self.d_S_plus + self.d_S_minus]
        h_Ver = hidden_states[:, :, self.d_V + self.d_S_plus + self.d_S_minus:]
        
        return h_V, h_S_plus, h_S_minus, h_Ver
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        safety_labels: Optional[torch.LongTensor] = None,
        is_easy_case: Optional[torch.BoolTensor] = None,
        is_danger_case: Optional[torch.BoolTensor] = None,
        output_hidden_states: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass
        
        Args:
            input_ids: 入力トークンID
            attention_mask: アテンションマスク
            labels: 通常の言語モデル用ラベル（次トークン予測）
            safety_labels: 各サンプルに対する安全ラベル（0/1/2）
            is_easy_case: Easyケースフラグ（過エスカ抑制用）
            is_danger_case: Hardケースフラグ（危険ALLOW罰用）
            output_hidden_states: 隠れ状態を出力するか
        
        Returns:
            出力辞書
        """
        # ベースモデル実行
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )
        
        lm_loss = outputs.loss
        hidden_states = outputs.hidden_states  # tuple of (layer+emb) [B,T,D]
        
        # SO(8)回転ゲートを中間レイヤーに適用（LLMベストプラクティス）
        # hidden_states[0]はembedding層、hidden_states[1:]は各Transformer層
        processed_hidden_states = list(hidden_states)
        
        if self.so8_rotation_gate is not None and self.so8t_cfg.so8_apply_to_intermediate_layers:
            # 中間レイヤーのみにSO(8)回転ゲートを適用
            # hidden_states[0]はembedding層、hidden_states[1]からがTransformer層
            # インデックス調整: so8_layer_start/endはTransformer層のインデックス（0始まり）
            for layer_idx in range(self.so8_layer_start, min(self.so8_layer_end, len(processed_hidden_states) - 1)):
                # hidden_states[0]はembedding、hidden_states[1]が最初のTransformer層
                # したがって、Transformer層のインデックスlayer_idxに対応するhidden_statesインデックスはlayer_idx + 1
                hidden_idx = layer_idx + 1
                if hidden_idx < len(processed_hidden_states):
                    processed_hidden_states[hidden_idx] = self.so8_rotation_gate(
                        processed_hidden_states[hidden_idx], 
                        apply_right=True
                    )
            
            # 直交誤差の測定とログ出力
            if self.so8t_cfg.so8_log_orthogonal_error:
                import logging
                logger = logging.getLogger(__name__)
                orth_error = self.so8_rotation_gate.get_orthogonality_loss()
                det_error = self.so8_rotation_gate.get_determinant_loss()
                
                if orth_error.item() > self.so8t_cfg.so8_orthogonal_error_threshold:
                    logger.warning(
                        f"[SO8T] High orthogonal error detected: {orth_error.item():.6f} "
                        f"(threshold: {self.so8t_cfg.so8_orthogonal_error_threshold})"
                    )
                else:
                    logger.debug(
                        f"[SO8T] Orthogonal error: {orth_error.item():.6f}, "
                        f"Determinant error: {det_error.item():.6f}"
                    )
        
        # 最終層hidden（[B,T,D]）- 中間レイヤーにSO(8)回転ゲートを適用した後の最終層
        last_hidden = processed_hidden_states[-1]
        
        # 最終層にはSO(8)回転ゲートを適用しない（中間レイヤーのみ適用）
        # これにより、最終層の表現を保持し、四重推論を可能にする
        
        # 四成分に分割
        h_V, h_S_plus, h_S_minus, h_Ver = self.split_hidden_states(last_hidden)
        
        # 各サンプルの「最終トークン」の隠れ状態を代表として使用
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
            pooled_S_plus = h_S_plus[batch_indices, lengths]  # [B, d_S_plus]
            pooled_Ver = h_Ver[batch_indices, lengths]  # [B, d_Ver]
        else:
            pooled_S_plus = h_S_plus[:, -1, :]  # [B, d_S_plus]
            pooled_Ver = h_Ver[:, -1, :]  # [B, d_Ver]
        
        # Safety head: Spinor+成分から3分類
        safety_logits = self.safety_head(pooled_S_plus)  # [B, num_safety_labels]
        
        # Verifier head
        verifier_scores = None
        if self.verifier_head is not None:
            verifier_scores = self.verifier_head(pooled_Ver)  # [B, num_verifier_dims]
        
        # PET損失（高周波成分カット）- 中間レイヤーにも適用（LLMベストプラクティス）
        pet_loss = torch.tensor(0.0, device=last_hidden.device)
        
        if self.so8t_cfg.pet_apply_to_intermediate_layers and self.so8t_cfg.so8_apply_to_intermediate_layers:
            # 中間レイヤーにもPET正則化を適用して高周波成分をカット
            # 高周波成分カットオフ: pet_high_freq_cutoffが高いほど強いカット
            intermediate_pet_losses = []
            for layer_idx in range(self.so8_layer_start, min(self.so8_layer_end, len(processed_hidden_states) - 1)):
                hidden_idx = layer_idx + 1
                if hidden_idx < len(processed_hidden_states):
                    layer_pet_loss = self.group_structure.compute_pet_loss(processed_hidden_states[hidden_idx])
                    # 高周波成分カットオフを適用（重み付け）
                    weight = 1.0 - self.so8t_cfg.pet_high_freq_cutoff
                    intermediate_pet_losses.append(layer_pet_loss * weight)
            
            if intermediate_pet_losses:
                pet_loss = torch.stack(intermediate_pet_losses).mean()
        
        # 最終層のPET損失も追加（既存実装との互換性）
        final_layer_pet_loss = self.group_structure.compute_pet_loss(last_hidden)
        pet_loss = pet_loss + final_layer_pet_loss
        
        # 幾何学的制約損失
        norm_loss = self.geometric_constraints.compute_norm_constraint(
            h_V, h_S_plus, h_S_minus, h_Ver
        )
        orth_loss = self.geometric_constraints.compute_orthogonality_constraint(
            h_V, h_S_plus, h_S_minus, h_Ver
        )
        iso_loss = self.geometric_constraints.compute_isometry_constraint()
        
        # 安全関連損失
        safety_loss, detail = self.compute_safety_losses(
            safety_logits=safety_logits,
            safety_labels=safety_labels,
            is_easy_case=is_easy_case,
            is_danger_case=is_danger_case,
        )
        
        # SO(8)回転ゲートの直交性損失
        so8_orth_loss = torch.tensor(0.0, device=last_hidden.device)
        so8_det_loss = torch.tensor(0.0, device=last_hidden.device)
        if self.so8_rotation_gate is not None:
            so8_orth_loss = self.so8_rotation_gate.get_orthogonality_loss()
            so8_det_loss = self.so8_rotation_gate.get_determinant_loss()
        
        # Alpha Gateアニーリング（シグモイドアニーリングでα=Φ^(-2)=0.432を目標）
        alpha_gate_value = None
        alpha_gate_loss = torch.tensor(0.0, device=last_hidden.device)
        if self.alpha_gate is not None:
            # シグモイドアニーリング: αを段階的に更新
            # 進行度を計算（0.0から1.0）
            progress = min(1.0, self.alpha_gate_step / self.alpha_gate_annealing_steps)
            
            # シグモイド関数で滑らかに遷移: S(x) = 1 / (1 + e^(-k*(x-0.5)))
            # xを-0.5から0.5の範囲に正規化（中心で転移）
            relative_progress = progress - 0.5
            sigmoid_factor = 1 / (1 + math.exp(-self.alpha_gate_steepness * relative_progress))
            
            # Alpha Gate値を更新: start_alphaからtarget_alphaへ
            # ただし、target_alphaはシグモイド後の値（0.432）なので、
            # シグモイド前の値に変換: logit(0.432) ≈ -0.28
            target_alpha_raw = math.log(self.alpha_gate_target / (1 - self.alpha_gate_target))
            current_alpha_raw = self.alpha_gate_start + (target_alpha_raw - self.alpha_gate_start) * sigmoid_factor
            
            # Alpha Gateパラメータを更新（勾配計算のため、detachしてから更新）
            with torch.no_grad():
                self.alpha_gate.data = torch.tensor(current_alpha_raw, device=self.alpha_gate.device, dtype=self.alpha_gate.dtype)
            
            # シグモイド変換後のAlpha Gate値
            alpha_gate_value = torch.sigmoid(self.alpha_gate)
            
            # Alpha Gate損失: ターゲット値（0.432）からの偏差を最小化
            alpha_gate_target_tensor = torch.tensor(self.alpha_gate_target, device=alpha_gate_value.device, dtype=alpha_gate_value.dtype)
            alpha_gate_loss = (alpha_gate_value - alpha_gate_target_tensor) ** 2
            
            # ステップカウンターを更新
            self.alpha_gate_step += 1
        
        # 直交誤差を0に保つための損失（ベイズ最適化で調整される重みを使用）
        orthogonal_error_loss = so8_orth_loss * self.so8t_cfg.alpha_gate_orthogonal_weight
        
        # PET正則化による学習発散防止（ベイズ最適化で調整される重みを使用）
        pet_divergence_loss = pet_loss * self.so8t_cfg.alpha_gate_pet_weight
        
        # 合成損失
        total_loss = None
        if lm_loss is not None:
            total_loss = lm_loss
            if safety_loss is not None:
                total_loss = (
                    total_loss
                    + self.so8t_cfg.alpha_safety * safety_loss
                    + pet_loss  # 既存のPET損失
                    + pet_divergence_loss  # 学習発散防止のための追加PET損失
                    + self.so8t_cfg.mu_norm * norm_loss
                    + self.so8t_cfg.nu_orth * orth_loss
                    + self.so8t_cfg.rho_iso * iso_loss
                    + orthogonal_error_loss  # 直交誤差を0に保つ
                    + so8_det_loss
                    + alpha_gate_loss  # Alpha Gateターゲット損失
                )
            else:
                total_loss = (
                    total_loss
                    + pet_loss  # 既存のPET損失
                    + pet_divergence_loss  # 学習発散防止のための追加PET損失
                    + self.so8t_cfg.mu_norm * norm_loss
                    + self.so8t_cfg.nu_orth * orth_loss
                    + self.so8t_cfg.rho_iso * iso_loss
                    + orthogonal_error_loss  # 直交誤差を0に保つ
                    + so8_det_loss
                    + alpha_gate_loss  # Alpha Gateターゲット損失
                )
        else:
            if safety_loss is not None:
                total_loss = (
                    self.so8t_cfg.alpha_safety * safety_loss
                    + pet_loss  # 既存のPET損失
                    + pet_divergence_loss  # 学習発散防止のための追加PET損失
                    + self.so8t_cfg.mu_norm * norm_loss
                    + self.so8t_cfg.nu_orth * orth_loss
                    + self.so8t_cfg.rho_iso * iso_loss
                    + orthogonal_error_loss  # 直交誤差を0に保つ
                    + so8_det_loss
                    + alpha_gate_loss  # Alpha Gateターゲット損失
                )
            else:
                total_loss = (
                    pet_loss  # 既存のPET損失
                    + pet_divergence_loss  # 学習発散防止のための追加PET損失
                    + self.so8t_cfg.mu_norm * norm_loss
                    + self.so8t_cfg.nu_orth * orth_loss
                    + self.so8t_cfg.rho_iso * iso_loss
                    + orthogonal_error_loss  # 直交誤差を0に保つ
                    + so8_det_loss
                    + alpha_gate_loss  # Alpha Gateターゲット損失
                )
        
        return {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "safety_loss": safety_loss,
            "pet_loss": pet_loss,
            "norm_loss": norm_loss,
            "orth_loss": orth_loss,
            "iso_loss": iso_loss,
            "so8_orth_loss": so8_orth_loss,
            "so8_det_loss": so8_det_loss,
            "alpha_gate_value": alpha_gate_value.item() if alpha_gate_value is not None else None,
            "alpha_gate_loss": alpha_gate_loss.item() if alpha_gate_value is not None else 0.0,
            "orthogonal_error_loss": orthogonal_error_loss.item(),
            "pet_divergence_loss": pet_divergence_loss.item(),
            "safety_logits": safety_logits,
            "verifier_scores": verifier_scores,
            "safety_loss_detail": detail,
            "logits": outputs.logits,
            "hidden_states": hidden_states,
        }
    
    def compute_safety_losses(
        self,
        safety_logits: torch.FloatTensor,
        safety_labels: Optional[torch.LongTensor],
        is_easy_case: Optional[torch.BoolTensor],
        is_danger_case: Optional[torch.BoolTensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        安全関連損失を計算
        
        Args:
            safety_logits: [B, num_safety_labels] 安全分類ロジット
            safety_labels: [B] 安全ラベル
            is_easy_case: [B] Easyケースフラグ
            is_danger_case: [B] Hardケースフラグ
        
        Returns:
            (total_loss, detail_dict)
        """
        detail = {
            "ce": safety_logits.new_tensor(0.0),
            "danger_penalty": safety_logits.new_tensor(0.0),
            "safe_allow_reward": safety_logits.new_tensor(0.0),
            "escalate_penalty": safety_logits.new_tensor(0.0),
        }
        
        if safety_labels is None:
            return None, detail
        
        # 通常のCE
        ce_loss_fct = CrossEntropyLoss()
        ce = ce_loss_fct(safety_logits, safety_labels)
        detail["ce"] = ce
        
        total = ce
        
        # softmax
        probs = safety_logits.softmax(dim=-1)
        pred = probs.argmax(dim=-1)
        
        # 危険ケースでのALLOWに重罰
        if is_danger_case is not None:
            mask = is_danger_case & (pred == 0)  # 0=ALLOW
            if mask.any():
                danger_penalty = probs[mask, 0].mean()
                detail["danger_penalty"] = danger_penalty
                total = total + self.so8t_cfg.beta_danger_penalty * danger_penalty
        
        # Easyケース: ALLOWが正解なのにESCALATE/REFUSEの場合に小罰
        if is_easy_case is not None:
            easy_mask = is_easy_case
            if easy_mask.any():
                esc_or_ref_mask = easy_mask & (pred != 0)
                if esc_or_ref_mask.any():
                    esc_penalty = probs[esc_or_ref_mask, pred[esc_or_ref_mask]].mean()
                    detail["escalate_penalty"] = esc_penalty
                    total = total + self.so8t_cfg.delta_escalate_penalty * esc_penalty
                
                # 安全なALLOWへの微報酬（Loss減算）
                allow_mask = easy_mask & (pred == 0)
                if allow_mask.any():
                    reward = probs[allow_mask, 0].mean()
                    detail["safe_allow_reward"] = reward
                    total = total - self.so8t_cfg.gamma_safe_allow_reward * reward
        
        return total, detail
    
    @torch.no_grad()
    def safety_gate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Dict[str, Any]:
        """
        推論時: Safety headに基づいて ALLOW / ESCALATE / REFUSE を判定
        
        Args:
            input_ids: 入力トークンID
            attention_mask: アテンションマスク
        
        Returns:
            判定結果の辞書
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]
        
        # 厳密なSO(8)回転ゲートを適用
        if self.so8_rotation_gate is not None:
            last_hidden = self.so8_rotation_gate(last_hidden, apply_right=True)
        
        # 四成分に分割
        h_V, h_S_plus, h_S_minus, h_Ver = self.split_hidden_states(last_hidden)
        
        # 最終トークンのSpinor+成分を取得
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
            pooled = h_S_plus[batch_indices, lengths]
        else:
            pooled = h_S_plus[:, -1, :]
        
        safety_logits = self.safety_head(pooled)
        probs = safety_logits.softmax(dim=-1)
        conf, pred = probs.max(dim=-1)
        
        label_map = {0: "ALLOW", 1: "ESCALATE", 2: "REFUSE"}
        decisions = [label_map[int(p)] for p in pred]
        
        # 信頼度が低い場合はESCALATEに倒す（Fail-Closed）
        final_decisions = []
        for d, c in zip(decisions, conf):
            if c.item() < self.so8t_cfg.safety_conf_threshold and d == "ALLOW":
                final_decisions.append("ESCALATE")
            else:
                final_decisions.append(d)
        
        return {
            "safety_logits": safety_logits,
            "safety_probs": probs,
            "raw_decisions": decisions,
            "decisions": final_decisions,
            "confidence": conf,
        }
    
    @torch.no_grad()
    def export_to_standard_format(self) -> Dict[str, torch.Tensor]:
        """
        既存エコシステム互換形式にエクスポート
        
        SO(8)回転ゲートを重み行列に吸収し、標準的な重み形式に変換する。
        
        Returns:
            エクスポートされた重み行列の辞書
        """
        if self.so8_rotation_gate is None:
            # SO(8)回転ゲートがない場合は、そのまま返す
            return {}
        
        exported_weights = {}
        
        # ベースモデルの重みを取得
        base_state_dict = self.base_model.state_dict()
        
        # 注意機構の重み（Q, K, V）にSO(8)回転を吸収
        for name, weight in base_state_dict.items():
            if 'q_proj.weight' in name or 'k_proj.weight' in name or 'v_proj.weight' in name:
                # 重み行列にSO(8)回転を吸収
                absorbed_weight = self.so8_rotation_gate.export_weights(weight)
                exported_weights[name] = absorbed_weight
        
        return exported_weights
    
    @torch.no_grad()
    def generate_answer(
        self,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_new_tokens: int = 256,
        do_self_verification: bool = False,
        num_paths: int = 3,
        temperature: float = 0.4,
        top_p: float = 0.9,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        高レベルAPI:
        - Safety Gateで判定
        - ALLOWなら（任意で）Self-Verificationを行い最終回答を生成
        - ESCALATE/REFUSEならその旨を返す
        
        Args:
            tokenizer: トークナイザー
            prompt: 入力プロンプト
            max_new_tokens: 最大生成トークン数
            do_self_verification: Self-Verificationを行うか
            num_paths: Self-Verificationのパス数
            temperature: サンプリング温度
            top_p: Top-pサンプリング
            device: デバイス
        
        Returns:
            生成結果の辞書
        """
        self.eval()
        tokenizer.pad_token = tokenizer.eos_token
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        sg = self.safety_gate(**inputs)
        decision = sg["decisions"][0]
        conf = sg["confidence"][0].item()
        
        if decision == "REFUSE":
            return {
                "decision": "REFUSE",
                "reason": "safety_head_refuse",
                "confidence": conf,
                "answer": "I'm sorry, but I can't assist with that request.",
            }
        
        if decision == "ESCALATE":
            return {
                "decision": "ESCALATE",
                "reason": "safety_head_escalate_or_low_conf",
                "confidence": conf,
                "answer": "This request requires human review or higher-level approval.",
            }
        
        # ALLOWの場合
        if not do_self_verification or num_paths <= 1:
            output_ids = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            answer = text[len(prompt):].lstrip()
            return {
                "decision": "ALLOW",
                "confidence": conf,
                "answer": answer,
            }
        
        # Self-Verification付き ALLOW
        paths = []
        scores = []
        for i in range(num_paths):
            tagged_prompt = (
                f"[STRATEGY_{i}] Carefully reason step by step.\n" + prompt
            )
            in_i = tokenizer(tagged_prompt, return_tensors="pt").to(device)
            out_i = self.base_model.generate(
                **in_i,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            txt_i = tokenizer.decode(out_i[0], skip_special_tokens=True)
            ans_i = txt_i[len(tagged_prompt):].lstrip()
            paths.append(ans_i)
            
            # 簡易自己評価（Verifierロール風）
            score_i = self.simple_self_score(prompt, ans_i, device=device)
            scores.append(score_i)
        
        # ベストパス選択
        best_idx = max(range(num_paths), key=lambda i: scores[i]["overall"])
        best_path = paths[best_idx]
        best_score = scores[best_idx]
        
        # 最終回答を短く再生成（要約）
        final_prompt = (
            "Summarize the following reasoning into a concise final answer for the user.\n"
            "Reasoning:\n"
            + best_path
            + "\nFinal answer (no meta-commentary):"
        )
        in_final = tokenizer(final_prompt, return_tensors="pt").to(device)
        out_final = self.base_model.generate(
            **in_final,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        txt_final = tokenizer.decode(out_final[0], skip_special_tokens=True)
        final_answer = txt_final[len(final_prompt):].lstrip()
        
        return {
            "decision": "ALLOW",
            "confidence": conf,
            "paths": paths,
            "scores": scores,
            "best_index": best_idx,
            "best_score": best_score,
            "answer": final_answer,
        }
    
    @torch.no_grad()
    def simple_self_score(
        self,
        prompt: str,
        answer: str,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        簡易な自己評価器（Verifierロールの代替）。
        ここではモデル自身を使わず、ルールベースのダミー実装。
        実際には:
          - 別プロンプトでモデルに JSON スコアを出させる
          - もしくは別Verifierモデルを使う
        
        Args:
            prompt: 入力プロンプト
            answer: 生成された回答
            device: デバイス
        
        Returns:
            スコアの辞書
        """
        # デモなので固定値＋長さ・構造で微調整
        length = len(answer.split())
        logical = 0.6 + 0.4 * (1.0 if length > 5 else 0.0)
        constraints = 0.6
        math = 0.6
        safety = 0.9
        overall = (logical + constraints + math + safety) / 4.0
        return {
            "logical": float(logical),
            "constraints": float(constraints),
            "math": float(math),
            "safety": float(safety),
            "overall": float(overall),
        }

