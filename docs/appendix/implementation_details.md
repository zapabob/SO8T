# 付録B: 実装詳細

## SO8T回転ゲート実装

### クラス: SO8TRotationGate

```python
class SO8TRotationGate(nn.Module):
    """
    SO(8)回転ゲート層
    
    hidden_dimを8次元ブロックに分割し、各ブロックに直交回転を適用
    """
    
    def __init__(self, num_blocks: int, block_size: int = 8):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_size = block_size
        
        # 歪対称行列パラメータ（28自由度/ブロック）
        skew_dim = block_size * (block_size - 1) // 2
        self.skew_params = nn.Parameter(
            torch.zeros(num_blocks, skew_dim)
        )
    
    def _build_skew_symmetric(self, params: torch.Tensor) -> torch.Tensor:
        """
        28パラメータから8×8歪対称行列構築
        
        Args:
            params: [num_blocks, 28]
        
        Returns:
            skew: [num_blocks, 8, 8] 歪対称行列
        """
        batch_size = params.size(0)
        skew = torch.zeros(batch_size, 8, 8, device=params.device, dtype=params.dtype)
        
        # 上三角要素を設定
        idx = 0
        for i in range(8):
            for j in range(i+1, 8):
                skew[:, i, j] = params[:, idx]
                skew[:, j, i] = -params[:, idx]  # 歪対称性
                idx += 1
        
        return skew
    
    def _matrix_exp(self, A: torch.Tensor) -> torch.Tensor:
        """
        行列指数関数（Padé近似）
        
        Args:
            A: [batch, 8, 8] 歪対称行列
        
        Returns:
            R: [batch, 8, 8] 回転行列
        """
        I = torch.eye(8, device=A.device, dtype=A.dtype).unsqueeze(0)
        
        # (1,1) Padé近似
        numerator = I + A / 2
        denominator = I - A / 2
        
        R = torch.linalg.solve(denominator, numerator)
        
        # 直交性正則化（SVD）
        U, S, V = torch.svd(R)
        R_ortho = U @ V.transpose(-2, -1)
        
        return R_ortho
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: [batch, seq_len, hidden_dim]
        
        Returns:
            rotated: [batch, seq_len, hidden_dim]
        """
        batch, seq_len, hidden_dim = x.shape
        
        # ブロック分割
        x_blocks = x.view(batch, seq_len, self.num_blocks, self.block_size)
        
        # 歪対称行列構築
        skew = self._build_skew_symmetric(self.skew_params)
        
        # 回転行列生成
        R = self._matrix_exp(skew)  # [num_blocks, 8, 8]
        
        # ブロック毎に回転適用
        rotated_blocks = torch.einsum('bsnk,nkm->bsnm', x_blocks, R)
        
        # 再結合
        rotated = rotated_blocks.view(batch, seq_len, hidden_dim)
        
        return rotated
```

## PET正規化実装

### クラス: PETLoss

```python
class PETLoss(nn.Module):
    """PET正規化損失"""
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        二階差分ペナルティ計算
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            pet_loss: scalar
        """
        if hidden_states.size(1) < 3:
            return torch.tensor(0.0, device=hidden_states.device)
        
        # 時系列スライス
        h_t = hidden_states[:, :-2, :]      # [batch, T-2, d]
        h_t1 = hidden_states[:, 1:-1, :]    # [batch, T-2, d]
        h_t2 = hidden_states[:, 2:, :]      # [batch, T-2, d]
        
        # 二階差分
        second_diff = h_t2 - 2.0 * h_t1 + h_t
        
        # L2ノルム（平均）
        pet_loss = torch.mean(second_diff ** 2)
        
        return pet_loss
```

### 勾配計算

**自動微分による勾配**:
```
∂L_PET/∂h_t = 2(h_t - 2h_{t+1} + h_{t+2})
∂L_PET/∂h_{t+1} = 2(-2h_t + 4h_{t+1} - 2h_{t+2})
∂L_PET/∂h_{t+2} = 2(h_t - 2h_{t+1} + h_{t+2})
```

**時系列方向の結合**:
各時刻の隠れ状態は、t-2, t-1, t, t+1, t+2の範囲から勾配を受け取る。

## QLoRAの数学的詳細

### LoRA行列の初期化

**A行列**:
```
A ~ N(0, σ²)
σ = 1 / √r
```

**B行列**:
```
B = 0（ゼロ初期化）
```

**理由**: 初期時点で W' = W + BA = W（事前学習重みそのまま）

### スケーリング

**実効更新**:
```
ΔW = (α / r) × BA

ここで:
- α: lora_alpha（128）
- r: lora_r（64）
- α/r = 2.0（スケーリング係数）
```

**設計根拠**:
α/rを>1にすることで、LoRAの寄与を強化。

### 8bit量子化の詳細

**線形量子化**:
```
W_8bit = round(255 × (W_fp16 - W_min) / (W_max - W_min))
```

**逆量子化**:
```
W_dequant = W_min + (W_8bit / 255) × (W_max - W_min)
```

**量子化誤差**:
```
ε = ||W_fp16 - W_dequant||

期待値: ε ≈ (W_max - W_min) / (2 × 255) ≈ 0.2%
```

## 勾配ノイズ注入の理論

### ランジュバン動力学

**確率的勾配降下**:
```
θ_{t+1} = θ_t - η ∇L(θ_t) + √(2ηT) ξ_t

ここで:
- η: 学習率
- T: 温度パラメータ
- ξ_t ~ N(0, I): ガウスノイズ
```

**効果**:
- 局所最適解脱出
- 広い谷探索
- 汎化性能向上

### 実装

```python
class GradientNoiseInjector:
    def __init__(self, std: float = 0.01):
        self.std = std
    
    def inject(self, model: nn.Module):
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * self.std
                    param.grad.add_(noise)
```

## SWAの数学的解析

### 平均化の効果

**Loss Landscape近似**:
```
L(θ) ≈ L(θ*) + (1/2)(θ - θ*)^T H (θ - θ*)

ここで:
- θ*: 谷の中心
- H: Hessian
```

**複数チェックポイントの平均**:
```
θ_swa = (1/n) Σ θ_i

期待値:
E[θ_swa] → θ*（谷の中心）
```

### 較正品質への影響

**過確信の抑制**:
```
SWA平均により、極端な重みが平滑化
→ softmax確率が適切に分散
→ ECE改善
```

**実験結果（文献）**:
```
SWA無し: ECE ≈ 0.08
SWA有り: ECE ≈ 0.04（50%改善）
```

## 学習率スケジューラの数学

### Cosine Annealing

**式**:
```
η(t) = η_min + (η_max - η_min) × (1 + cos(π × t / T)) / 2
```

**導関数**:
```
dη/dt = -(π/2T) (η_max - η_min) sin(π × t / T)
```

**特性**:
- t=0: η = η_max（初期学習率）
- t=T/2: η = η_min（最小）
- t=T: η = η_max（再上昇、ただし通常は使わない）

### Warmup

**線形Warmup**:
```
η(t) = η_max × min(1, t / T_warmup)
```

**効果**:
- 初期の不安定性回避
- 大きな勾配によるパラメータ破壊防止
- 収束品質向上

---

**付録B終了**

