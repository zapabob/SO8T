# NKAT-SO8T RTX 3060 実装ガイド

## 概要

この実装は、`Borea-phi3.5-mini-instruct-jp`に対する「NKAT-SO8T アダプタ」の追加学習を行い、70時間の学習時間を12時間以内に短縮しつつ、理論的に正しいAlpha Gateの相転移挙動を実現します。

### 核心概念: Alpha Gate (α) と SO(8) Triality

- **Alpha Gate**: $h_{out} = h_{frozen\_mlp} + \sigma(\alpha) \cdot h_{so8t}$
  - 初期化: α = -5.0 (σ(-5.0) ≈ 0.006)
  - 学習初期は幾何学パスを「ほぼ遮断」し、自然な相転移を促す

- **アニーリングスケジューラ**: λ(t) = 0.0 → 0.1 (学習初期から徐々に幾何学的制約を強化)

## RTX 3060 最適化仕様

### メモリ管理
- **完全凍結**: ベースモデルパラメータは `requires_grad = False`
- **マイクロバッチ**: `batch_size = 1`, `gradient_accumulation_steps = 32` (実質バッチサイズ 32)
- **BF16 精度**: Ampere GPU ネイティブサポート
- **グラジエントチェックポイント**: VRAM 使用量を大幅削減

### パフォーマンス目標
- **学習時間**: 70時間 → 12時間以内
- **VRAM 使用量**: 12GB 以内
- **理論的正確性**: Alpha Gate の相転移挙動を維持

## 実装ファイル

### 1. NKAT_Wrapper クラス
**ファイル**: `src/layers/nkat_wrapper.py`

```python
class NKAT_Wrapper(nn.Module):
    """NKAT-SO8T Adapter with Alpha Gate"""

    def __init__(self, base_model, init_alpha: float = -5.0):
        # Alpha Gate 実装
        # SO(8) 幾何学推論
        # 凍結ロジック
```

### 2. RTX 3060 最適化トレーニング
**ファイル**: `scripts/training/train_nkat_so8t_adapter_optimized.py`

```python
class OptimizedNKATSO8TTrainer(NKATSO8TTrainer):
    """RTX 3060 optimized trainer with VRAM monitoring"""

    def enable_memory_optimizations(self):
        # 動的バッチサイズ調整
        # VRAM 監視
        # メモリ効率的注意機構
```

### 3. 相転移検証システム
**ファイル**: `scripts/validation/validate_nkat_so8t_phase_transition.py`

```python
class PhaseTransitionAnalyzer:
    """Alpha Gate 相転移行動の理論的検証"""

    def analyze_phase_transition(self):
        # 相転移検出
        # SO(8) 幾何学整合性検証
        # 数学的推論能力評価
```

### 4. 完全パイプライン
**ファイル**: `scripts/training/run_complete_nkat_so8t_pipeline.bat`

```batch
# トレーニング + 検証 + GGUF変換 + 性能テスト
call scripts/training/run_complete_nkat_so8t_pipeline.bat
```

## 使用方法

### 1. 環境準備

```bash
# データセット準備 (既存の4値分類データを使用)
# RTX 3060 で実行することを確認
nvidia-smi  # GPU メモリを確認
```

### 2. トレーニング実行

```bash
# 完全パイプライン実行
call scripts/training/run_complete_nkat_so8t_pipeline.bat

# または個別実行
py -3 scripts/training/train_nkat_so8t_adapter_optimized.py ^
    --model-path "models/Borea-Phi-3.5-mini-Instruct-Jp" ^
    --output-dir "D:/webdataset/checkpoints/nkat_so8t_rtx3060" ^
    --train-data "data/splits/train_four_class.jsonl" ^
    --max-steps 5000 ^
    --batch-size 1 ^
    --gradient-accumulation 16 ^
    --learning-rate 5e-5
```

### 3. 検証実行

```bash
# 相転移検証
py -3 scripts/validation/validate_nkat_so8t_phase_transition.py ^
    --checkpoint-dir "D:/webdataset/checkpoints/nkat_so8t_rtx3060" ^
    --output-report "_docs/nkat_so8t_phase_transition_report.md" ^
    --plot-phase-transition "_docs/nkat_so8t_phase_transition.png"
```

## 理論的検証項目

### Alpha Gate 相転移
- [ ] α の初期値が -5.0 付近
- [ ] 学習中に σ(α) が自然に増加
- [ ] 相転移タイミングの検出
- [ ] 幾何学的推論能力の向上

### SO(8) 幾何学整合性
- [ ] 回転行列の直交性: R^T @ R ≈ I
- [ ] 行列式の値: |R| = ±1
- [ ] 幾何学的制約の満足

### 学習安定性
- [ ] VRAM 使用量 12GB 以内
- [ ] 学習時間の 12 時間以内達成
- [ ] 損失関数の安定収束
- [ ] 過学習の防止

## 期待される結果

### 性能指標
- **学習時間**: ≤ 12 時間
- **VRAM 使用量**: ≤ 11.5 GB
- **最終損失**: < 1.0
- **相転移検出**: Step 1000-2000 付近

### 理論的達成
- Alpha Gate の自然な開口
- SO(8) 幾何学の整合性維持
- 数学的・幾何学的推論能力の向上
- ベースモデルの言語能力維持

## トラブルシューティング

### VRAM 不足
```python
# バッチサイズをさらに削減
--batch-size 1 --gradient-accumulation 8
```

### 学習不安定
```python
# 学習率を調整
--learning-rate 1e-5
```

### 相転移が発生しない
```python
# アニーリングを調整
--annealing-warmup 500
```

## 技術的詳細

### Alpha Gate 数式
```
α ∈ ℝ (学習可能パラメータ)
σ(α) = 1 / (1 + exp(-α)) ∈ (0, 1)
h_out = h_base + σ(α) · h_so8t
```

### SO(8) 回転生成
```
A = 0.5 * (W - W^T)  # Skew-symmetric
R = exp(A)           # SO(8) rotation
R^T @ R = I          # Orthogonality
det(R) = ±1          # Rotation property
```

### アニーリングスケジュール
```
λ(t) = 0.1 * min(1.0, t / warmup_steps)
L_total = L_LM + λ(t) * L_triality
```

この実装により、RTX 3060 の制約環境下で理論的に正しい NKAT-SO8T アダプタの学習が可能となります。


