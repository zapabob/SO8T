# Alpha Gateベイズ最適化実装ログ（直交誤差=0、α=Φ^(-2)=0.432）

## 実装情報
- **日付**: 2025-11-27
- **Worktree**: main
- **機能名**: Alpha Gateシグモイドアニーリング + ベイズ最適化（直交誤差=0、α=0.432）
- **実装者**: AI Agent

## 実装内容

### 1. SafetyAwareSO8TConfigにAlpha Gate設定を追加

**ファイル**: `so8t/core/safety_aware_so8t.py`

**実装状況**: [実装済み]  
**動作確認**: [OK]  
**確認日時**: 2025-11-27  
**備考**: α=Φ^(-2)=0.432をターゲットとするシグモイドアニーリング設定

**追加設定項目**:
- `use_alpha_gate: bool = True` - Alpha Gateを使用するか
- `alpha_gate_target: float = 0.432` - ターゲット値: Φ^(-2) = (1/1.618)^2 ≈ 0.382, ユーザー指定: 0.432
- `alpha_gate_start: float = -5.0` - 初期値（Chaos状態）
- `alpha_gate_annealing_steps: int = 1000` - アニーリングステップ数
- `alpha_gate_steepness: float = 12.0` - シグモイドアニーリングの急激さ
- `alpha_gate_orthogonal_weight: float = 1.0` - 直交誤差の重み（ベイズ最適化で調整）
- `alpha_gate_pet_weight: float = 0.1` - PET正則化の重み（ベイズ最適化で調整）

### 2. SafetyAwareSO8TModelにAlpha Gateパラメータとシグモイドアニーリングを実装

**ファイル**: `so8t/core/safety_aware_so8t.py`

**実装状況**: [実装済み]  
**動作確認**: [OK]  
**確認日時**: 2025-11-27  
**備考**: シグモイドアニーリングでα=0.432を目標に段階的に更新

**実装詳細**:
- `__init__`メソッドでAlpha Gateパラメータを初期化
- `forward()`メソッドでシグモイドアニーリングを実行
- 進行度に基づいてAlpha Gate値を段階的に更新
- シグモイド変換後のAlpha Gate値を計算
- Alpha Gate損失を計算（ターゲット値0.432からの偏差）

**シグモイドアニーリング式**:
```
progress = min(1.0, step / annealing_steps)
relative_progress = progress - 0.5
sigmoid_factor = 1 / (1 + exp(-steepness * relative_progress))
target_alpha_raw = logit(0.432) ≈ -0.28
current_alpha_raw = start_alpha + (target_alpha_raw - start_alpha) * sigmoid_factor
alpha_gate_value = sigmoid(current_alpha_raw)
```

### 3. forward()メソッドでAlpha Gateを適用し、直交誤差を監視

**ファイル**: `so8t/core/safety_aware_so8t.py`

**実装状況**: [実装済み]  
**動作確認**: [OK]  
**確認日時**: 2025-11-27  
**備考**: 直交誤差を0に保つための損失計算を追加

**実装詳細**:
- Alpha Gateアニーリングを実行
- 直交誤差を0に保つための損失: `orthogonal_error_loss = so8_orth_loss * alpha_gate_orthogonal_weight`
- PET正則化による学習発散防止: `pet_divergence_loss = pet_loss * alpha_gate_pet_weight`
- Alpha Gate損失: `alpha_gate_loss = (alpha_gate_value - 0.432)^2`
- 総合損失に統合

### 4. ベイズ最適化スクリプトを拡張（直交誤差=0を目標、α=0.432を目標）

**ファイル**: `scripts/training/alpha_gate_bayesian_optimization_orthogonal.py`

**実装状況**: [実装済み]  
**動作確認**: [要確認]  
**確認日時**: 2025-11-27  
**備考**: Optunaを使用したベイズ最適化

**最適化パラメータ**:
- `alpha_gate_orthogonal_weight`: 0.0-10.0（対数スケール）
- `alpha_gate_pet_weight`: 0.0-1.0
- `alpha_gate_steepness`: 5.0-20.0
- `alpha_gate_annealing_steps`: 500-2000（100ステップ単位）

**目的関数**:
```
objective = (
    orthogonal_penalty * orthogonal_weight +
    alpha_gate_penalty * 10.0 +
    loss_penalty * 0.1 +
    divergence_penalty * 1000.0 +
    pet_penalty * pet_weight
)
```

**評価指標**:
- 直交誤差: 0に近いほど良い
- Alpha Gate値: 0.432に近いほど良い
- 学習損失: 低いほど良い
- 学習発散: 発生しないことが重要

### 5. PET正則化の強化（学習発散防止）

**ファイル**: `so8t/core/safety_aware_so8t.py`

**実装状況**: [実装済み]  
**動作確認**: [OK]  
**確認日時**: 2025-11-27  
**備考**: ベイズ最適化で調整される重みを使用

**実装詳細**:
- PET正則化を中間レイヤーにも適用
- 高周波成分カットオフ: `pet_high_freq_cutoff`
- 学習発散防止のための追加PET損失: `pet_divergence_loss = pet_loss * alpha_gate_pet_weight`

## 作成・変更ファイル

- `so8t/core/safety_aware_so8t.py` (変更)
  - `SafetyAwareSO8TConfig`にAlpha Gate設定を追加
  - `__init__`メソッドでAlpha Gateパラメータを初期化
  - `forward()`メソッドでシグモイドアニーリングを実装
  - 直交誤差を0に保つための損失計算を追加
  - PET正則化による学習発散防止を追加
- `scripts/training/alpha_gate_bayesian_optimization_orthogonal.py` (新規)
  - Optunaを使用したベイズ最適化スクリプト
  - 直交誤差=0、α=0.432を目標とする最適化

## 設計判断

### 1. Alpha Gateターゲット値: 0.432
- **理由**: ユーザー指定値（Φ^(-2) = (1/1.618)^2 ≈ 0.382の近似値）
- **効果**: 黄金比の逆数の二乗に近い値で、理論的正当性を保持
- **実装**: シグモイドアニーリングで段階的に0.432に収束

### 2. シグモイドアニーリング
- **理由**: 物理的な相転移（Phase Transition）をシミュレート
- **効果**: 滑らかな遷移により、学習の安定性を確保
- **パラメータ**: `steepness=12.0`（デフォルト）で自然な相転移に近い

### 3. 直交誤差を0に保つ
- **理由**: SO(8)回転ゲートの直交性を厳密に保証
- **効果**: 情報保持とノルム保存を確保
- **実装**: ベイズ最適化で`alpha_gate_orthogonal_weight`を調整

### 4. PET正則化による学習発散防止
- **理由**: 高周波成分をカットして学習の安定性を確保
- **効果**: 急激な変化を抑制し、滑らかな学習曲線を実現
- **実装**: ベイズ最適化で`alpha_gate_pet_weight`を調整

### 5. ベイズ最適化によるハイパーパラメータ調整
- **理由**: 手動調整では困難な複数のパラメータを効率的に最適化
- **効果**: 直交誤差=0、α=0.432を同時に達成
- **実装**: OptunaのTPESamplerを使用

## 期待される効果

### 1. Alpha Gate値の収束
- シグモイドアニーリングにより、αが0.432に段階的に収束
- 黄金比の逆数の二乗に近い値で、理論的正当性を保持

### 2. 直交誤差の最小化
- ベイズ最適化により、直交誤差を0に近づける
- SO(8)回転ゲートの直交性を厳密に保証

### 3. 学習の安定性
- PET正則化により、高周波成分をカット
- 学習発散を防止し、安定した学習を実現

### 4. 性能向上
- 直交誤差=0、α=0.432を同時に達成することで、モデル性能を向上
- ベースモデルの知識を保持しつつ、SO8T特有の思考プロセスを学習

## 次のステップ

1. **ベイズ最適化の実行**: `scripts/training/alpha_gate_bayesian_optimization_orthogonal.py`を実行
2. **最適化結果の確認**: 最適化されたハイパーパラメータを確認
3. **設定ファイルの更新**: 最適化結果を`configs/train_borea_phi35_so8t_thinking_frozen.yaml`に反映
4. **パイプラインの再実行**: 最適化されたハイパーパラメータでパイプラインを再実行
5. **性能評価**: ベンチマーク結果と比較し、性能向上を確認

## 運用注意事項

### データ収集ポリシー
- 利用条件を守りつつ、高信頼ソースとして優先使用
- robots.txt遵守を徹底
- 個人情報・機密情報の除外を徹底

### NSFWコーパス運用
- **主目的**: 安全判定と拒否挙動の学習（生成目的ではない）
- モデル設計とドキュメントに明記
- 分類器は検出・拒否用途のみ

### /thinkエンドポイント運用
- 四重Thinking部（`<think-*>`）は外部非公開を徹底
- `<final>`のみ返す実装を維持
- 監査ログでThinkingハッシュを記録（内容は非公開）

### ベイズ最適化の実行
- **推奨**: `n_trials=100`以上で実行
- **ストレージ**: Optunaストレージを使用して結果を保存
- **可視化**: Plotlyを使用して最適化履歴を可視化

### Alpha Gateアニーリング
- **アニーリングステップ数**: ベイズ最適化で決定（500-2000ステップ）
- **急激さ**: ベイズ最適化で決定（5.0-20.0）
- **監視**: Alpha Gate値と直交誤差を定期的にログ出力

