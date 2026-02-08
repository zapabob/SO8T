# 2026-02-08_SO8T_GrandDesign_Implementation

## 概要

SO8Tリポジトリに対し、Borea-phi3.5-instinct-jpを核にした「グランドデザイン」に基づく増築方式の学習基盤を実装。
四重推論アダプター、PET正則化（二階差分）、進化的動的凍結（CEM）、GRAPE→SFT、mHC射影、GRPO（信頼領域つき）を統合。

## 実装内容

### 1. モデルアーキテクチャ

- **SO8T Adapter Bank**: `src/models/adapters_so8t.py`
  - $L \times P$ 個の `ResidualAdapter` (down->GELU->up) を MLP 残差に挿入。
  - 学習可能なゲート係数 $\alpha_{\ell,p}$ を導入。初期値 0。
- **PET 正則化**: `src/models/losses_pet.py`
  - パス方向および層方向の二階差分損失 $\mathcal{L}_{PET}$ を実装。
- **mHC 射影**: `src/models/mhc_projection.py`
  - $\alpha$ の L2 射影 / Clip による多様体安定化。

### 2. 学習・進化プロセス

- **EvoFreeze-CEM**: `src/training/evofreeze_cem.py`
  - パラメータを G0-G8 のグループに分類し、CEM による解凍確率制御。
  - 収束・KL乖離・ドリフト・PETスパイクを監視する **Rollback Engine** を搭載。
- **訓練フェーズ**:
  - `src/training/grape_sft.py`: ベースモデル整合性による候補選択。
  - `src/training/grpo_rl.py`: 群相対方策最適化（参照方策との KL 制約 / 更新ノルム制約）。

### 3. マルチモーダル拡張 (初期)

- **Vision (SO8ViT)**: `src/models/vision_encoder.py`, `projector.py`
  - 凍結 ViT + 学習可能 Projector による画像理解基盤。
- **Audio (Codec-LM)**: `src/models/codec_lm.py`
  - トークンベースの音声合成・制御基盤。

### 4. 統合と検証

- `src/training/train_unsloth_so8t.py`: 全コンポーネントを Unsloth Trainer に統合。
- `scripts/acceptance_test_so8t.py`: 受入れテスト（Pass-ID による出力変化の確認）。

## 特記事項

- **安定性重視**: 初期状態では LLM 本体を凍結し、アダプターと接続部から順次解凍する動的制御。
- **ロールバック**: KL閾値超過時に直近 stable チェックポイントへ自動復帰する機構。
