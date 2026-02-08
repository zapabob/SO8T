# SO8T スキル集

## GitHub 操作ポリシー

すべての GitHub 操作（PR作成、レビュー、コミット情報取得、Issue管理等）は GitHub CLI（`gh`）を使用すること。

```bash
# PR作成
gh pr create --title "title" --body "body"

# PR情報取得
gh pr view <pr-number>

# コミット情報取得
gh pr checks <pr-number>

# Issue作成
gh issue create --title "title" --body "body"
```

---

## データ収集ポリシー

### 重複データセットの処理

データ収集時には、収集済みのデータをスキップし、重複を避けること。

- **チェックポイント利用**: パイプラインは定期的にチェックポイントを保存し、中断箇所から再開可能
- **既存データセットの再利用**: `--use-existing-datasets` フラグでデータ収集フェーズをスキップし、統合済みデータから直接SFT訓練を開始可能
- **ローリングチェックポイント**: 3-5分間隔で保存し、再開時に収集済みデータを再取得しない
- **データセット識別子**: 各データセットには一意の識別子を付与し、重複を検出

### 参照ドキュメント

- `docs/2026-01-29_ムーンショットパイプライン2025-2026再稼働とBF16対応.md`: `--use-existing-datasets` フラグの説明
- `src/utils/checkpoint_manager.py`: チェックポイント管理システム

---

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
