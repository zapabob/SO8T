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

---

## DeepResearch ベストプラクティス

DeepResearch（深層研究）は、AI エージェントが自律的に調査・分析を行い、複雑な問いに対して深い洞察を提供する能力です。

### 1. 研究フェーズの構造化

DeepResearch は以下のフェーズで進行します：

```python
# 研究フェーズの定義
RESEARCH_PHASES = [
    "question_analysis",      # 問いの分析と分解
    "hypothesis_generation",  # 仮説生成
    "information_gathering",  # 情報収集
    "evidence_evaluation",    # 証拠評価
    "synthesis",              # 統合
    "verification",           # 検証
    "reporting"               # 報告
]
```

### 2. 自律的調査戦略

- **多角的アプローチ**: 複数の情報源（学術論文、ニュース、政府文書）から情報を収集
- **反復的深化**: 初期調査結果に基づいて調査を深化
- **信頼性評価**: 情報源の信頼性を評価（学術誌 > 政府機関 > ニュース）
- **矛盾検出**: 複数情報源の矛盾を検出し検証

### 3. ツール・ツール連鎖

```python
# 推奨されるツール連鎖
TOOL_CHAINS = {
    "research": ["web_search", "arxiv_search", "llm_analysis"],
    "fact_check": ["web_search", "source_verification", "cross_reference"],
    "trend_analysis": ["data_collection", "statistical_analysis", "visualization"]
}
```

### 4. 検証と品質保証

- **根拠付き回答**: すべての主張に情報源を明示
- **不確実性の定量化**: 確信度を数値化（0-1）
- **論理的整合性チェック**: 推論の妥当性を検証
- **専門家レビュー**: 重要結論は専門家によるレビューを経る

### 5. 代表的な DeepResearch タスク

```python
DEEP_RESEARCH_TASKS = [
    "政策影響分析",           # 政策変更の影響を多角的に分析
    "技術トレンド予測",       # 新興技術の将来を予測
    "リスク評価",             # リスク要因の特定と評価
    "比較分析",              # 複数オプションの比較評価
    "シナリオ分析",          # 将来シナリオの構築と評価
    "因果推論",              # 因果関係の特定と検証
    "知識統合",              # 複数分野の知識を統合
    "長期的影響予測"         # 長期的な影響を予測
]
```

### 6. 実装ガイドライン

- **研究ログの記録**: すべての調査過程をログに記録
- **中間成果物の保存**: 部分的な発見を保存し再利用可能に
- **並列研究**: 独立した調査は並列実行して効率化
- **人間によるレビュー**: 重要な判断は人間にレビュー依頼

### 7. 参照ファイル

- `scripts/research/deep_research_events.py`: DeepResearch イベント処理
- `scripts/research/shinka_neat_ai_scientist.py`: 進化的研究エージェント
- `src/agents/sakana_ai_integrated_agent.py`: 自律研究エージェント統合
