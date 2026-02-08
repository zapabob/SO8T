# 実装方針（Software Engineering Best Practices）

この文書は、SO8T / AEGIS 系の「自動再学習（SFT+GRPO）」「ABCベンチマーク」「統計解析（多重補正t検定＋ANOVA）」「Prism向けMD/LaTeX出力」「ブート時自動起動＋チェックポイント保全」を、**保守性・再現性・安全性**を最優先して実装するための方針です。

---

## 0. 最優先の設計原則

1. **再現性（Reproducibility First）**
   - 乱数seed、評価プロンプト、抽出ロジック、データセット版、git commit hash を必ず記録。
   - 生成物（結果JSON/CSV/PNG/MD/TEX）から、同条件で再実行できることをDefinition of Doneとする。
2. **分離（Separation of Concerns）**
   - 収集・整形・学習・評価・統計・可視化・公開（HF/GH）を明確に分割し、責務境界を崩さない。
3. **壊れにくさ（Resumability / Fault-tolerance）**
   - 電源断・OOM・ネットワーク断で止まっても**checkpointから復帰**できる。
   - “途中結果を捨てない”ことを標準挙動にする。
4. **資源制約の明示（RTX3060 / RAM32GB）**
   - VRAM 12GB前提：LoRA/QLoRA、grad checkpoint、低バッチ、offloadを基本戦略にする。
5. **観測性（Observability）**
   - tqdm風の進捗＋構造化ログ＋SQLite履歴（後述）で「何が起きたか」を後から追える。

---

## 1. ブランチ／worktree 運用（OpenCode分離）

- **main**：安定運用（ブート自動化、再現可能ベンチ、HF/GH公開の成果物）を保持。
- **OpenCode（git worktree）**：実験・検証・大規模改修・外部研究の統合を先行。
- **マージ原則**
  - OpenCode → main は「再現可能性」と「ログ・ドキュメント」が揃った時点でPR/マージ。
  - 実験中の破壊的変更は main に入れない（特にデータパス、チェックポイント形式、評価条件）。

---

## 2. リポジトリ構造・モジュール分割

### 2.1 ディレクトリ責務（例）
- `scripts/utils/`：ブート起動、モニタ、チェックポイント、進捗記録（ランタイム機能）
- `scripts/pipeline/`：統合パイプライン（データ→学習→評価→レポート）
- `scripts/training/`：SFT/GRPO の実行器（RTX3060向け最適化を集約）
- `scripts/evaluation/`：ABCテスト、LM-eval呼び出し、結果保存
- `scripts/analysis/`：統計解析・可視化・レポート生成（MD/LaTeX）
- `data/`：データ実体（巨大ファイル）＋ `data/manifest/`（後述：ハッシュとメタ）
- `docs/`：RUNBOOK / 方針 / モデルカード / Prism用成果物（MD/TeX/PNG）

### 2.2 1000行ルール
- 1ファイル1000行を目安に分割（特に `scripts/evaluation/*` と `scripts/pipeline/*`）。
- 共通処理は `shared/` 相当（現状は `utils/` や `scripts/utils/`）へ寄せ、重複を許さない。

---

## 3. 設定管理（Config）と秘密情報

- **設定はファイル化**：`config/*.json|yaml` に寄せ、CLIは config を上書きできる設計。
- **秘密情報はコミット禁止**：`.env.local` / OS環境変数を使用。ログにも出さない。
- **実行条件を記録**：実行時のconfigを必ず `results/<run_id>/config_snapshot.json` に保存。

---

## 4. ロギング／進捗／SQL（永続追跡）

### 4.1 ログ
- すべての主要処理は `logs/` に出力（例：`boot_pipeline_launcher.log`）。
- ログは「人間が読めるテキスト」＋「機械が読めるJSON（任意）」の二層。

### 4.2 進捗（tqdm風）
- “Simple English”で統一（例：`progress: |######----| training running`）。
- 監視UI（rich等）は任意だが、**ログだけで復元可能**な状態を残す。

### 4.3 SQLite履歴（推奨）
- `logs/pipeline_progress.sqlite` に以下を保存：
  - run_id, timestamp, phase, seed, dataset_version, checkpoint_path, metrics(JSON)
- 再起動後の再開時は SQLite を参照して「最後の正常checkpoint」を確定する。

---

## 5. チェックポイント設計

- **内側（training）**：`checkpoints/latest_checkpoint.json` を原本として更新（原子書き込み推奨）。
- **外側（boot/rolling）**：5分間隔で3世代保持（`checkpoints/rolling_snapshots/`）。
- checkpointには必ず以下を含める：
  - git commit hash / config hash / dataset hash / seed / step / optimizer state要約

---

## 6. データセットの工学的運用（Manifest＋検証）

- `data/manifest/` に各データセットの
  - SHA256、行数、スキーマ、ライセンス、生成スクリプト、生成日、用途（SFT/GRPO/Eval）
  を記録し、**HF公開時に追跡可能**にする。
- 入力データは必ずスキーマ検証（JSONLの必須key、role整合、長さ上限）。

---

## 7. 学習（SFT+GRPO）実装方針（RTX3060前提）

- VRAM 12GB運用の既定：
  - LoRA/QLoRA、gradient checkpointing、低batch、必要に応じてCPU offload。
- OOM時は「落ちる」ではなく「自動で段階的に設定を落としてリトライ」。
- `TORCH_COMPILE_DISABLE` 等の環境変数は、Windows安定動作のため統一して設定する。

---

## 8. ABCテスト／統計解析（論文水準の再現性）

- **ABCテスト**：A/B/Cを同一条件（prompt、抽出、seed、サンプル）で実行し、seed反復を必須。
- **統計**：
  - 主要：Welch t-test（不等分散）＋ Holm もしくは Bonferroni
  - 全体：one-way ANOVA（η²も併記）
- **成果物**：
  - `summary_statistics.csv`（Prismに近い形）
  - `docs/abc_prism_summary.md` / `.tex`（Prism整形用）
  - `docs/figures/*.png`（エラーバー）

---

## 9. ドキュメント（運用・採用目線・学術目線）

- `docs/RUNBOOK.md`：実行手順（ブート、再開、評価、公開）
- `docs/IMPLEMENTATION_LOG.md`：実装ログ（いつ/何を/どう変えたか）
- 学術形式：MD＋LaTeXの両方を用意し、引用は `References` または `thebibliography` で統一。

---

## 10. 品質保証（テスト／CI）

- 重要関数（統計、集計、チェックポイント入出力）は unit test を置く。
- “動く”だけでなく、**再実行して同じ結論が出る**ことを検証項目にする。

---

## 11. 大容量ファイル（Git LFS）

- `.npy/.gguf/.safetensors` 等はLFSに載せる前提で `.gitattributes` を整備する。
- 「LFS pointerであるべきなのに違う」警告が出たファイルは、追跡方針を明確にして移行する。

---

## Definition of Done（完了条件）

- ブート時に自動起動し、進捗ログ＋rolling checkpointが機能する。
- ABCテストがA/B/Cで実行され、要約統計量＋エラーバー＋多重補正＋ANOVAがMD/LaTeXで再現可能に出力される。
- データセット出典・生成手順・バージョンがmanifestと文書に残っている。

