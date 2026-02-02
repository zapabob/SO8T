# 実裁E��針！Eoftware Engineering Best Practices�E�E
こ�E斁E��は、SO8T / AEGIS 系の「�E動�E学習！EFT+GRPO�E�」「ABCベンチ�Eーク」「統計解析（多重補正t検定＋ANOVA�E�」「Prism向けMD/LaTeX出力」「ブート時自動起動＋チェチE��ポイント保�E」を、E*保守性・再現性・安�E性**を最優先して実裁E��るため�E方針です、E
---

## 0. 最優先�E設計原剁E
1. **再現性�E�Eeproducibility First�E�E*
   - 乱数seed、評価プロンプト、抽出ロジチE��、データセチE��版、git commit hash を忁E��記録、E   - 生�E物�E�結果JSON/CSV/PNG/MD/TEX�E�から、同条件で再実行できることをDefinition of Doneとする、E2. **刁E���E�Eeparation of Concerns�E�E*
   - 収集・整形・学習�E評価・統計�E可視化・公開！EF/GH�E�を明確に刁E��し、責務墁E��を崩さなぁE��E3. **壊れにくさ�E�Eesumability / Fault-tolerance�E�E*
   - 電源断・OOM・ネットワーク断で止まってめE*checkpointから復帰**できる、E   - “途中結果を捨てなぁE��ことを標準挙動にする、E4. **賁E��制紁E�E明示�E�ETX3060 / RAM32GB�E�E*
   - VRAM 12GB前提�E�LoRA/QLoRA、grad checkpoint、低バチE��、offloadを基本戦略にする、E5. **観測性�E�Ebservability�E�E*
   - tqdm風の進捗＋構造化ログ�E�SQLite履歴�E�後述�E�で「何が起きたか」を後から追える、E
---

## 1. ブランチE��worktree 運用�E�EpenCode刁E���E�E
- **main**�E�安定運用�E�ブート�E動化、�E現可能ベンチ、HF/GH公開�E成果物�E�を保持、E- **OpenCode�E�Eit worktree�E�E*�E�実験�E検証・大規模改修・外部研究の統合を先行、E- **マ�Eジ原則**
  - OpenCode ↁEmain は「�E現可能性」と「ログ・ドキュメント」が揁E��た時点でPR/マ�Eジ、E  - 実験中の破壊的変更は main に入れなぁE��特にチE�Eタパス、チェチE��ポイント形式、評価条件�E�、E
---

## 2. リポジトリ構造・モジュール刁E��

### 2.1 チE��レクトリ責務（例！E- `scripts/utils/`�E�ブート起動、モニタ、チェチE��ポイント、E��捗記録�E�ランタイム機�E�E�E- `scripts/pipeline/`�E�統合パイプライン�E�データ→学習�E評価→レポ�Eト！E- `scripts/training/`�E�SFT/GRPO の実行器�E�ETX3060向け最適化を雁E��E��E- `scripts/evaluation/`�E�ABCチE��ト、LM-eval呼び出し、結果保孁E- `scripts/analysis/`�E�統計解析�E可視化・レポ�Eト生成！ED/LaTeX�E�E- `data/`�E�データ実体（巨大ファイル�E�！E`data/manifest/`�E�後述�E�ハチE��ュとメタ�E�E- `docs/`�E�RUNBOOK / 方釁E/ モチE��カーチE/ Prism用成果物�E�ED/TeX/PNG�E�E
### 2.2 1000行ルール
- 1ファイル1000行を目安に刁E���E�特に `scripts/evaluation/*` と `scripts/pipeline/*`�E�、E- 共通�E琁E�E `shared/` 相当（現状は `utils/` めE`scripts/utils/`�E�へ寁E��、E��褁E��許さなぁE��E
---

## 3. 設定管琁E��Eonfig�E�と秘寁E��報

- **設定�Eファイル匁E*�E�`config/*.json|yaml` に寁E��、CLIは config を上書きできる設計、E- **秘寁E��報はコミット禁止**�E�`.env.local` / OS環墁E��数を使用。ログにも�EさなぁE��E- **実行条件を記録**�E�実行時のconfigを忁E�� `results/<run_id>/config_snapshot.json` に保存、E
---

## 4. ロギング�E�進捗／SQL�E�永続追跡�E�E
### 4.1 ログ
- すべての主要�E琁E�E `logs/` に出力（例：`boot_pipeline_launcher.log`�E�、E- ログは「人間が読めるチE��スト」＋「機械が読めるJSON�E�任意）」�E二層、E
### 4.2 進捗！Eqdm風�E�E- “Simple English”で統一�E�例：`progress: |######----| training running`�E�、E- 監視UI�E�Eich等）�E任意だが、E*ログだけで復允E��能**な状態を残す、E
### 4.3 SQLite履歴�E�推奨�E�E- `logs/pipeline_progress.sqlite` に以下を保存！E  - run_id, timestamp, phase, seed, dataset_version, checkpoint_path, metrics(JSON)
- 再起動後�E再開時�E SQLite を参照して「最後�E正常checkpoint」を確定する、E
---

## 5. チェチE��ポイント設訁E
- **冁E�E�E�Eraining�E�E*�E�`checkpoints/latest_checkpoint.json` を原本として更新�E�原子書き込み推奨�E�、E- **外�E�E�Eoot/rolling�E�E*�E�E刁E��隔で3世代保持�E�Echeckpoints/rolling_snapshots/`�E�、E- checkpointには忁E��以下を含める�E�E  - git commit hash / config hash / dataset hash / seed / step / optimizer state要紁E
---

## 6. チE�EタセチE��の工学皁E��用�E�Eanifest�E�検証�E�E
- `data/manifest/` に吁E��ータセチE��の
  - SHA256、行数、スキーマ、ライセンス、生成スクリプト、生成日、用途！EFT/GRPO/Eval�E�E  を記録し、E*HF公開時に追跡可能**にする、E- 入力データは忁E��スキーマ検証�E�ESONLの忁E��Eey、role整合、E��さ上限�E�、E
---

## 7. 学習！EFT+GRPO�E�実裁E��針！ETX3060前提�E�E
- VRAM 12GB運用の既定！E  - LoRA/QLoRA、gradient checkpointing、低batch、忁E��に応じてCPU offload、E- OOM時�E「落ちる」ではなく「�E動で段階的に設定を落としてリトライ」、E- `TORCH_COMPILE_DISABLE` 等�E環墁E��数は、Windows安定動作�Eため統一して設定する、E
---

## 8. ABCチE��ト／統計解析（論文水準�E再現性�E�E
- **ABCチE��チE*�E�A/B/Cを同一条件�E�Erompt、抽出、seed、サンプル�E�で実行し、seed反復を忁E��、E- **統訁E*�E�E  - 主要E��Welch t-test�E�不等�E散�E�！EHolm もしく�E Bonferroni
  - 全体：one-way ANOVA�E�η²も併記！E- **成果物**�E�E  - `summary_statistics.csv`�E�Erismに近い形�E�E  - `docs/abc_prism_summary.md` / `.tex`�E�Erism整形用�E�E  - `docs/figures/*.png`�E�エラーバ�E�E�E
---

## 9. ドキュメント（運用・採用目線�E学術目線！E
- `docs/RUNBOOK.md`�E�実行手頁E��ブート、�E開、評価、�E開！E- `docs/IMPLEMENTATION_LOG.md`�E�実裁E��グ�E�いつ/何を/どぁE��えたか�E�E- 学術形式：MD�E�LaTeXの両方を用意し、引用は `References` また�E `thebibliography` で統一、E
---

## 10. 品質保証�E�テスト／CI�E�E
- 重要E��数�E�統計、E��計、チェチE��ポイント�E出力）�E unit test を置く、E- “動く”だけでなく、E*再実行して同じ結論が出めE*ことを検証頁E��にする、E
---

## 11. 大容量ファイル�E�Eit LFS�E�E
- `.npy/.gguf/.safetensors` 等�ELFSに載せる前提で `.gitattributes` を整備する、E- 「LFS pointerであるべきなのに違う」警告が出たファイルは、追跡方針を明確にして移行する、E
---

## Definition of Done�E�完亁E��件�E�E
- ブ�Eト時に自動起動し、E��捗ログ�E�rolling checkpointが機�Eする、E- ABCチE��トがA/B/Cで実行され、要紁E��計量�E�エラーバ�E�E�多重補正�E�ANOVAがMD/LaTeXで再現可能に出力される、E- チE�EタセチE��出典・生�E手頁E�Eバ�Eジョンがmanifestと斁E��に残ってぁE��、E

