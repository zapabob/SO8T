# 引き継ぎメタプロンプト: AEGIS-v3.0 フルパイプライン継続

## 1. コンテキスト

### プロジェクト概要

- **リポジトリ**: `c:\Users\downl\Desktop\SO8T` (zapabob/SO8T)
- **目標**: AEGIS-phi3.5-mini-jp-v3.0 の完全な再学習とベンチマーク測定
- **ベースモデル**: `AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp`

### 高度統合技術スタック

```
AEGIS-v3.0 Pipeline
├── 高度学習技術
│   ├── Unsloth 4-bit LoRA SFT/GRPO
│   ├── mHC (Manifold Hyperbolic Curvature)
│   ├── GRAPE (Group Representational Position Encoding)
│   ├── SO8T Residual Adapter（四重推論）
│   └── imatrix 量子化
│
├── 進化的最適化エンジン (Sakana AI)
│   ├── ShinkaEvolveEngine
│   └── AI Scientist 2 研究ライフサイクル
│
└── インテリジェンス
    └── OSINTAIAgent（マルチソース収集・SO8T四重推論分析）
```

---

## 2. 現在のステータス

### ✅ 完了済み

| Phase | 内容                                       | ステータス |
| ----- | ------------------------------------------ | ---------- |
| **1** | 環境整備                                   | ✅ 完了    |
| -     | 5分間隔ローリングチェックポイント（3世代） | ✅         |
| -     | 電源投入時自動再開 (Windows Startup)       | ✅         |
| -     | 進捗管理・エラーログ表示 (PowerShell)      | ✅         |

### ⏳ 未完了

| Phase | 内容                                           | ステータス |
| ----- | ---------------------------------------------- | ---------- |
| **2** | データ収集・加工（OSINT + HF CLI）             | ⏳ 未開始  |
| **3** | 本格的再学習（mHC/GRPO/GRAPE/SO8T/imatrix）    | ⏳ 未開始  |
| **4** | Sakana AI 統合エージェント実行                 | ⏳ 未開始  |
| **5** | 統計ベンチマーク（ANOVA/Cohen's d ABC テスト） | ⏳ 未開始  |
| **6** | HF アップロード                                | ⏳ 未開始  |

---

## 3. 重要ファイル

### パイプライン制御

- `scripts/pipeline/run_aegis_continuous.ps1` - 自動継続運転 PowerShell
- `scripts/pipeline/auto_resume_aegis.py` - 自動再開エントリーポイント
- `src/infrastructure/pipeline/integrated_moonshot_pipeline_2025_2026.py` - メインオーケストレーター

### 学習

- `src/training/train_unsloth_so8t.py` - Unsloth SFT/GRPO 学習スクリプト
- `src/infrastructure/config/borea_training.json` - 学習設定

### エージェント

- `src/agents/sakana_ai_integrated_agent.py` - Sakana AI 統合エージェント

### 評価

- `src/evaluation/phase6_statistical_benchmark.py` - 統計ベンチマーク

### 実装ログ

- `_docs/2026-02-06_AEGIS_Continuous_Operation_Enhancement.md`
- `_docs/2026-02-05_AEGIS_v3_Implementation.md`
- `docs/2026-02-05_sakana_ai_integrated_agent.md`

---

## 4. 次のアクション

### Phase 2: データ収集・加工

```powershell
py -3 src/data/processing/osint_source_collector.py
py -3 src/data/processing/hf_cli_dataset_fetch.py --base-dir data/hf_downloads
```

### Phase 3: 本格的再学習

```powershell
$env:SO8T_USE_UNSLOTH = "1"
py -3 src/training/train_unsloth_so8t.py --phase full --config src/infrastructure/config/borea_training.json
```

### Phase 4: Sakana AI エージェント

```powershell
py -3 src/agents/sakana_ai_integrated_agent.py
```

### Phase 5: 統計ベンチマーク

```powershell
py -3 src/evaluation/phase6_statistical_benchmark.py
```

### Phase 6: HF アップロード

```powershell
py -m huggingface_hub.commands.huggingface_cli upload zapabobouj/AEGIS-phi3.5-jp-v3.0 ./models/moonshot_2025_2026 --repo-type model
```

---

## 5. 特記事項

### 既知の問題

- **パイプラインのスタブ動作**: 環境変数 `SO8T_USE_UNSLOTH=1` が設定されていないと、学習フェーズがスタブ（ダミー）として動作する
- **Unsloth 依存**: `pip install unsloth[colab-new]` が必要
- **GPU 要件**: RTX 3060 (12GB VRAM) で動作確認済み

### 環境変数

```powershell
$env:SO8T_USE_UNSLOTH = "1"
$env:SO8T_DRYRUN = "0"
$env:SO8T_HF_UPLOAD = "1"
$env:SO8T_GRAPE_VARIANT = "multiplicative"
$env:SO8T_CHECKPOINT_INTERVAL = "300"  # 5分
$env:SO8T_CHECKPOINT_ROLLING = "3"     # 3世代
```

### ユーザールール

- 実装完了時は `_docs/yyyy-mm-dd_機能名.md` として実装ログを保存する
- コードは Rust 2024 / Python ベストプラクティスで記述
- 1000行超のファイルは分割する

---

## 6. 参考資料

| 技術          | 出典                       | 実装ファイル                    |
| ------------- | -------------------------- | ------------------------------- |
| AI Scientist  | Sakana AI arXiv:2408.06292 | `sakana_ai_integrated_agent.py` |
| ShinkaEvolve  | Sakana AI Apache-2.0       | 同上                            |
| GRAPE         | arXiv:2512.07805           | `grape_position_encoding.py`    |
| mHC           | SO8T 独自                  | `mhc_manifold.py`               |
| SO8T 四重推論 | SO8T 独自                  | `so8t_residual_adapter.py`      |
