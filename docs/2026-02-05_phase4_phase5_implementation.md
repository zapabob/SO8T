# 2026-02-05 Phase 4-5 実装ログ

## 作成ファイル

### Phase 4: 高度データ拡充パイプライン

- **ファイル**: `src/data/phase4_data_enrichment_pipeline.py`
- **機能**:
  - 薬物検知データ収集（PMDA, UNODC, WHO, 等）
  - NSFW検知データセット統合
  - 地政学OSINTデータ収集（2024-2026年国際情勢）
  - 科学・数学・CoTデータ取得（HF CLI経由）
  - MCP/Skill (Tool-calling) データ生成
  - ShareGPT形式への統一フォーマット変換

### Phase 5: 全自動再学習パイプライン

- **ファイル**: `src/training/phase5_auto_retraining_pipeline.py`
- **機能**:
  - Borea-Phi-3.5 をベースモデルとした重み凍結再学習
  - LoRA/QLoRA による効率的なアダプター学習
  - SFT + GRPO 統合学習フロー
  - 5分間隔ローリングチェックポイント（3世代）
  - 電源投入時自動再開対応
  - BF16/Flash-attention 最適化（RTX 3060 12GB対応）

### 統合オーケストレーター

- **ファイル**: `src/run_aegis_pipeline.py`
- **機能**:
  - Phase 4-6 の一括実行
  - `--phase` オプションで個別フェーズ指定可能
  - `--resume` オプションでチェックポイントからの再開

## 実行方法

```powershell
# 全フェーズ実行
python src/run_aegis_pipeline.py --phase all

# Phase 4 のみ
python src/run_aegis_pipeline.py --phase 4

# Phase 5 (再開モード)
python src/run_aegis_pipeline.py --phase 5 --resume
```
