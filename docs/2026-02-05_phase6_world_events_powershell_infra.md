# 2026-02-05 Phase 6 実装・2024-2026年世界情勢データ・PowerShellインフラ

## 作成ファイル

### 2024-2026年世界情勢データ収集

- **ファイル**: `src/data/collect_world_events_2024_2026.py`
- **収集対象**:
  - **地政学**: ベネズエラ情勢、ウクライナ戦争推移、日中対立（外交・経済安保・国家安保）
  - **テクノロジー**: メモリ/SSD高騰、GPU不足、Opus 4.5、Codex、MCP、Skill OSS化
  - **カルチャー**: ガンダム（SEED FREEDOM、GQuuuuuuX、ハサウェイ第2部）、批評

### Phase 6: 統計的ベンチマーク

- **ファイル**: `src/evaluation/phase6_statistical_benchmark.py`
- **機能**:
  - Model A/B/C 三者比較（lm-eval-harness, DeepEval, ELYZA-100）
  - ANOVA（一元配置分散分析）
  - Cohen's d（効果量）
  - 学術形式モデルカード自動生成

### PowerShell パイプラインマネージャー

- **ファイル**: `src/infrastructure/aegis_pipeline_manager.ps1`
- **機能**:
  - 5分間隔ローリングチェックポイント（3個）
  - 電源投入時自動再開（スタートアップ登録）
  - tqdm風進捗バー表示
  - Phase 4-6 統合実行

## 実行方法

### Python オーケストレーター

```powershell
py -3 src/run_aegis_pipeline.py --phase all
py -3 src/run_aegis_pipeline.py --phase 6
```

### PowerShell マネージャー

```powershell
# 通常実行
.\src\infrastructure\aegis_pipeline_manager.ps1 -Phase all

# 自動再開インストール
.\src\infrastructure\aegis_pipeline_manager.ps1 -Install

# 再開モード
.\src\infrastructure\aegis_pipeline_manager.ps1 -Resume

# アンインストール
.\src\infrastructure\aegis_pipeline_manager.ps1 -Uninstall
```
