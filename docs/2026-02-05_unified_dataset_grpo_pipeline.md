# 2026-02-05 データセット収集・加工・GRPO自己内省パイプライン実装

## 作成ファイル

### 統合データセット収集パイプライン

- **ファイル**: `src/data/unified_dataset_collection_pipeline.py`
- **機能**:
  - `SO8TQuadralityFormatter`: 四重推論CoT形式変換器
    - 数学視点: algebraic, geometric, analytic, topological
    - VSSI視点: think-task, think-analysis, think-safety, think-policy
  - 薬理学データ収集（研究目的）
  - NSFW/安全検知データ変換
  - Skill/MCPツールコーリング
  - CoTデータセット統合
  - GRPO報酬データセット生成

### GRPO自己内省トレーニング

- **ファイル**: `src/training/grpo_self_reflection_training.py`
- **機能**:
  - `GRPOConfig`: GRPO設定（group_size, kl_penalty, reward_weights等）
  - `RewardFunction`: 報酬関数コレクション
    - correctness: 正確性
    - reasoning_quality: 推論品質
    - quadrality_balance: 四重推論バランス
    - safety_compliance: 安全性準拠
    - format_adherence: 形式遵守
  - `GRPOSelfReflection`: 自己内省ループ
  - `OSINTAgentGRPO`: OSINT特化報酬関数
    - source_diversity: ソース多様性
    - temporal_awareness: 時間認識
    - geopolitical_context: 地政学コンテキスト
    - cross_verification: クロス検証

## 実行方法

```bash
# データセット収集
py -3 src/data/unified_dataset_collection_pipeline.py

# GRPO自己内省テスト
py -3 src/training/grpo_self_reflection_training.py
```

## 出力

- `data/unified_so8t_dataset/unified_so8t_YYYYMMDD_HHMMSS.jsonl`
- `data/unified_so8t_dataset/grpo_reward_YYYYMMDD_HHMMSS.jsonl`
- `data/unified_so8t_dataset/dataset_stats_YYYYMMDD_HHMMSS.json`

## 関連既存資産

- `src/data/osint_source_collector.py`
- `src/data/collect_drug_pharmaceutical_detection_dataset.py`
- `src/infrastructure/pipeline/safety/nsfw_drug_detection_qlora_training_data_pipeline.py`
- `config/osint_sources.yaml`
