# 2026-02-05 Sakana AI方式 汎用科学研究・OSINT AIエージェント実装

## 概要

Sakana AIの研究成果（AI Scientist 2024/2025, ShinkaEvolve 2025）に基づき、汎用科学研究およびOSINT AIエージェントを実装。

## 参考論文・技術

| 技術                | 出典                              | 主要貢献                   |
| ------------------- | --------------------------------- | -------------------------- |
| **AI Scientist**    | Sakana AI (2024) arXiv:2408.06292 | 完全自動研究ライフサイクル |
| **AI Scientist-v2** | Sakana AI (2025) ICLR Workshop    | ピアレビュー通過           |
| **ShinkaEvolve**    | Sakana AI (2025) Apache-2.0       | 効率的進化最適化           |

## 作成ファイル

### 統合エージェント

- **ファイル**: `src/agents/sakana_ai_integrated_agent.py`
- **クラス構成**:

```
SakanaAIIntegratedAgent
├── ShinkaEvolveEngine (進化的最適化)
│   ├── NoveltyJudge (新規性判定)
│   ├── BanditLLMEnsemble (UCB1ベースLLM選択)
│   ├── Adaptive Parent Sampling
│   ├── Code Novelty Rejection
│   └── Island-based Evolution
│
├── AIScientistAgent (科学研究)
│   ├── generate_ideas()
│   ├── conduct_experiment()
│   ├── write_paper()
│   ├── automated_review()
│   └── run_research_cycle()
│
└── OSINTAIAgent (インテリジェンス)
    ├── collect_intelligence()
    ├── cross_verify()
    ├── generate_analysis() (SO8T四重推論)
    └── run_osint_cycle()
```

## ShinkaEvolve 3つの革新

1. **Adaptive Parent Sampling**
   - 探索と活用のバランス
   - 温度付きソフトマックス選択
   - UCB1スコア計算

2. **Code Novelty Rejection**
   - ハッシュベース重複検出
   - コサイン類似度チェック
   - 新規性スコア付与

3. **Bandit-Based LLM Ensembling**
   - 複数LLMプロバイダ（GPT, Gemini, Claude, DeepSeek）
   - ε-greedy + UCB1選択
   - 動的報酬更新

## AI Scientist 8フェーズ研究ライフサイクル

1. **IDEATION**: アイデア生成
2. **LITERATURE_REVIEW**: 文献調査
3. **IMPLEMENTATION**: 実装（Agentic Tree Search）
4. **EXPERIMENTATION**: 実験実行
5. **ANALYSIS**: 分析
6. **WRITING**: 論文執筆
7. **REVIEW**: 自動ピアレビュー
8. **REFINEMENT**: 改善ループ

## OSINT機能

- **マルチソース情報収集**: Reuters, AP, 防衛白書, JAXA, arXiv, GDELT
- **信頼性評価**: ソース別信頼度スコア
- **クロス検証**: 複数ソース一致チェック
- **SO8T四重推論分析**:
  - `<think-task>`: タスク定義
  - `<think-analysis>`: 情報分析
  - `<think-safety>`: セキュリティ考慮
  - `<think-policy>`: 政策提言

## 実行方法

```bash
# 統合エージェント起動
py -3 src/agents/sakana_ai_integrated_agent.py

# 科学研究モード
from src.agents.sakana_ai_integrated_agent import SakanaAIIntegratedAgent
agent = SakanaAIIntegratedAgent()
result = agent.run_scientific_research("LLM推論能力向上")

# OSINT分析モード
result = agent.run_osint_analysis("ウクライナ情勢2024-2026")

# ハイブリッド分析
result = agent.run_hybrid_analysis("AI規制動向")
```

## 出力ディレクトリ

- `data/ai_scientist_research/`: 研究ログ
- `data/osint_intelligence/`: OSINT分析ログ

## 既存資産との統合

- `src/data/research/autonomous_researcher.py`: 既存AutonomousResearcher
- `src/external/OpenCode_src/scripts/training/shinka_evolve.py`: 基礎ShinkaEvolve
- `src/data/unified_dataset_collection_pipeline.py`: データ収集
- `src/training/grpo_self_reflection_training.py`: GRPO訓練
