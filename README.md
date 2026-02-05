---
language:
  - en
  - ja
license: apache-2.0
tags:
  - so8-quadrality-inference
  - mathematical-reasoning
  - continual-learning
  - enhanced-moonshot-pipeline
  - deepseek-grpo
  - mhc-manifold
  - geometric-scaling
  - imatrix-quantization
  - autonomous-research
  - sakana-ai
  - windows-automation
datasets:
  - gsm8k
  - math
  - elyza/ELYZA-tasks-100
  - moonshot-domain-knowledge
  - moonshot-arxiv-papers
metrics:
  - accuracy
  - statistical_significance
library_name: transformers
pipeline_tag: text-generation
---

# AEGIS v3.0: Full-Automation & Continuous Operations Edition

# AEGIS v3.0: 全自動・継続運転エディション

**Autonomous Research Pipeline with DeepSeek-V3 (GRPO), Sakana AI (Evolution), and SO8T Quadrality Reasoning.**
**DeepSeek-V3 (GRPO)、Sakana AI (進化モデル)、および SO8T 四重推論を統合した自律研究パイプライン。**

---

## 🚀 Overview / 概要

AEGIS v3.0 is a state-of-the-art AI development pipeline designed for 24/7 continuous operation. Based on the **Borea-Phi-3.5-mini-Instruct-Jp** model, it integrates cutting-edge reasoning techniques and a robust automated infrastructure for research, training, and evaluation.

AEGIS v3.0 は、24時間365日の連続稼働を前提に設計された最先端の AI 開発パイプラインです。**Borea-Phi-3.5-mini-Instruct-Jp** モデルを基盤とし、最新の推論技術と、研究・学習・評価のための堅牢な自動化インフラを統合しています。

---

## 🛠️ Key Features / 主な機能

### 1. Continuous Operation & Reliability / 継続運転と信頼性 [NEW]

- **Rolling Checkpoints**: Automatic 5-minute saving with 3-generation rotation to prevent data loss.
- **Power-on Auto-Resume**: Automatic recovery tracking from the last successful phase upon system restart.
- **Windows Startup Integration**: Seamless background execution starting immediately on power-on.

- **ローリングチェックポイント**: 5分間隔の自動保存と3世代の世代交代により、データ損失を防止。
- **電源投入時自動再開**: システム再起動時に、最後の中断フェーズから自動的に処理を継続。
- **Windows スタートアップ統合**: 電源投入と同時にバックグラウンドで全自動実行を開始。

### 2. Advanced Reasoning Architectures / 高度な推論アーキテクチャ

- **DeepSeek-V3 GRPO**: Group Relative Policy Optimization for emergent mathematical reasoning.
- **Sakana AI Evolution**: Evolutionary optimization logic inspired by ShinkaEvolve for autonomous research.
- **SO8T Quadrality**: Four-perspective logic (Algebraic, Geometric, Analytic, Topological) for deep understanding.

- **DeepSeek-V3 GRPO**: 創発的な数学的推論を実現するグループ相対ポリシー最適化。
- **Sakana AI Evolution**: ShinkaEvolve に着想を得た自律研究のための進化的最適化ロジック。
- **SO8T Quadrality**: 代数・幾何・解析・位相の4つの視点による深い理解。

### 3. High-Fidelity Deliverables / 高精度な成果物自動生成

- **Advanced HF CLI Upload**: Automated one-click upload of Safetensors, BF16 GGUF, and cited benchmark plots.
- **Academic Citation Graphs**: Statistical plots featuring explicit citations for DeepSeek (GRPO) and Sakana AI.

- **高度 HF CLI アップロード**: Safetensors, BF16 GGUF、および引用付きグラフの一括自動アップロード。
- **学術的引用グラフ**: DeepSeek (GRPO) や Sakana AI の引用を明記した高度な統計プロット。

---

## 📊 Benchmark Results / ベンチマーク結果

| Benchmark / ベンチマーク | AEGIS v3.0 (Borea-Phi-3.5) | Microsoft Phi-3.5 Baseline | Improvement / 改善 |
| :----------------------- | :------------------------: | :------------------------: | :----------------: |
| **GSM8K**                |       **78.2** ± 1.2       |         72.9 ± 1.4         |      +5.3 pts      |
| **MATH**                 |       **45.8** ± 2.8       |         32.6 ± 2.3         |     +13.2 pts      |
| **ELYZA Tasks 100**      |       **84.5** ± 1.1       |         79.6 ± 1.4         |      +4.9 pts      |
| **MMLU (Japanese)**      |       **71.2** ± 1.3       |         64.5 ± 1.7         |      +6.7 pts      |

*Results are based on Phase 6 Statistical Benchmark with 10 random seeds.
*Phase 6 の 10 試行統計ベンチマークに基づく結果。

---

## 🏃 Quick Start / クイックスタート

### Automatic Execution (PowerShell) / 全自動実行 (PowerShell)

To start the full pipeline with auto-resume and monitoring:
自動再開と監視機能を備えた全工程を開始するには：

```powershell
# Run the continuous automation wrapper
# 継続自動化ラッパーを実行
.\scripts\pipeline\run_aegis_continuous.ps1
```

### Manual Resume / 手動での再開

If you need to resume manually from the latest checkpoint:
最新のチェックポイントから手動で再開する場合：

```bash
# Execute the auto-resume entry point
# 自動再開エントリポイントを実行
python scripts/pipeline/auto_resume_aegis.py
```

---

## 📚 Tech Citations / 学術的引用

AEGIS v3.0 integrates methodologies from the following groundbreaking works:
AEGIS v3.0 は、以下の画期的な研究成果の知見を統合しています。

- **GRPO**: DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024).
- **Evolution**: Akiba et al., "Evolutionary Optimization of Model Merging" (2024).
- **SO8T**: SO8T Quadrality & Manifold-Constrained Hyper-Connections (2025-2026).

---

## ⚖️ License / ライセンス

This project is licensed under the Apache License 2.0.
本プロジェクトは Apache License 2.0 の下で公開されています。

---

_Developed by the SO8T Research Initiative (2025-2026)._
