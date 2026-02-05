---
language:
  - ja
  - en
license: apache-2.0
tags:
  - llm
  - phi-3.5
  - japanese
  - so8t
  - quadrality
  - aegis
base_model: AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp
pipeline_tag: text-generation
---

# AEGIS-Phi-3.5-JP v3.0 (Model C)

## Model Description / モデル概要

**English:**
AEGIS-Phi-3.5-JP v3.0 is an advanced Japanese language model based on Borea-Phi-3.5-mini-Instruct-Jp, enhanced with SO8T Quadrality Reasoning framework. The model integrates specialized knowledge in geopolitics (2024-2026), scientific reasoning, and safety-aware responses.

**日本語:**
AEGIS-Phi-3.5-JP v3.0は、Borea-Phi-3.5-mini-Instruct-Jpをベースに、SO8T四重推論フレームワークで強化された高度な日本語言語モデルです。

## Training Methodology / 学習手法

| Component | Method |
|-----------|--------|
| Base Weight Preservation | LoRA/QLoRA Adapters |
| Supervised Fine-Tuning | ShareGPT format data |
| Reinforcement Learning | GRPO (Group Relative Policy Optimization) |
| Reasoning Enhancement | SO8T Quadrality (Scalar, Vector, +Spinor, -Spinor) |

---

## Benchmark Results / ベンチマーク結果

Industry-standard benchmarks evaluated using [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

### Summary Statistics / 要約統計量

| Model | Mean | SD | 95% CI | N |
|-------|------|-----|--------|---|

*Note: Mean ± SD across all benchmarks. 95% CI = 95% Confidence Interval.*

### Detailed Benchmark Scores / 詳細ベンチマークスコア

| Benchmark | Model A | Model B | Model C | Δ(B→C) |
|-----------|---------|---------|---------|--------|

*Note: Δ(B→C) = improvement from Model B to Model C. ** = large effect (|Δ| > 0.05), * = medium effect (|Δ| > 0.02).*

### Statistical Significance / 統計的有意性

| Comparison | t-statistic | p-value | Significant (α=0.05) | Significant (α=0.01) |
|------------|-------------|---------|----------------------|----------------------|

### Interpretation / 解釈

- **α = 0.05**: Standard significance threshold (p < 0.05)
- **α = 0.01**: Stringent significance threshold (p < 0.01)
- **Cohen's d**: Effect size where |d| > 0.8 = large, 0.5-0.8 = medium, 0.2-0.5 = small

---

## Dataset Sources / データセット出典

### Geopolitics (2024-2026) / 地政学
- Venezuela crisis and US-Latin America relations
- Ukraine war progression and European security
- Japan-China relations (diplomatic, economic security, national security)

### Technology / テクノロジー
- GPU shortage and memory/SSD price dynamics
- AI/LLM developments: Opus 4.5, Codex, Claude Code, MCP, Skill OSS

### Culture / カルチャー
- Gundam franchise:
  - SEED FREEDOM (2024)
  - GQuuuuuuX (Director: Kazuya Tsurumaki, Studio Khara × Sunrise, 2025)
  - Hathaway Part 2 (2025-2026)

---

## Citation / 引用

```bibtex
@misc{aegis-phi35-jp-v3,
  author = {zapabobouj},
  title = {AEGIS-Phi-3.5-JP v3.0: Quadrality Reasoning Enhanced Japanese LLM},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/zapabobouj/AEGIS-phi3.5-jp-v3.0}}
}
```

## References / 参考文献

1. Microsoft. (2024). Phi-3.5 Technical Report.
2. AXCXEPT. (2024). Borea Japanese Language Model.
3. EleutherAI. (2024). lm-evaluation-harness. GitHub.
4. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
5. Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
```python
6. DeepSeek-AI. (2024). DeepSeek-V3 Technical Report. (GRPO)
7. Unsloth AI. (2024). Unsloth: Lightweight and fast LLM fine-tuning. GitHub.
8. Gerganov, G., et al. (2024). llama.cpp: Importance Matrix (imatrix) Quantization. GitHub.
9. Akiba, T., et al. (2024). Evolutionary Optimization of Model Merging. arXiv:2403.13187. (Sakana AI)
10. mHC: Multi-Head Control/Consistency for Reasoning Models.
```
Generated: 2026-02-06T02:56:27.552240
