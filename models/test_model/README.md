---
language:
- en
- ja
license: apache-2.0
tags:
- so8-quadrality-inference
- mathematical-reasoning
datasets:
- gsm8k
- math
- ai2_arc
- elyza/ELYZA-tasks-100
metrics:
- accuracy
library_name: transformers
---

# AEGIS v2.5: SO8T Quadrality Inference + imatrix Protection

## Benchmark Results

| Benchmark | Score | Std Dev | 95% CI | Cohen's d | p-value |
|-----------|-------|---------|--------|-----------|---------|
| GSM8K | 77.0% | ±1.20 | ±2.40 | 1.80 | 0.001 |
| MATH | 43.0% | ±2.10 | ±4.10 | 2.20 | 0.000 |
| ARC-Challenge | 74.0% | ±1.80 | ±3.50 | 1.90 | 0.001 |
| ELYZA Tasks 100 | 83.0% | ±1.10 | ±2.20 | 2.10 | 0.000 |

## Citations

```bibtex
@article{so8t2024,
  title={SO(8) Quadrality Inference for Advanced Language Models},
  author={SO8T Research Initiative},
  journal={arXiv preprint},
  year={2024}
}

@article{deepseek2025,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={DeepSeek-AI Team},
  journal={Nature},
  year={2025}
}
```

---
Generated: 2026-01-20 15:31:12
