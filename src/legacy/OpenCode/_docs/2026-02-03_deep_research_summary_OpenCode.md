# Deep Research Summary: KromHC + DGPO + ShinkaEvolve + Benchmarks
Created: 2026-02-03
Worktree: OpenCode

## Sources Reviewed
- KromHC paper (arXiv:2601.21579)
- MathForge / DGPO paper (arXiv:2601.20614)
- ShinkaEvolve paper (arXiv:2509.19349)
- mHC paper (arXiv:2512.24880)
- DeepSeekMath paper (arXiv:2402.03300)
- KromHC repository: https://github.com/wz1119/KromHC
- MathForge repository: https://github.com/AMAP-ML/MathForge
- ShinkaEvolve repository: https://github.com/SakanaAI/ShinkaEvolve
- mHC repository: https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness

## Research Findings
### KromHC
- Core idea: Kronecker factorization of doubly-stochastic residual matrices to guarantee exact double stochasticity while reducing parameter complexity to O(n^2 C).
- Repo implementation emphasizes compatibility with HC/mHC baselines and avoids custom kernels.
- Actionable for OpenCode: implement KromHC residual matrices as a drop-in replacement for mHC H_res, with configuration for factor sizes and Sinkhorn iterations.

### mHC
- Core idea: project residual matrices onto the Birkhoff polytope via Sinkhorn-Knopp.
- Repo implementation structure: H_res with Sinkhorn, H_pre/H_post using non-negative mixing.
- Actionable for OpenCode: keep an mHC-compatible mode for baseline comparisons, with configurable projection method.

### MathForge (DGPO + MQR)
- DGPO: difficulty-aware group advantage estimation to re-balance GRPO learning toward harder samples.
- MQR: multi-aspect question reformulation to generate harder variants while preserving correctness.
- Repo provides scripts and recipes for evaluation and dataset handling.
- Actionable for OpenCode: integrate difficulty estimation module, difficulty-weighted advantage computation, and MQR-based data augmentation.

### ShinkaEvolve
- Core idea: LLM-driven mutation + evolutionary search with island model and archive for open-ended program evolution.
- Repo uses Hydra for config separation and includes example workflows and tests.
- Actionable for OpenCode: add configuration isolation for evolution parameters, implement island migration and archive logging, and use LLM mutators as modular operators.

## GitHub Codebase Review Highlights
### KromHC repository
- Small, focused implementation targeting mHC/HC replacement.
- Likely easiest to port via a residual-matrix module and projection utility.

### MathForge repository
- Structured with scripts and evaluation workflows; best used as reference for DGPO integration and benchmarking process.

### ShinkaEvolve repository
- Emphasizes configurable evolutionary experiments and reproducible runs.
- Useful for adding training-time search over prompts, reward designs, or tool usage policies.

### mHC repository
- Clear reference for Sinkhorn-based double stochasticity and compatible residual pathways.

## Benchmark Plan (LM-eval-harness centered)
### Core LM-eval-harness integration
- Use LM-eval-harness for task orchestration and standardization.

### Mathematics and advanced reasoning
- MATH, MATH-500, GSM8K-Plus, AIME, Minerva-Math.

### Science and hard reasoning
- ARC-Challenge, GPQA.

### Coding
- HumanEval, MBPP, APPS-Intro.

### General reasoning and AGI proxies
- MMLU, BIG-bench Hard, TruthfulQA.

### Japanese
- JGLUE (MARC-ja, JSNLI), ELYZA-tasks-100.

## Integration Recommendations for zapabobouj-AEGIS-phi3.5-jp-v3.0
- Use KromHC residual matrices for the trainable model C while keeping A and B frozen.
- Apply DGPO with difficulty-aware sampling on math and reasoning benchmarks first, then expand to broader tasks.
- Use MQR to generate harder variants of math and reasoning problems.
- Use ShinkaEvolve for design-space search over reward shaping, tool policies, or data augmentation strategies.

## Next Implementation Steps
1. Add a KromHC/mHC compatibility flag to residual modules.
2. Extend DGPO difficulty estimation with calibration data from math benchmarks.
3. Wire MQR reformulations into data pipeline.
4. Implement ShinkaEvolve archive logging and island migration utilities.
5. Build LM-eval-harness runner config covering math, science, coding, AGI, and Japanese tasks.
