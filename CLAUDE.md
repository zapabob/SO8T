# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SO8T (Aegis-VSSI) is an autonomous AI research pipeline retraining **Borea-Phi-3.5-mini-Instruct-Jp** into a specialized reasoning model with "Quadrality" VSSI (Vector-Spinor-Spinor-Integration) architecture. The system implements AEGIS v3.0 for 24/7 continuous training, evaluation, and deployment, combining SO(8) NKAT theory-based adapters, DeepSeek-V3 GRPO, Sakana AI evolutionary optimization, and Unsloth 4-bit QLoRA (optimized for RTX 3060 x2).

## Build & Run Commands

### Full Pipeline Launch (requires admin)
```batch
setup_ab_test_automation.bat
auto_ab_test_pipeline.bat
```

### Continuous Operation (PowerShell)
```powershell
.\scripts\pipeline\run_aegis_continuous.ps1
```

### Manual Resume from Checkpoint
```bash
python scripts/pipeline/auto_resume_aegis.py
```

### Quick Verification
```batch
py -3 simple_rlpo_test.py
```

### CI Tests
```bash
python -m pytest tests/test_ci_smoke.py -v --cov=so8t --cov-report=xml
python tests/test_minimal.py
python tests/test_imports.py
```

### Linting
```bash
flake8 so8t/ src/ scripts/ --exclude scripts/training,src/training,scripts/conversion,src/eval,src/infra,src/core/inference/tests,src/core/models/so8t_residual_adapter_old.py --count --select=E9,F63,F7,F82 --show-source --statistics
black --check so8t/
isort --check-only so8t/
mypy so8t/ src/ --ignore-missing-imports --exclude "src/training|src/eval|src/infra|src/core/inference/tests|src/core/models/so8t_residual_adapter_old.py"
```

### GGUF Conversion
```bash
py external/llama.cpp-master/convert_hf_to_gguf.py {model_dir} --outfile D:/webdataset/gguf_models/{model_name}/{model_name}_Q8_0.gguf --outtype q8_0
```

### Ollama Model Testing
```bash
ollama create {model_name}:latest -f {modelfile_path}
ollama run {model_name}:latest "prompt"
```

## VSSI Quadruple Reasoning

All model inference uses a 4-step internal thinking process enclosed in `<think>...</think>` tags:
1. **Vector State** (`<think-task>`): Observation, raw facts, problem statement
2. **Positive Spinor** (`<think-analysis>`): Deduction, logical constructs, standard methodology
3. **Negative Spinor** (`<think-safety>`): Abduction, edge cases, safety checks, counter-narratives
4. **Quadrality Integration** (`<think-policy>`): Synthesis, final policy decision, golden ratio convergence

Rendering is handled by `src/utils/vssi_template.py` (`render_thinking()` function). The tag style is controlled by `SO8T_THINK_TAG_STYLE` env var (default: "xml"). Quadruple token mode via `SO8T_QUADRUPLE_TOKENS=1`.

## Current Pipeline State (Phase 3)

The integrated pipeline runs via `src/infrastructure/pipeline/integrated_moonshot_pipeline_2025_2026.py`. Ollama has been removed; all dataset fetching uses HF CLI. ArXiv/BioRxiv collection targets 100k papers (50k each) with strict VSSI tagging.

### Primary Pipeline Command
```powershell
$env:SO8T_BASE_MODEL="AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
$env:SO8T_ARXIV_COUNT="50000"
$env:SO8T_BIORXIV_COUNT="50000"
$env:SO8T_SKIP_OLLAMA="1"
py -3 src/infrastructure/pipeline/integrated_moonshot_pipeline_2025_2026.py
```

### Critical Files
- `src/infrastructure/pipeline/integrated_moonshot_pipeline_2025_2026.py` - Main orchestrator
- `src/training/train_unsloth_so8t.py` - Unsloth + GRPO training loop
- `src/data/processing/process_arxiv_biorxiv.py` - VSSI dataset generator
- `src/utils/vssi_template.py` - Thinking block renderer
- `src/utils/checkpoint_manager.py` - Rolling checkpoint + emergency save
- `logs/sft_progress.log` - SFT training progress

## Architecture

### Integrated Pipeline Phases
1. **Dataset Discovery / HF CLI Fetch** - ArXiv/BioRxiv 100k papers, NSFW safety, OSINT, MCP skills
2. **SFT/RLPO** - Unsloth 4-bit QLoRA training with rolling checkpoints
3. **Advanced Techniques** - mHC (Manifold Harmonic Correction), GRPO (DaGRPO variant), GRAPE position encoding, imatrix quantization, GGUF export
4. **Autonomous Research** - Sakana AI integrated agent
5. **HF CLI Upload** - Plots, GGUF, stats, model card

### Advanced Techniques
- **mHC**: Geometric latent space alignment via Manifold Harmonic Correction
- **GRPO**: Group Relative Policy Optimization (DeepSeek-style with DaGRPO customization)
- **GRAPE**: Position encoding variants (Multiplicative/Additive)
- **imatrix**: Importance matrix for quantization accuracy

### Source Layout
- `src/core/` - Core pipeline orchestration, model implementations (SO8T adapters, thinking model, ViT), inference (Ollama integration), checkpoint management
- `src/core/models/` - NKAT adapters (`so8t_residual_adapter.py`), MHC manifold (`mhc_manifold.py`), GRAPE positional encoding, thinking model
- `src/core/pipeline/` - Main orchestrators: `moonshot_pipeline.py`, `complete_so8t_pipeline.py`, safety validation, automated workflows
- `src/data/` - 230+ scripts for data collection (ArXiv, domain knowledge via Playwright, NSFW detection datasets, quadruple thinking datasets)
- `src/training/` - AEGIS v2/v3 training pipelines, QLoRA, auto-resume system, alpha gate annealing
- `src/evaluation/` - A/B testing framework, statistical analysis, benchmark suites (GSM8K, MATH, ELYZA-100, MMLU-JP), HF upload automation
- `src/infrastructure/external/` - Vendored dependencies: llama.cpp, lm-evaluation-harness, GRAPE
- `scripts/pipeline/` - Pipeline launchers (.ps1, .bat, .py)
- `docs/` - 100+ documentation files including model card, safety policy, runbooks, changelogs

### Safety Architecture
- **Dual-head design**: TaskHeadA (execution) + SafetyHeadB (judgment) with three-class output (ALLOW/REFUSE/ESCALATE)
- **PET Regularization**: Positional embedding tuning to preserve safety gates during fine-tuning
- **Non-commutative gates**: Safety check always runs before command execution (R_safe -> R_cmd)
- NSFW corpus is used exclusively for safety judgment training, not generation

### Checkpoint & Model Storage
All model outputs **must** be saved to `D:\webdataset`:
- `D:\webdataset\gguf_models\` - GGUF converted models
- `D:\webdataset\checkpoints\training\` - Training checkpoints
- `D:\webdataset\checkpoints\finetuning\` - Fine-tuning checkpoints
- `D:\webdataset\weights\` - Intermediate weights
- `D:\webdataset\models\final\` - Completed models

### Results Output
- `results/ab_test_results/` - Evaluation results and statistics
- `hf_upload_package/` - Ready-to-upload HF package

## Code Style

- PEP 8 with 100-char line limit (CI uses 127)
- **No emojis** in code, comments, or output - use plain text: `[OK]`, `[NG]`, `[START]`, etc. (prevents Windows Unicode encoding errors)
- Type hints required for public functions; Google-style docstrings
- Black for formatting, isort for imports, flake8 for linting, mypy for type checking
- Always use absolute paths, never relative
- UTF-8 encoding mandatory (`chcp 65001` for Windows batch scripts)
- All directories must be created with `mkdir(parents=True, exist_ok=True)`

## Implementation Logs

When creating implementation logs, save to `_docs/` with format: `yyyy-mm-dd_{worktree_name}_{feature_name}.md`. Detect worktree name from `git rev-parse --git-dir` (use "main" if not in a worktree). Each log entry must include progress tracking fields (implementation status, verification status, date, notes) and operational notes (data collection policy, NSFW corpus usage, /think endpoint handling).

## Key Conventions

- Windows-first development environment (PowerShell, .bat scripts, Task Scheduler)
- Bilingual project (English + Japanese)
- Rolling checkpoint system: 3-5 minute intervals with multi-slot recovery
- Auto-resume on power-on from last successful pipeline phase
- Direct Ollama testing preferred over Python test scripts
- After A/B testing, winning models must go through GGUF conversion -> Ollama import -> Japanese LLM performance testing
- `so8t/` is the primary importable package (used by CI linting and formatting checks)
- Hardware requirements: RTX 3060+ (12GB VRAM), 16GB+ RAM, 50GB+ disk on D:\ drive
