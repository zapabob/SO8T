# 2026-02-07 Codebase Review - SO8T (Aegis-VSSI)

## Overview / 概要

Full codebase review of SO8T autonomous AI pipeline. 50+ source files analyzed.
SO8Tコードベースの包括的レビュー。50以上のソースファイルを分析。

- **Implementation Status**: Review completed
- **Verification Status**: Issues identified, fixes pending
- **Date**: 2026-02-07
- **Reviewer**: Claude Code (Opus 4.6)
- **Worktree**: main

---

## Current Pipeline State / パイプライン現状

- Phase 5 (SFT) previously completed but with workarounds
- Phase 6 (Benchmarking) produced **empty results** (`benchmark_results: {}` for all 3 models)
- Advanced features (SO8T, mHC, GRPO, GRAPE) integrated but **NOT enabled** in recent runs
- Current uploaded model `AEGIS-phi3.5-jp-v3.0` is standard SFT + LoRA only
- SFT training blocked by `ModuleNotFoundError: No module named 'src.utils'` (see logs/sft_progress.log)

---

## CRITICAL Issues (P0) - Must Fix Before Next Run

### C-1: ModuleNotFoundError in train_unsloth_so8t.py [BLOCKING]

**File**: `src/training/train_unsloth_so8t.py:46-55`
**Evidence**: `logs/sft_progress.log` lines 11-22

```
ModuleNotFoundError: No module named 'src.utils'
```

**Root Cause**: Fallback block retries same import without modifying sys.path first.

**Fix**: Add at top of file before any imports:
```python
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
```

---

### C-2: Checkpoint Index Out of Bounds

**File**: `src/infrastructure/pipeline/integrated_moonshot_pipeline_2025_2026.py:248`

- `rolling_count = 3` generates index `idx = 1, 2, or 3` (1-based)
- `rolling_checkpoints` list uses indices 0-2 (0-based)
- When `idx=3`, `rolling_checkpoints[3]` raises `IndexError`

**Fix**: `self.rolling_checkpoints[idx - 1]` (convert to 0-based)

---

### C-3: Missing Methods in checkpoint_manager.py

**File**: `src/utils/checkpoint_manager.py`

Called from `train_unsloth_so8t.py` but not defined:
- `EmergencyCheckpointManager.register_model()` (lines 810, 870)
- `RollingCheckpointManager.get_checkpoint_info()` (line 1150)
- `RollingCheckpointManager.force_save_now()` (line 112)

**Impact**: `AttributeError` at runtime, training crashes.

---

### C-4: Dataset Field Format Mismatch

**File**: `src/training/train_unsloth_so8t.py`

- Line 688: `_combine_and_preprocess_datasets()` creates `{"messages": [...]}` format
- Line 847: `SFTTrainer(dataset_text_field="text")` expects `text` field
- Mismatch causes `ValueError: Text field text not found in dataset`

**Fix**: Remove `dataset_text_field="text"` for chat template format, or change data format to use `text` field.

---

### C-5: Wrong Import Path in multi_domain_enrichment.py

**File**: `src/data/processing/multi_domain_enrichment.py:20-22`

```python
from src.data_processing.process_arxiv_biorxiv import ...  # WRONG
```

**Correct**: `src.data.processing.process_arxiv_biorxiv`

---

### C-6: Duplicate Adapter Locations

Two copies of `so8t_residual_adapter.py`:
1. `src/models/so8t_residual_adapter.py` (older)
2. `src/core/models/so8t_residual_adapter.py` (newer)

13 files import from inconsistent paths. Silent version mismatch risk.

---

### C-7: numpy.random.choice() with Strings

**File**: `src/training/train_unsloth_so8t.py:726, 792`

```python
np.random.choice(['+', '-', '*', '/'])    # CRASH
np.random.choice(['ALLOW', 'ESCALATE'])   # CRASH
```

**Fix**: Use `random.choice()` from Python stdlib.

---

### C-8: Broken Test Imports

| Test File | Missing Module | Impact |
|-----------|---------------|--------|
| `test_imports.py:27-34` | `agents.so8t.model_safety`, `safety_losses`, `shared.data` | Always fails |
| `test_safety.py:3-4` | `so8t_core.self_verification`, `so8t_core.triality_heads` | Always fails |
| `test_minimal.py:37` | `aegis_v2_test_config.json` (missing file) | FileNotFoundError |

---

## HIGH Issues (P1) - Should Fix Soon

### H-1: Empty Benchmark Results

**File**: `src/evaluation/results/phase6_industry/benchmark_results_20260206_221505.json`

All 3 models (Baseline, Borea, AEGIS) returned `"benchmark_results": {}`.
Phase 6 ran but produced no actual measurements.

---

### H-2: Emoji Usage in 20+ Files

Per CLAUDE.md: "No emojis in code" (causes `UnicodeEncodeError` on Windows CP932).

Found in:
- `src/core/models/so8t_residual_adapter.py:151,165`
- `src/evaluation/*.py` (multiple files)
- `src/utils/check_aegis_data.py`, `check_environment.py`

Replace with: `[OK]`, `[NG]`, `[WARN]`, `[SO8T]`

---

### H-3: CI Excludes All Training Code

**File**: `.github/workflows/ci.yml`

```yaml
flake8 ... --exclude scripts/training,src/training,...
mypy ... --exclude src/training|...
bandit ... -x scripts/training,src/training,...
```

Training bugs are **never caught** by CI. The blocking ModuleNotFoundError (C-1) went undetected.

---

### H-4: Hard-Coded Drive Paths (149 Occurrences in 28 Files)

`D:/webdataset`, `H:/from_D/webdataset` hard-coded throughout.
Not portable to different machines. No environment variable fallback.

---

### H-5: Relative Paths in JSON Configs

**File**: `src/infrastructure/config/borea_training.json`

```json
"base_model": "models/Borea-Phi-3.5-mini-Instruct-Jp"  // RELATIVE
```

Violates CLAUDE.md: "Always use absolute paths, never relative"

---

### H-6: ELYZA Dataset No Fallback

**Files**: `abc_testing.py:382`, `run_benchmarks.py:505`, `setup_lm_eval_elyza.py:44`

No error handling if ELYZA-tasks-100 becomes unavailable or changes split structure.

---

### H-7: Ollama Dependency Still in Evaluation

**File**: `src/evaluation/elyza_benchmark.py:22-49`

Uses `subprocess.run(["ollama", ...])` despite CLAUDE.md stating "Ollama has been removed".

---

## MEDIUM Issues (P2)

| ID | File | Issue |
|----|------|-------|
| M-1 | `integrated_moonshot_pipeline.py:1068-1074` | Duplicate shutdown code (checkpoint stopped twice, db end_run called twice) |
| M-2 | `checkpoint_manager.py:92` | Bare `except: pass` - catches KeyboardInterrupt, hides errors |
| M-3 | `integrated_moonshot_pipeline.py:427,447` | `subprocess.run(check=False)` for critical data collection |
| M-4 | `train_unsloth_so8t.py:197,1055` | Duplicate `_load_reward_strategy_map()` method definition |
| M-5 | `run_aegis_continuous.ps1:133-136` | Retry loop without exponential backoff |
| M-6 | `grape_position_encoding.py:46` | Learnable freq unbounded - risk of NaN via `torch.exp()` |
| M-7 | `so8t_thinking_model.py:16-17` | Wrong import paths (`src.models` instead of `src.core.models`) |
| M-8 | `build_quadrality_think_dataset.py:21` | Missing import fallback (unlike `convert_to_quadrality_cot.py`) |
| M-9 | MODEL_CARD.md | Empty benchmark tables |

---

## LOW Issues (P3)

| ID | File | Issue |
|----|------|-------|
| L-1 | `src/so8t/` | Package essentially empty (only checkpointing.py) |
| L-2 | `test_ci_smoke.py` | Meaningless test (`assert True`) |
| L-3 | Pipeline files | No environment variable validation module |
| L-4 | `aegis_startup.bat:6` | Hard-coded user path |
| L-5 | `so8t_residual_adapter.py:88` | Norm clipping threshold 1.0 undocumented |
| L-6 | `vssi_template.py` | Stub implementation (incomplete quadrality) |
| L-7 | `hf_cli_dataset_fetch.py:78` | No validation of downloaded data |
| L-8 | `multi_domain_enrichment.py:29-42` | Silent JSONL parse failures (no logging) |
| L-9 | Various | Inconsistent import styles (absolute vs relative vs sys.path) |

---

## Positive Findings / 良好な点

- Rolling checkpoint system: Robust 3-generation design with 5-min intervals
- SO8T adapter: FP32 precision enforcement + NaN recovery with identity fallback
- PathResolver / ConfigLoader: Properly structured with UTF-8 and fallback
- PipelineDB (SQLite): Clean run tracking with phase/dataset/result metadata
- MHC Manifold: Clean Sinkhorn-Knopp implementation with proper dtype preservation
- Auto-resume on power failure: Checkpoint index system functional
- VSSI thinking model: Proper quadruple reasoning implementation
- `so8t/checkpointing.py`: Well-structured TrainerCallback with retry logic

---

## Test Coverage Gaps

| Module | Test Coverage | Status |
|--------|--------------|--------|
| `so8t/checkpointing.py` | None | No dedicated tests |
| `src/data/processing/` (20 files) | None | 0 dedicated tests |
| `src/evaluation/` (60 files) | Minimal | Broken imports |
| `src/infrastructure/pipeline/` | None | 0 integration tests |
| `src/training/` | None | Excluded from CI |
| `src/core/models/` | None | 0 unit tests |
| `src/utils/` | None | 0 unit tests |

Estimated overall test coverage: **<5%**

---

## Recommended Fix Priority

### Week 1 (Immediate)
1. Fix ModuleNotFoundError in train_unsloth_so8t.py (C-1) - 30 min
2. Fix checkpoint index OOB (C-2) - 15 min
3. Add missing checkpoint methods (C-3) - 1 hour
4. Fix dataset field mismatch (C-4) - 30 min
5. Fix import path in multi_domain_enrichment.py (C-5) - 5 min
6. Fix numpy.random.choice (C-7) - 15 min
7. Remove emoji usage (H-2) - 2 hours

### Week 2 (Short-term)
8. Consolidate adapter locations (C-6) - 3 hours
9. Fix broken test files (C-8) - 1 hour
10. Update CI to include training code (H-3) - 30 min
11. Investigate empty benchmarks (H-1) - 2 hours
12. Remove Ollama from evaluation (H-7) - 1 hour

### Week 3+ (Medium-term)
13. Convert hard-coded paths to env vars (H-4) - 4 hours
14. Add integration test suite - 2 days
15. Add environment variable validation module - 3 hours
16. Standardize import strategy - 3 hours
17. Expand unit test coverage - 1 week

---

## Operational Notes

- Data collection policy: ArXiv/BioRxiv 100k target via HF CLI
- NSFW corpus: Safety judgment training only, never generation
- /think endpoint: Controlled by `SO8T_THINK_TAG_STYLE` env var
- Previous ArXiv JSONL had 0 valid samples (`top_0_papers`) - needs re-collection
- Many HF datasets inaccessible (renamed/gated/removed) - see sft_progress.log
- `dataset_num_proc=1` enforced on Windows to avoid multiprocessing serialization errors
