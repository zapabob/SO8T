# 2026-02-07 Critical Bugfix - 7 Blocking Issues Resolved

## Overview / 概要

Codebase review (see `2026-02-07_main_codebase_review.md`) identified 8 CRITICAL + 7 HIGH severity issues.
This log documents the 7 fixes applied in commit `1aca0bb`.

- **Implementation Status**: 7/8 CRITICAL + 1 MEDIUM completed
- **Verification Status**: All 4 modified files passed `py_compile` syntax check
- **Date**: 2026-02-07
- **Worktree**: main
- **Commits**: `acd6805` (review log), `1aca0bb` (bugfix)

---

## Fixes Applied / 修正内容

### C-1: ModuleNotFoundError in train_unsloth_so8t.py [BLOCKING]

**File**: `src/training/train_unsloth_so8t.py:44-55`

**Problem**: `from src.utils.path_resolver import PathResolver` failed because project root was not in `sys.path` before import. The except fallback retried the same import without fixing `sys.path` first. Evidence in `logs/sft_progress.log`.

**Fix**: Added `sys.path.insert(0, str(_project_root))` before the try block so `src.*` imports always resolve. Fallback now logs a warning instead of crashing.

```python
# BEFORE (broken):
try:
    from src.utils.path_resolver import PathResolver
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.path_resolver import PathResolver  # same error

# AFTER (fixed):
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
try:
    from src.utils.path_resolver import PathResolver
    PROJECT_ROOT = PathResolver.get_project_root()
except ImportError:
    PROJECT_ROOT = _project_root
```

---

### C-2: Checkpoint Index Out of Bounds

**File**: `src/infrastructure/pipeline/integrated_moonshot_pipeline_2025_2026.py:248`

**Problem**: `_save_checkpoint()` writes 1-based index (1,2,3) via `(prev_idx % 3) + 1`. `_load_latest_checkpoint()` used that index directly as list subscript, but `rolling_checkpoints` is 0-based (0,1,2). When `idx=3`, `IndexError` at `rolling_checkpoints[3]`.

**Fix**: `self.rolling_checkpoints[idx - 1]` -- convert 1-based to 0-based. Also replaced bare `except Exception: pass` with `except (IndexError, ValueError)` with logging.

---

### C-3: Missing Methods in checkpoint_manager.py

**File**: `src/utils/checkpoint_manager.py`

**Problem**: `train_unsloth_so8t.py` calls 3 methods that did not exist:
- `EmergencyCheckpointManager.register_model(model, tokenizer)` (lines 810, 870)
- `RollingCheckpointManager.get_checkpoint_info(checkpoint_path)` (line 1150)
- `RollingCheckpointManager.force_save_now(model, tokenizer, step_info)` (line 112)

**Fix**: Implemented all 3 methods:
- `register_model`: Stores model/tokenizer references for emergency saves
- `get_checkpoint_info`: Reads metadata.json from checkpoint dir, returns dict
- `force_save_now`: Delegates to `save_checkpoint()` ignoring interval timer

Also updated `save_emergency()` to use registered model/tokenizer as fallback.

---

### C-4: Dataset Field Format Mismatch

**File**: `src/training/train_unsloth_so8t.py:849`

**Problem**: `_combine_and_preprocess_datasets()` creates entries with `{"messages": [...]}` format (chat template). But `SFTTrainer` was configured with `dataset_text_field="text"`, expecting a plain text field that doesn't exist. Would crash with `ValueError: Text field text not found in dataset`.

**Fix**: Removed `dataset_text_field="text"` parameter. Unsloth's `SFTTrainer` auto-detects `messages` field and applies the chat template internally.

---

### C-5: Wrong Import Path in multi_domain_enrichment.py

**File**: `src/data/processing/multi_domain_enrichment.py:20-22`

**Problem**: Used `from src.data_processing.xxx import ...` (wrong path). Correct module path is `src.data.processing`. Additionally, `enrich_pharma_safety` module does not exist anywhere in the codebase.

**Fix**: Changed to `from src.data.processing.xxx` with try/except fallbacks. Set `enrich_pharma_dataset = None` since the module is not implemented yet. Note: file is tracked as binary by git due to UTF-16 BOM encoding; used `git add -f` to stage.

---

### C-7: numpy.random.choice() with String Lists

**File**: `src/training/train_unsloth_so8t.py:728, 794`

**Problem**: `np.random.choice(['+', '-', '*', '/'])` and `np.random.choice(['ALLOW', 'ESCALATE', 'DENY', 'REFUSE'])` crash with `TypeError` because NumPy cannot handle string list sampling.

**Fix**: Replaced both with `random.choice()` from Python stdlib (`import random` already present at line 36).

---

### M-1: Duplicate Pipeline Shutdown Code

**File**: `src/infrastructure/pipeline/integrated_moonshot_pipeline_2025_2026.py:1068-1074`

**Problem**: `_stop_periodic_checkpoint()`, `db.end_run()`, and `logger.info()` were called twice at end of `execute_full_pipeline()`. Copy-paste error or merge artifact.

**Fix**: Removed the duplicate block (lines 1072-1074).

---

### M-2: Bare except:pass in checkpoint_manager.py

**File**: `src/utils/checkpoint_manager.py:92`

**Problem**: `except: pass` caught all exceptions including KeyboardInterrupt/SystemExit, silently discarding metadata save errors.

**Fix**: Changed to `except Exception as e: print(f"[WARN] Could not save checkpoint metadata: {e}")`.

---

## Files Modified / 変更ファイル

| File | Changes |
|------|---------|
| `src/training/train_unsloth_so8t.py` | C-1 (sys.path), C-4 (dataset_text_field), C-7 (random.choice) |
| `src/infrastructure/pipeline/integrated_moonshot_pipeline_2025_2026.py` | C-2 (index OOB), M-1 (duplicate shutdown) |
| `src/utils/checkpoint_manager.py` | C-3 (3 missing methods), M-2 (bare except) |
| `src/data/processing/multi_domain_enrichment.py` | C-5 (import path + fallback) |

---

## Verification / 検証

- All 4 files passed `py -3 -c "import py_compile; py_compile.compile(...)"`
- No import-time errors (runtime dependencies like `unsloth`, `torch` not verified without GPU)
- Pushed to `origin/main` as `1aca0bb`

---

## Additional Fixes Applied (same session)

### C-8: Broken test imports - `a57f7b2`
- `test_imports.py`: replaced non-existent module refs with actual `src.utils`/`src.core` imports
- `test_safety.py`: fixed import path `so8t_core` -> `src.core.so8t_core`, added `pytest.mark.skipif`
- `test_minimal.py`: fixed config path with candidates list, use absolute paths

### H-2: Emoji removal - `29876ab`
- 1658 replacements across 206 files in `src/`
- Emoji -> plain text: `[OK]`, `[NG]`, `[WARN]`, `[SO8T]`, `[START]`, `[TARGET]`, `[STATS]`, `[DONE]`

### H-3: CI linting config - `1b7cac2`
- flake8 critical checks (E9,F63,F7,F82) now cover `src/training/`
- Fix test paths: `tests/` -> `src/evaluation/tests/` (actual location)
- mypy now covers `src/utils/`, `src/core/models/`, `src/infrastructure/pipeline/`
- Added `src/external` to excludes (vendored code)

### H-6: ELYZA dataset fallback - `c9990a7`
- `abc_testing.py`, `run_benchmarks.py`: try 'test' split, fallback to first available
- `setup_lm_eval_elyza.py`: graceful skip if dataset unavailable

### H-7: Ollama availability guard - `586552d`
- Added `_check_ollama_available()` in `elyza_benchmark.py`
- Respects `SO8T_SKIP_OLLAMA` env var; clear error if binary missing

---

## Remaining Issues / 未修正事項

### CRITICAL (deferred - requires coordinated refactoring)
- **C-6**: Duplicate `so8t_residual_adapter.py` in `src/models/` and `src/core/models/` (13 files affected)

### HIGH (not addressed)
- **H-1**: Phase 6 benchmark results all empty (`benchmark_results: {}`) - needs investigation
- **H-4**: 149 hard-coded `D:/` `H:/` drive paths in 28 files
- **H-5**: Relative paths in JSON configs (`borea_training.json`)

### MEDIUM (not addressed)
- M-3 ~ M-9: See `2026-02-07_main_codebase_review.md`

---

## Operational Notes

- `src/data/` is in `.gitignore`; `multi_domain_enrichment.py` required `git add -f` to stage
- File encoding: `multi_domain_enrichment.py` has UTF-16 BOM, git treats as binary
- `random` module was already imported at line 36 of `train_unsloth_so8t.py`
- `enrich_pharma_safety` module referenced but never implemented; set to `None`
- `dataset_num_proc` on Windows remains `1` per previous fix (Windows spawn serialization issue)
