# Handover: Moonshot Pipeline Restart & Multiprocessing Fix

**Date:** 2026-02-06
**Status:** Pipeline Restarted / Monitoring Phase 1
**Previous Agent:** Antigravity

## Context

The user is restarting the **Integrated Moonshot Pipeline (AEGIS-v3.0)** with advanced features enabled (`SO8T`, `mHC`, `GRPO`, `imatrix`).
The previous runs were failing or skipping phases due to:

1.  **Residual Checkpoints:** Old `.json` and `.ptr` files caused the pipeline to think it was already finished.
2.  **Windows Multiprocessing Error:** `ModuleNotFoundError: No module named 'UnslothSFTTrainer'` occurred during the SFT phase because `dill`/`multiprocessing` could not serialize the trainer class correctly when `dataset_num_proc > 1` on Windows.

## Actions Taken

1.  **Cleanup:**
    - Deleted all checkpoints in `data/collected_2025_2026/*.json` and `*.ptr`.
    - Deleted marker files in `data/sunset_pipeline/markers/`.
    - Removed old log files (`run_log.txt`, `moonshot_pipeline_2025_2026.log`).
2.  **Code Fix (Critical):**
    - Modified `src/training/train_unsloth_so8t.py` to enforce `dataset_num_proc = 1` when running on Windows.
    - Affected methods: `run_sft_training` (~line 780) and `run_grpo_training` (~line 970).
    - _Reasoning:_ This bypasses the serialization issues with Unsloth's compiled cache on Windows.
3.  **Pipeline Launch:**
    - Executed the runner script with full feature flags enabled:
    ```powershell
    $env:PYTHONPATH="."; $env:SO8T_SO8_ENABLE="1"; $env:SO8T_MHC_ENABLE="1"; ...
    py -3 src/core/run_moonshot_pipeline_2025_2026.py --collect-new-data --use-unsloth --mcp-api-skill --enable-mhc --enable-so8 > run_log.txt 2>&1
    ```

## Current Status

- The pipeline process has been started (Background).
- The logs (`run_log.txt`) should now show `[PHASE START] collect`.

## Next Steps for New Agent

1.  **Monitor Execution:**
    - Run `Get-Content run_log.txt -Tail 50 -Wait` to watch the progress.
    - Verify that it successfully passes Phase 1 (Data Collection).
2.  **Verify SFT Phase:**
    - Watch for the transition to Phase 5 (SFT).
    - **Crucial:** Confirm that the `UnslothSFTTrainer` error does not recur. If it does, ensure the `dataset_num_proc=1` fix was actually applied and saved.
3.  **Verify Advanced Integration:**
    - Confirm Phase 6 (Advanced) starts and initializes `SO8T`/`GRPO`.
4.  **Artifact Verification:**
    - Once complete, check for model artifacts in `models/moonshot_2025_2026` or `data/sunset_pipeline/checkpoints`.

## Key Files

- `src/training/train_unsloth_so8t.py`: Contains the multiprocessing fix.
- `src/core/run_moonshot_pipeline_2025_2026.py`: Main runner script.
- `run_log.txt`: Stdout/Stderr capture of the current run.
- `moonshot_pipeline_2025_2026.log`: Internal pipeline application log.

## Known Environment Constraints

- **OS:** Windows 11
- **Shell:** PowerShell
- **Python:** 3.12 (`py -3`)
- **Encoding:** UTF-8 preferred for logs, but system default is CP932/UTF-16LE. Be careful when reading files with `Get-Content`.
