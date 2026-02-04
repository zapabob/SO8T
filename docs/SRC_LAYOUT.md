# src/ Layout (SO8T)

This project is transitioning toward a `src/`-first layout for clarity and tooling compatibility.

## Target Structure

```
src/
  core/        # core utilities, model patches, loaders
  data/        # dataset ingestion + registry
  agents/      # agent logic
  training/    # training pipelines
  eval/        # evaluation and benchmarks
  infra/       # checkpointing, auto-resume, progress, ops
```

## Compatibility Wrappers

To avoid breaking existing scripts, **wrapper modules** are provided in `src/` that import the current
implementations from `scripts/` and `utils/`. This enables gradual migration while keeping legacy
entrypoints functional.

## New Pipelines

- `src/training/borea_adapter_pipeline.py` ? Model B adapter training (LoRA + SFT + GRPO + mHC/GRAPE)

Legacy shims:
- `scripts/training/borea_adapter_pipeline.py`
- `scripts/training/retrain_borea_phi35_with_so8t.py`

## Next Migration Steps (optional)

1. Move `scripts/agents/*` into `src/agents/` and fix imports.
2. Move shared helpers from `utils/` into `src/core/` and `src/infra/`.
3. Update entrypoints under `scripts/` to import from `src/*` only.
