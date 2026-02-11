# SO8T Proof-of-Concept Overview

## Components
- dataset_synth.py: Generates synthetic ENV/CMD/SAFE dialogues with contract-driven labels.
- gents/so8t/model.py: Implements TinyNCGTransformerSO8T with SO(8) gating and PET loss.
- 	rain.py / eval.py / infer.py: End-to-end pipeline scripts.
- gents/cli.py: Command entry point for codexCLI orchestration.

## Experiment Flow
1. Generate data via python dataset_synth.py --count 3000.
2. Train the model with mixed precision using python train.py --config configs/train_default.yaml.
3. Evaluate and summarize results with python eval.py --checkpoint chk/so8t_default.pt and python scripts/summarize_results.py.

## Artifacts
- Checkpoints stored in chk/ with vocab + label metadata.
- Dataset splits written to data/ (JSONL + metadata).
- Training logs appended to chk/so8t_default_train_log.jsonl and summarized in chk/experiment_summary.md.
