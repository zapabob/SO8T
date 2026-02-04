# Implementation Log - 2026-02-05

## Scope
- Remove shim dependency by updating imports to src.*
- Productionize Phase4 data sources config
- Add ANOVA visualization outputs
- Add HF publish GitHub Actions workflow

## Changes
- Updated imports across repo (excluding scripts/ shims) from scripts.* -> src.*
- config/phase4_pipeline.yaml now uses real API/PDF/HF sources
- src/eval/stat_report.py now outputs boxplot/violin/mean+sd plots
- Added workflow: .github/workflows/hf_publish.yml
- Added docs: docs/HF_PUBLISH_WORKFLOW.md, updated PHASE4/EVAL docs and README

## Notes
- Shims kept for backward CLI entrypoints, but core imports now target src.*
- HF workflow expects secret HF_TOKEN

