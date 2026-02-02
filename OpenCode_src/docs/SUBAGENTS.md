# Subagent System (ClaudeCode-style)

This project includes a lightweight subagent registry to coordinate research, OCR, CoT generation, curation, evaluation, and compliance workflows.
All subagent tasks inherit the **research-only** and **citation-required** policy.

## Files
- config/subagents/registry.yaml
- config/subagents/agents/*.yaml
- src/subagents/registry.py
- src/subagents/router.py
- scripts/subagents/delegate_task.py

## Quick start
```bash
python scripts/subagents/delegate_task.py "arXiv/BioRxiv search and download" --strategy parallel
python scripts/subagents/delegate_task.py "PDF OCR for 東大/京大 past exams" --strategy parallel
python scripts/subagents/delegate_task.py "ANOVA + Tukey for ABC benchmarks" --strategy single
```

## Notes
- The compliance-citation subagent enforces citations in README/model cards.
- Evaluation subagent expects ANOVA + Tukey with p-values, power, and error bars.
- CoT generator subagent uses BF16 models and reloads between benchmarks.

