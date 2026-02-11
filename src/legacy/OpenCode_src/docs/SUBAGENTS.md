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

## Permissions
- Use `--permissions` to require specific capabilities (e.g., `network-read`, `write-data`, `gpu-use`).
- The router filters subagents that lack required permissions.

```bash
python scripts/subagents/delegate_task.py "arXiv API download" --permissions network-read,write-data,write-metadata
```

### Deny/Allow policy
Permissions are further constrained by `config/subagents/policy.yaml`:
- **defaults**: base allow/deny
- **environments**: per-environment deny/allow (e.g., production)
- **subagents**: per-agent overrides

## Operational routing
Generate a practical routing plan for the production pipeline:

```bash
python scripts/subagents/operational_router.py
```

## Notes
- The compliance-citation subagent enforces citations in README/model cards.
- Evaluation subagent expects ANOVA + Tukey with p-values, power, and error bars.
- CoT generator subagent uses BF16 models and reloads between benchmarks.

