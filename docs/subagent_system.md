# SO8T Subagent System

This project now includes a lightweight ClaudeCode-style subagent framework for coordinating specialized AI personas during development.

## Structure

```
subagents/
  definitions/        # YAML definitions for each subagent
  definitions.py      # Data models
  registry.py         # Registry + matching
  router.py           # Routing strategies
  manager.py          # Project configuration + routing
  validator.py        # Validation helpers
  task.py             # Task + routing data models
```

Project-level configuration lives in `config/subagents.yaml`.

## Quick Start

List available subagents:

```bash
python tools/subagent_cli.py list
```

Validate definitions:

```bash
python tools/subagent_cli.py validate
```

Delegate a task:

```bash
python tools/subagent_cli.py delegate "evaluate ABC benchmarks" \
  --routing-strategy parallel \
  --tags "benchmark,statistics"
```

Generate a schedule from `config/subagent_tasks.yaml`:

```bash
python tools/subagent_cli.py schedule
```

Create a new subagent:

```bash
python tools/subagent_cli.py create security-reviewer \
  --role "Security Auditor" \
  --expertise "OWASP,access control" \
  --capability "vulnerability_scanning:Scan code for issues" \
  --trigger-pattern "security|vulnerability"
```

Update project config:

```bash
python tools/subagent_cli.py config \
  --context "AEGIS-phi3.5mini-jp-v3.0" \
  --enable-subagents "security-reviewer,performance-analyzer"
```

## Included Subagents

- **security-reviewer** — security and safety checks
- **performance-analyzer** — profiling and optimization
- **data-pipeline-engineer** — dataset ingestion/QA
- **research-specialist** — literature review and design risk analysis
- **evaluation-analyst** — benchmarking and statistical testing
- **deployment-manager** — packaging and release orchestration
- **architecture-integrator** — mHC/SO8T/GRAPE integration support
- **rl-trainer** — GRPO/RL training oversight
- **quantization-engineer** — imatrix/GGUF calibration and packaging

These definitions are starting points; extend them as needed for new workflows.

## Pipeline Integration

`scripts/pipeline/integrated_moonshot_pipeline_2025_2026.py` routes each phase
through the subagent manager and logs the recommended assignments. Routing decisions
are stored in rolling checkpoints under `subagent_routing`.

The pipeline also auto-generates a schedule (`results/subagent_schedule.json`)
from `config/subagent_tasks.yaml` when `SO8T_SUBAGENT_SCHEDULE=1` (default).
