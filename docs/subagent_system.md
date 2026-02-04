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

These definitions are starting points; extend them as needed for new workflows.
