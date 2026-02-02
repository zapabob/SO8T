#!/usr/bin/env python3
"""Split DeepResearch workload into subagent tasks and log plan."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.subagents.registry import load_registry
from src.subagents.router import DynamicTaskRouter


TASK_GROUPS = [
    {
        "name": "arxiv_ml_cs_ai_2024_2026",
        "task": "arXiv ML/CS/AI 2024-2026 top-cited download",
        "permissions": ["network-read", "write-data", "write-metadata"],
    },
    {
        "name": "biorxiv_2024_2026",
        "task": "BioRxiv 2024-2026 API download",
        "permissions": ["network-read", "write-data", "write-metadata"],
    },
    {
        "name": "hf_cli_cot_sources",
        "task": "HF CLI: MCP/skill/CoT datasets download",
        "permissions": ["network-read", "write-data", "write-metadata"],
    },
    {
        "name": "jp_national_universities_pdf",
        "task": "Japanese national university exam PDFs (math/physics/chemistry)",
        "permissions": ["network-read", "write-data", "write-metadata"],
    },
    {
        "name": "jp_common_test_pdf",
        "task": "Common Test PDFs (math/physics/chemistry/biology/English/Japanese)",
        "permissions": ["network-read", "write-data", "write-metadata"],
    },
    {
        "name": "prep_school_pdf",
        "task": "Kawai/Sundai/Yozemi 2024-2026 exam PDFs",
        "permissions": ["network-read", "write-data", "write-metadata"],
    },
    {
        "name": "lean4_mathlib",
        "task": "Lean4 mathlib dataset for formal proofs",
        "permissions": ["network-read", "write-data", "write-metadata"],
    },
    {
        "name": "quad_cot_generation",
        "task": "Quadruple CoT generation with think tags (BF16)",
        "permissions": ["read-data", "write-data", "gpu-use"],
    },
    {
        "name": "dataset_curation",
        "task": "Data cleaning/dedup for quadruple CoT",
        "permissions": ["read-data", "write-data", "write-metadata"],
    },
    {
        "name": "citations",
        "task": "Model card + README citations for research-only compliance",
        "permissions": ["read-docs", "write-docs", "write-reports"],
    },
]


def main():
    registry = load_registry(PROJECT_ROOT / "config" / "subagents" / "registry.yaml")
    router = DynamicTaskRouter(registry)
    plan = []
    for group in TASK_GROUPS:
        decision = router.route_task(
            group["task"], strategy="parallel", required_permissions=group["permissions"]
        )
        plan.append({
            "name": group["name"],
            "task": group["task"],
            "required_permissions": group["permissions"],
            "decision": {
                "strategy": decision.strategy,
                "reasoning": decision.reasoning,
                "assignments": [
                    {
                        "subagent": a.subagent_name,
                        "task_portion": a.task_portion,
                        "score": a.score,
                        "capabilities": a.capabilities,
                    }
                    for a in decision.assignments
                ],
            },
        })

    logs_dir = PROJECT_ROOT / "logs" / "subagents"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"deep_research_tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DeepResearch task plan saved: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
