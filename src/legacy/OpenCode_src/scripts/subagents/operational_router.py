#!/usr/bin/env python3
"""Generate operational routing plan for subagents."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.subagents.registry import load_registry
from src.subagents.router import DynamicTaskRouter


TASKS = [
    {
        "name": "deep_research_arxiv_biorxiv",
        "task": "arXiv/BioRxiv API download for 2024-2026",
        "permissions": ["network-read", "write-data", "write-metadata"],
    },
    {
        "name": "deep_research_hf_cli",
        "task": "HF CLI dataset download with provenance",
        "permissions": ["network-read", "write-data", "write-metadata"],
    },
    {
        "name": "pdf_ocr_exam",
        "task": "PDF OCR for Japanese entrance exams (math/physics/chemistry)",
        "permissions": ["network-read", "write-data", "write-metadata"],
    },
    {
        "name": "cot_generation",
        "task": "Quadruple CoT generation with think tags and BF16 models",
        "permissions": ["read-data", "write-data", "gpu-use"],
    },
    {
        "name": "dataset_curation",
        "task": "Dataset cleaning, dedup, schema validation",
        "permissions": ["read-data", "write-data", "write-metadata"],
    },
    {
        "name": "evaluation_stats",
        "task": "ANOVA + Tukey benchmarks with p-values and error bars",
        "permissions": ["read-results", "write-reports", "gpu-use"],
    },
    {
        "name": "compliance_citation",
        "task": "Model card + README citation audit (research-only)",
        "permissions": ["read-docs", "write-docs", "write-reports"],
    },
]


def main():
    registry_path = PROJECT_ROOT / "config" / "subagents" / "registry.yaml"
    registry = load_registry(registry_path)
    router = DynamicTaskRouter(registry)
    routes = []
    for item in TASKS:
        decision = router.route_task(
            item["task"], strategy="parallel", required_permissions=item["permissions"]
        )
        routes.append({
            "name": item["name"],
            "task": item["task"],
            "required_permissions": item["permissions"],
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
    log_path = logs_dir / f"operational_routes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(routes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved operational routing plan: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
