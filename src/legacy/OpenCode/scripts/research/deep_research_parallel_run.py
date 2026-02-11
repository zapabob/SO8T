#!/usr/bin/env python3
"""Parallel DeepResearch acquisition runner.

This script reads the latest DeepResearch task split and executes each task
with configurable parallelism. It records outcomes to logs so the pipeline
can proceed even when credentials are missing.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List


logger = logging.getLogger(__name__)


def _load_tasks(logs_dir: Path) -> List[Dict]:
    task_files = sorted(logs_dir.glob("deep_research_tasks_*.json"), reverse=True)
    if not task_files:
        return []
    return json.loads(task_files[0].read_text(encoding="utf-8"))


def _run_task(task: Dict) -> Dict:
    name = task.get("name", "unknown")
    result = {
        "name": name,
        "started_at": datetime.now().isoformat(),
        "status": "skipped",
        "details": "No executor bound",
    }

    if name == "web_search_codex_gemini":
        result["status"] = "pending"
        result["details"] = (
            "Requires Codex web search + GeminiCLI integration. "
            "Set GEMINI_API_TOKEN and provide search endpoints."
        )

    return result


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    logs_dir = project_root / "logs" / "subagents"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    tasks = _load_tasks(logs_dir)
    if not tasks:
        logger.warning("No DeepResearch task split found. Run splitter first.")
        return 0

    parallelism = int(os.environ.get("DEEP_RESEARCH_PARALLELISM", "4"))
    results: List[Dict] = []

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_map = {executor.submit(_run_task, task): task for task in tasks}
        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as exc:
                task = future_map[future]
                results.append({
                    "name": task.get("name", "unknown"),
                    "started_at": datetime.now().isoformat(),
                    "status": "error",
                    "details": str(exc),
                })

    output_path = logs_dir / f"deep_research_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("DeepResearch parallel run results saved: %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
