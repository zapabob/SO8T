#!/usr/bin/env python3
"""Aggregate ANOVA/Tukey statistics and generate summary artifacts."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return {}


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    evaluation_dir = project_root / "evaluation"
    logs_dir = project_root / "logs" / "evaluation"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    corrected_stats = _load_json(evaluation_dir / "corrected_benchmark_statistics.json")
    enhanced_stats = _load_json(evaluation_dir / "enhanced_statistical_evaluation_results.json")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "corrected_benchmark_statistics": corrected_stats,
        "enhanced_statistical_evaluation": enhanced_stats,
        "notes": [
            "ANOVA/Tukey aggregation ready. Populate with actual benchmark outputs.",
            "Ensure benchmarks output raw scores for ANOVA and Tukey post-hoc tests.",
        ],
    }

    summary_path = logs_dir / f"anova_tukey_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("ANOVA/Tukey summary saved: %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
