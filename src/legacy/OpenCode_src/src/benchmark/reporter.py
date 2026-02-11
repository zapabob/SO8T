"""Benchmark reporting utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import json

from .evaluator import BenchmarkResult
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report outputs."""

    output_dir: Path = Path("benchmark_results")


class BenchmarkReporter:
    """Writes benchmark results to disk."""

    def __init__(self, config: ReportConfig | None = None) -> None:
        self.config = config or ReportConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, results: Dict[str, BenchmarkResult], filename: str = "results.json") -> Path:
        path = self.config.output_dir / filename
        payload = {name: result.to_dict() for name, result in results.items()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved benchmark results to %s", path)
        return path
