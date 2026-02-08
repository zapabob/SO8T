"""
Data ingestion helpers for Phase 4 datasets.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def load_jsonl(path: Path, max_rows: int | None = None) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def build_instruction_sample(instruction: str, input_text: str, output: str) -> Dict:
    return {"instruction": instruction, "input": input_text, "output": output}


def ensure_paths(paths: Iterable[Path]) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
