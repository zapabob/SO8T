"""Artifact QA utilities for GGUF/imatrix outputs."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def summarize_artifacts(paths: Iterable[Path]) -> List[Dict]:
    summary = []
    for path in paths:
        if not path.exists():
            continue
        summary.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return summary


def collect_artifacts(directory: Path, patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))
    return sorted(set(files))


def write_report(report_path: Path, payload: Dict) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        __import__("json").dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path
