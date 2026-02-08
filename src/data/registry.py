"""Dataset registry for Phase 4 multi-domain ingestion."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_registry(path: Path | None = None) -> Dict:
    if yaml is None:
        return {}
    path = path or (PROJECT_ROOT / "config" / "data_sources.yaml")
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def list_source_files(registry: Dict) -> List[Path]:
    files: List[Path] = []
    for entry in registry.get("sources", {}).values():
        for item in entry.get("files", []) or []:
            files.append(PROJECT_ROOT / item)
    return files
