"""
Auto-resume utilities for long-running pipelines.
Stores run state as JSON and reloads on restart.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_state(state_path: Path) -> Dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state_path: Path, state: Dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def update_phase(state: Dict, phase: str, status: str, progress: float = 0.0) -> Dict:
    phases = state.setdefault("phases", {})
    phases[phase] = {"status": status, "progress": float(progress)}
    return state
