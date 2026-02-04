#!/usr/bin/env python3
"""Auto-resume runner for Borea adapter training.
Reads run_state.json and resumes missing phases.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.infra.auto_resume import load_state
from src.training.borea_adapter_pipeline import main as run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/borea_training.json")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output or "models/borea_adapter")
    state_path = output_dir / "run_state.json"
    state = load_state(state_path)
    phases = state.get("phases", {})

    if phases.get("sft", {}).get("status") == "completed" and phases.get("grpo", {}).get("status") != "completed":
        return run_pipeline()  # default phase=full will skip missing if dataset absent
    return run_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
