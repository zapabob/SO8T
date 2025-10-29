"""
Dataset preparation utilities for SO8T LoRA fine-tuning jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def _load_samples(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def prepare_dataset(input_dir: Path, output_path: Path) -> None:
    seen: set[str] = set()
    merged: List[Dict[str, str]] = []

    for file in input_dir.glob("*.jsonl"):
        for sample in _load_samples(file):
            prompt = sample.get("prompt", "").strip()
            response = sample.get("response", "").strip()
            if not prompt or not response:
                continue
            fingerprint = f"{prompt}|{response}"
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            merged.append({"prompt": prompt, "response": response})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for sample in merged:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_dataset(args.input_dir, args.output)


if __name__ == "__main__":
    main()
