"""
Split the synthetic SO8T safety dataset into train/val/test shards.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

EXPECTED_LABELS = {"ALLOW", "ESCALATION", "DENY"}


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    @classmethod
    def from_args(cls, train: float, val: float, test: float) -> "SplitConfig":
        total = train + val + test
        if not math.isclose(total, 1.0, rel_tol=1e-3):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.3f}")
        return cls(train=train, val=val, test=test)

    def cumulative(self, size: int) -> Tuple[int, int]:
        train_end = int(size * self.train)
        val_end = train_end + int(size * self.val)
        # Ensure at least one example per split when possible
        train_end = min(max(train_end, 1), size - 2) if size >= 3 else size
        val_end = min(max(val_end, train_end + 1), size - 1) if size >= 2 else size
        return train_end, val_end


def load_samples(path: Path) -> List[dict]:
    samples: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            validate_sample(sample, line_no)
            samples.append(sample)
    if not samples:
        raise ValueError(f"No samples loaded from {path}")
    return samples


def validate_sample(sample: dict, line_no: int) -> None:
    missing = [key for key in ("id", "text", "expected_label") if key not in sample]
    if missing:
        raise ValueError(f"Line {line_no}: missing required keys {missing}")
    label = sample["expected_label"]
    if label not in EXPECTED_LABELS:
        raise ValueError(f"Line {line_no}: expected_label '{label}' not in {sorted(EXPECTED_LABELS)}")


def split_samples(samples: List[dict], config: SplitConfig, seed: int) -> Tuple[List[dict], List[dict], List[dict]]:
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    train_end, val_end = config.cumulative(len(shuffled))
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    return train, val, test


def write_jsonl(path: Path, samples: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")


def describe_split(name: str, samples: Iterable[dict]) -> str:
    labels = Counter(sample["expected_label"] for sample in samples)
    total = sum(labels.values())
    breakdown = ", ".join(f"{label}:{count}" for label, count in sorted(labels.items()))
    return f"{name}: {total} samples ({breakdown})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Split synthetic SO8T safety dataset into train/val/test shards.")
    parser.add_argument("--input", type=Path, required=True, help="Path to safety_1000.jsonl")
    parser.add_argument("--train-out", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--val-out", type=Path, default=Path("data/val.jsonl"))
    parser.add_argument("--test-out", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = SplitConfig.from_args(args.train_ratio, args.val_ratio, args.test_ratio)
    samples = load_samples(args.input)
    train, val, test = split_samples(samples, config, args.seed)
    write_jsonl(args.train_out, train)
    write_jsonl(args.val_out, val)
    write_jsonl(args.test_out, test)

    print(describe_split("train", train))
    print(describe_split("val", val))
    print(describe_split("test", test))


if __name__ == "__main__":
    main()
