"""
Exports SO8T checkpoints to GGUF using llama.cpp tooling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def merge_lora(base_model: Path, lora_path: Path, output: Path) -> None:
    base = torch.load(base_model, map_location="cpu")
    adapter = torch.load(lora_path, map_location="cpu")
    for key, value in adapter.items():
        if key in base:
            base[key] += value
        else:
            base[key] = value
    torch.save(base, output)


def convert_to_gguf(checkpoint: Path, output: Path) -> None:
    # Placeholder for downstream llama.cpp invocation
    torch.save({"checkpoint": str(checkpoint)}, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    temp = args.checkpoint
    if args.lora:
        merged = args.output.with_suffix(".merged.pt")
        merge_lora(args.checkpoint, args.lora, merged)
        temp = merged

    convert_to_gguf(temp, args.output)


if __name__ == "__main__":
    main()
