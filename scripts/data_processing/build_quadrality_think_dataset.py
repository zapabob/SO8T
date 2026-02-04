#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Quadrality <think> dataset from integrated JSONL.

Outputs samples with:
  output = <think>...</think><final>...</final>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "so8t-mmllm" / "src"))

from models.thinking_tokens import format_thinking_output


def build_quadrality_think(instruction: str, user_input: str, answer: str) -> str:
    # Simple deterministic quadrality template (placeholder)
    algebraic = f"Algebraic: Identify symbols, definitions, and formal structure of: {instruction}"
    geometric = f"Geometric: Consider spatial/structural intuition for: {user_input[:200]}"
    analytic = "Analytic: Compute/estimate and check consistency; focus on quantitative aspects."
    topological = "Topological: Focus on invariants, relations, and high-level structure."
    thinking = "\n".join([algebraic, geometric, analytic, topological])
    return format_thinking_output(thinking=thinking, final=answer, use_redacted=True)


def convert(input_path: Path, output_path: Path) -> int:
    converted = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            instruction = sample.get("instruction", "四重推論の観点から説明してください。")
            user_input = sample.get("input", "")
            answer = sample.get("output", "")
            if not answer:
                answer = sample.get("text", "")
            if not answer:
                continue

            out_text = build_quadrality_think(instruction, user_input, answer)
            new_sample: Dict[str, str] = {
                "instruction": instruction,
                "input": user_input,
                "output": out_text,
                "metadata": {
                    "source": sample.get("metadata", {}).get("source", "integrated"),
                    "quadrality_think": True,
                },
            }
            f_out.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
            converted += 1
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description="Build quadrality <think> dataset")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    count = convert(args.input, args.output)
    print(f"[OK] Converted {count} samples -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
