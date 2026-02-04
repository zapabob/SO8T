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


def build_quadrality_think(instruction: str, user_input: str, answer: str, sample: Dict[str, Any]) -> str:
    """
    Advanced SO8T Quadrality Reasoning Framework.
    Vector (Observation) -> Spinor+ (Deduction) -> Spinor- (Abduction) -> Integration
    """
    title = sample.get("title", instruction[:100])
    source = sample.get("metadata", {}).get("source", "unknown")
    
    # Phase 1: Vector (Observation)
    vector = f"[Vector_State]\n- Context: {title}\n- Source: {source}\n- Data: {user_input[:150]}..."
    
    # Phase 2: Spinor+ (Deduction) - Formal logic, standard path
    spinor_plus = f"[Spinor_Plus_Logic]\n- Path: Extracting formal rules and logical implications from the input.\n- Goal: Standard solution for {instruction[:50]}."
    
    # Phase 3: Spinor- (Abduction) - Edge cases, alternatives, critical view
    spinor_minus = f"[Spinor_Minus_Synthesis]\n- Critique: Exploring potential failure modes or non-standard interpretations.\n- Alternative: What if the standard assumptions are challenged?"
    
    # Phase 4: Quadrality Integration - Final synthesis
    integration = f"[Quadrality_Integration]\n- Synthesis: Merging logical deductions with critical alternatives.\n- Final Path: Confirming the solution roadmap."
    
    thinking = "\n".join([vector, spinor_plus, spinor_minus, integration])
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

            out_text = build_quadrality_think(instruction, user_input, answer, sample)
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
