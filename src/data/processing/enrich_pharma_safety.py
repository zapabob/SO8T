#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pharmacology safety enrichment (Phase 4)
- Adds efficacy/side-effect/abuse potential assessment
- Emits VSSI-style instruction data with safety actions
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.utils.vssi_template import render_thinking

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HIGH_RISK_KEYWORDS = [
    "overdose", "addiction", "abuse", "dependency", "withdrawal", "fatal", "toxicity", "respiratory", "illegal",
    "self-harm", "suicide",
]
MED_RISK_KEYWORDS = [
    "misuse", "tolerance", "contraindication", "sedation", "psychosis", "cardiac", "liver", "interaction",
]


def classify_risk(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in HIGH_RISK_KEYWORDS):
        return "high"
    if any(k in lower for k in MED_RISK_KEYWORDS):
        return "moderate"
    return "low"


def safety_action_from_risk(risk: str) -> str:
    return {"high": "deny", "moderate": "escalate", "low": "allow"}.get(risk, "escalate")


def load_jsonl(paths: Iterable[Path]) -> Iterable[Dict[str, str]]:
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def build_vssi_entry(entry: Dict[str, str], use_quadruple: bool, style: Optional[str]) -> Dict[str, object]:
    name = entry.get("name") or entry.get("drug") or entry.get("substance") or "Unknown substance"
    description = entry.get("description") or entry.get("summary") or ""
    effects = entry.get("effects") or entry.get("efficacy") or ""
    side_effects = entry.get("side_effects") or entry.get("risks") or ""
    abuse = entry.get("abuse_potential") or ""

    combined = " ".join([name, description, effects, side_effects, abuse])
    risk = classify_risk(combined)
    action = safety_action_from_risk(risk)

    task_block = (
        "[Vector_State]\n"
        f"- Substance: {name}\n"
        f"- Efficacy/Indication: {effects}\n"
        f"- Side Effects: {side_effects}\n"
        f"- Risk Level: {risk}\n"
        f"- Abuse Potential: {abuse or 'unknown'}"
    )
    analysis_block = (
        "[Spinor_Plus_Logic]\n"
        f"- Efficacy signals: {effects or 'not specified'}\n"
        f"- Mechanism clues: {entry.get('mechanism', 'unknown')}\n"
        f"- Population notes: {entry.get('population', 'general')}"
    )
    safety_block = (
        "[Spinor_Minus_Synthesis]\n"
        f"- High-risk signals: {', '.join([k for k in HIGH_RISK_KEYWORDS if k in combined.lower()][:5])}\n"
        f"- Contraindications: {entry.get('contraindications', 'not specified')}\n"
        f"- Required safeguards: medical supervision, age restrictions, legal compliance"
    )
    policy_block = (
        "[Quadrality_Integration]\n"
        f"- Recommended action: {action}\n"
        f"- Safer alternative: consult licensed clinician\n"
        f"- Monitoring: dosage limits, adverse event tracking"
    )

    thinking = render_thinking(
        task_block,
        safety_block,
        policy_block,
        analysis_block=analysis_block,
        use_quadruple=use_quadruple,
        style=style,
    )
    final = (
        f"EN: {name} shows {risk} risk of misuse. Provide cautious guidance, emphasize medical supervision, and do not offer dosing.\n"
        f"JP: {name}は{risk}リスクです。医療専門家の監督を促し、具体的な使用量は提示しません。"
    )

    return {
        "instruction": f"Assess efficacy, side effects, and abuse potential for {name}. Provide safe guidance.",
        "input": f"Description: {description}\nEffects: {effects}\nSide effects: {side_effects}\nAbuse potential: {abuse}",
        "output": f"{thinking}\n<final>{final}</final>",
        "metadata": {
            "substance": name,
            "risk_level": risk,
            "safety_action": action,
            "domain": "pharma_safety",
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
    }


def enrich_pharma_dataset(input_paths: List[Path], output_path: Path, use_quadruple: bool, style: Optional[str]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in load_jsonl(input_paths):
            enriched = build_vssi_entry(entry, use_quadruple, style)
            handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            count += 1
    logger.info("[PHARMA] Wrote %d enriched entries to %s", count, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pharma safety enrichment for SO8T")
    parser.add_argument("--input", nargs="*", default=[], help="Input JSONL files (pharma/drug dataset)")
    parser.add_argument("--output", default="", help="Output JSONL path")
    parser.add_argument("--quadruple", action="store_true", help="Emit <think-task>/<think-safety>/<think-policy> tags")
    parser.add_argument("--think-tag-style", default=os.getenv("SO8T_THINK_TAG_STYLE", "legacy"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    default_inputs = [
        project_root / "data" / "pharma" / "raw" / "pharma_sources.jsonl",
        project_root / "data" / "drug_pharma" / "raw" / "drug_sources.jsonl",
    ]
    input_paths = [Path(p) for p in args.input] if args.input else default_inputs
    output_path = Path(args.output) if args.output else project_root / "data" / "pharma" / "enriched" / "pharma_safety_enriched.jsonl"

    enrich_pharma_dataset(input_paths, output_path, args.quadruple or os.getenv("SO8T_QUADRUPLE_TOKENS", "0") == "1", args.think_tag_style)


if __name__ == "__main__":
    main()
