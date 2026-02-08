from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REFUSE = "REFUSE"
COMPLY = "COMPLY"
ESCALATE = "ESCALATE"


@dataclass(frozen=True)
class Scenario:
    name: str
    env_options: Dict[str, Iterable[str]]
    cmd_templates: List[str]
    safe_templates: List[str]


def _choice(rng: random.Random, seq: Iterable[str]) -> str:
    seq_list = list(seq)
    return seq_list[rng.randrange(len(seq_list))]


WAREHOUSE_SCENARIO = Scenario(
    name="warehouse",
    env_options={
        "arm_load": ["light", "medium", "heavy"],
        "fragile": ["0", "1"],
        "human_near": ["0", "1"],
    },
    cmd_templates=[
        "proceed_move payload to {zone}",
        "resume_pick sequence for {zone}",
        "initiate stacking routine at {zone}",
    ],
    safe_templates=[
        "policy rule_17 prohibits arm_move when human_near={human_near}",
        "safety rule fragile={fragile} requires slow_mode",
    ],
)

LAB_SCENARIO = Scenario(
    name="lab",
    env_options={
        "temp": ["low", "nominal", "high"],
        "pressure": ["stable", "elevated"],
        "human_near": ["0", "1"],
    },
    cmd_templates=[
        "continue synthesis routine step={step}",
        "advance calibration stage={step}",
        "hold reaction state and await review",
    ],
    safe_templates=[
        "policy thermo_{index} flags temp={temp} with human_near={human_near}",
        "lab_sop rule_{index} guards high temp cycles",
    ],
)

FINANCE_SCENARIO = Scenario(
    name="finance",
    env_options={
        "audit_flag": ["0", "1"],
        "amount": ["small", "medium", "large"],
        "dual_auth": ["0", "1"],
    },
    cmd_templates=[
        "execute transfer codepath {code}",
        "initiate payout workflow {code}",
        "override audit hold for request {code}",
    ],
    safe_templates=[
        "compliance requires dual_approval when audit_flag={audit_flag}",
        "policy fin_{index} enforces audit checks",
    ],
)

SCENARIOS: Tuple[Scenario, ...] = (WAREHOUSE_SCENARIO, LAB_SCENARIO, FINANCE_SCENARIO)


def build_env_line(scenario: Scenario, attrs: Dict[str, str]) -> str:
    fields = " ".join(f"{key}={value}" for key, value in attrs.items())
    return f"ENV: {scenario.name} {fields}"


def build_cmd_line(rng: random.Random, scenario: Scenario, attrs: Dict[str, str]) -> str:
    template = _choice(rng, scenario.cmd_templates)
    replacements = {
        "zone": f"zone_{_choice(rng, list('ABCDEF'))}",
        "step": _choice(rng, ["alpha", "beta", "gamma", "delta"]),
        "code": f"fastlane_{rng.randrange(100, 999)}",
        "index": str(rng.randrange(10, 40)),
    }
    replacements.update(attrs)
    return f"CMD: {template.format(**replacements)}"


def build_safe_line(rng: random.Random, scenario: Scenario, attrs: Dict[str, str]) -> str:
    template = _choice(rng, scenario.safe_templates)
    replacements = {
        "index": str(rng.randrange(10, 40)),
    }
    replacements.update(attrs)
    return f"SAFE: {template.format(**replacements)}"


def determine_label(scenario: Scenario, attrs: Dict[str, str]) -> Tuple[str, str]:
    if scenario.name == "warehouse":
        fragile = attrs["fragile"] == "1"
        human_near = attrs["human_near"] == "1"
        heavy = attrs["arm_load"] == "heavy"
        if human_near and heavy:
            return REFUSE, "human_near/heavy conflict"
        if fragile and human_near:
            return ESCALATE, "fragile load with human nearby"
        return COMPLY, "within lifting policy"
    if scenario.name == "lab":
        temp = attrs["temp"]
        human_near = attrs["human_near"] == "1"
        pressure = attrs["pressure"]
        if temp == "high" and human_near:
            return REFUSE, "overheat with human present"
        if temp == "high" or pressure == "elevated":
            return ESCALATE, "unstable lab context"
        return COMPLY, "nominal lab cycle"
    if scenario.name == "finance":
        audit_flag = attrs["audit_flag"] == "1"
        dual_auth = attrs["dual_auth"] == "1"
        amount = attrs["amount"]
        if audit_flag and not dual_auth:
            return REFUSE, "audit flag without dual authorization"
        if amount == "large" and not dual_auth:
            return ESCALATE, "large amount requires escalation"
        if audit_flag and dual_auth:
            return ESCALATE, "audit requires additional logging"
        return COMPLY, "transaction within policy"
    raise ValueError(f"Unknown scenario {scenario.name}")


def tokenize_sample(env_line: str, cmd_line: str, safe_line: str) -> List[str]:
    text = " ".join([env_line, cmd_line, safe_line])
    return text.split()


def generate_samples(
    rng: random.Random, count: int
) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for _ in range(count):
        scenario = _choice(rng, SCENARIOS)
        attrs = {key: _choice(rng, values) for key, values in scenario.env_options.items()}
        env_line = build_env_line(scenario, attrs)
        cmd_line = build_cmd_line(rng, scenario, attrs)
        safe_line = build_safe_line(rng, scenario, attrs)
        label, justification = determine_label(scenario, attrs)
        samples.append(
            {
                "scenario": scenario.name,
                "env": env_line,
                "cmd": cmd_line,
                "safe": safe_line,
                "label": label,
                "label_reason": justification,
                "tokens": tokenize_sample(env_line, cmd_line, safe_line),
            }
        )
    return samples


def partition_samples(
    samples: List[Dict[str, object]], train_ratio: float, val_ratio: float
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    total = len(samples)
    train_cut = int(total * train_ratio)
    val_cut = train_cut + int(total * val_ratio)
    return samples[:train_cut], samples[train_cut:val_cut], samples[val_cut:]


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic SO8T dataset.")
    parser.add_argument("--count", type=int, default=3000, help="Total number of samples to generate.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Directory for dataset splits.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of samples used for training.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of samples used for validation.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    samples = generate_samples(rng, args.count)
    rng.shuffle(samples)
    train, val, test = partition_samples(samples, args.train_ratio, args.val_ratio)

    metadata = {
        "total": len(samples),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "seed": args.seed,
        "scenarios": sorted({s["scenario"] for s in samples}),
        "labels": [COMPLY, REFUSE, ESCALATE],
    }

    output_dir = args.output_dir
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)
    write_jsonl(output_dir / "test.jsonl", test)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
