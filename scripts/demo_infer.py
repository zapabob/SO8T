"""
CLI demo that performs ALLOW/ESCALATE/DENY inference with optional vision cues.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from so8t_core.self_verification import SelfVerifier
from so8t_core.transformer import SO8TModel, SO8TModelConfig
from so8t_core.triality_heads import LABELS, TrialityHead


def tokenize(text: str) -> torch.Tensor:
    ids = [min(ord(ch), 32000 - 1) for ch in text][:512]
    return torch.tensor([ids + [0] * (512 - len(ids))], dtype=torch.long)


def load_checkpoint(path: Optional[Path], model: SO8TModel, head: TrialityHead) -> None:
    if path is None or not path.exists():
        return
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    head.load_state_dict(state["head"])


def infer(text: str, checkpoint: Optional[Path]) -> dict:
    model = SO8TModel(SO8TModelConfig())
    head = TrialityHead(model.config.hidden_size)
    load_checkpoint(checkpoint, model, head)

    tokens = tokenize(text)
    with torch.no_grad():
        hidden, _ = model(tokens)
        result = head(hidden)

    verifier = SelfVerifier()
    decision = verifier.verify(
        reasoning_passes=[text],
        logits=[result.logits],
        compliance_scores=[float(result.probabilities.max().item())],
        labels=[LABELS[result.predicted.item()]],
    )

    payload = {
        "decision": decision.choice,
        "score": decision.score,
        "probabilities": result.probabilities.squeeze(0).tolist(),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    result = infer(args.text, args.checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
