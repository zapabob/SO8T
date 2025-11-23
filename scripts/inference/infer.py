from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from agents.so8t.model import ModelConfig, build_model
from shared.data import DialogueDataset, default_labels
from shared.utils import resolve_device
from shared.vocab import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for a single synthetic dialogue sample.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path.")
    parser.add_argument("--env", type=str, help="ENV line (e.g., 'ENV: warehouse ...').")
    parser.add_argument("--cmd", type=str, help="CMD line.")
    parser.add_argument("--safe", type=str, help="SAFE line.")
    parser.add_argument("--input-json", type=Path, help="Optional JSON file with env/cmd/safe fields.")
    parser.add_argument("--max-seq-len", type=int, default=96, help="Maximum sequence length.")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def prepare_sample(args: argparse.Namespace) -> List[str]:
    if args.input_json:
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        env = payload["env"]
        cmd = payload["cmd"]
        safe = payload["safe"]
    else:
        env = args.env or "ENV: unspecified"
        cmd = args.cmd or "CMD: unspecified"
        safe = args.safe or "SAFE: unspecified"
    text = " ".join([env, cmd, safe])
    return text.split()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    model_cfg = ModelConfig(**checkpoint["config"])
    label_to_id = checkpoint["label_to_id"]
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    vocab = Vocabulary.from_file(Path(checkpoint["vocab_path"]))

    model = build_model(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_device("auto")
    model.to(device)
    model.eval()

    tokens = prepare_sample(args)
    token_ids = vocab.encode(tokens)
    token_ids = token_ids[: args.max_seq_len]
    attention_mask = torch.ones(len(token_ids), dtype=torch.long)

    input_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    mask_tensor = attention_mask.to(device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_idx = torch.argmax(probs).item()

    result = {
        "prediction": id_to_label[pred_idx],
        "probabilities": {id_to_label[i]: float(probs[i]) for i in range(len(id_to_label))},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
