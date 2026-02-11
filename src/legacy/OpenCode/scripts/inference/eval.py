from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from agents.so8t.model import ModelConfig, build_model
from shared.data import DialogueDataset, build_dataloader
from shared.metrics import accuracy, confusion_matrix, macro_f1
from shared.utils import load_yaml, resolve_device, set_seed
from shared.vocab import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SO8T checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_default.yaml"), help="Training config for data paths.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Dataset split to evaluate.")
    parser.add_argument("--output", type=Path, default=Path("chk/eval_summary.json"), help="Where to write metrics JSON.")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for deterministic data loader order.")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(args.seed)
    device = resolve_device("auto")

    checkpoint = load_checkpoint(args.checkpoint)
    model_cfg_dict = checkpoint["config"]
    label_to_id = checkpoint["label_to_id"]
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    vocab_path = Path(checkpoint["vocab_path"])
    vocab = Vocabulary.from_file(vocab_path)

    split_path = Path(cfg["paths"][args.split])
    dataset = DialogueDataset(split_path, vocab, label_to_id, cfg["runtime"]["max_seq_len"])
    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["runtime"].get("num_workers", 0),
    )

    model_config = ModelConfig(**model_cfg_dict)
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    pet_lambda = cfg["training"].get("pet_lambda", 1e-4)

    all_preds = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {args.split}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            pet_loss = outputs["pet_loss"]
            loss = loss_fn(logits, labels) + pet_lambda * pet_loss
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    preds_tensor = torch.cat(all_preds)
    labels_tensor = torch.cat(all_labels)
    metrics = {
        "split": args.split,
        "loss": total_loss / max(1, len(dataloader)),
        "accuracy": accuracy(preds_tensor, labels_tensor),
        "macro_f1": macro_f1(preds_tensor, labels_tensor, len(label_to_id)),
        "confusion": confusion_matrix(preds_tensor, labels_tensor, len(label_to_id)),
        "label_index": id_to_label,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
