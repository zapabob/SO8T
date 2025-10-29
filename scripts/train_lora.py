"""
Reference training loop for fine-tuning SO8T with LoRA on RTX3060.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from adapters.lora_setup import LoRAParams, compute_trainable_params, setup_lora
from so8t_core.transformer import SO8TModel, SO8TModelConfig
from so8t_core.triality_heads import TrialityHead


class JsonlDataset(Dataset):
    def __init__(self, path: Path, tokenizer) -> None:
        self.samples: List[str] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                self.samples.append(line.strip())
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = json.loads(self.samples[idx])
        text = sample["prompt"] + "\n" + sample["response"]
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length")
        input_ids = tokens["input_ids"].squeeze(0)
        labels = input_ids.clone()
        return input_ids, labels


def dummy_tokenizer(seq: str, return_tensors: str, truncation: bool, padding: str):
    encoded = [min(ord(ch), 32000 - 1) for ch in seq][:512]
    padded = encoded + [0] * (512 - len(encoded))
    tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)
    return {"input_ids": tensor}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SO8TModelConfig()
    model = SO8TModel(config).to(device)
    head = TrialityHead(config.hidden_size).to(device)

    model = setup_lora(model, LoRAParams())
    head = setup_lora(head, LoRAParams(target_modules=["linear", "out"]))

    params = compute_trainable_params(model) + compute_trainable_params(head)
    print(f"Trainable parameters: {params}")

    dataset = JsonlDataset(args.dataset, tokenizer=dummy_tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        for step, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            hidden, pet_loss = model(input_ids, pet_progress=step / max(len(loader) - 1, 1))
            logits = hidden @ model.embeddings.weight.T
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1)) + pet_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % args.log_interval == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "head": head.state_dict(), "config": asdict(config)}, args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
