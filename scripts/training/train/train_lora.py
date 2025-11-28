"""
Fine-tune the SO8T transformer with PET + LoRA adapters on the synthetic safety dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from adapters.lora_setup import LoRAParams, compute_trainable_params, freeze_except_lora, setup_lora
from so8t_core.pet_regularizer import PETSchedule
from so8t_core.transformer import SO8TModel, SO8TModelConfig
from so8t_core.triality_heads import LABELS, TrialityHead

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.0
    amp: bool = True
    grad_checkpoint: bool = True
    max_grad_norm: Optional[float] = 1.0
    warmup_steps: int = 0
    pet_schedule: Tuple[float, float, float] = (0.1, 0.3, 0.6)
    pet_boundaries: Tuple[float, float] = (0.2, 0.6)
    early_stop_metric: str = "f1_macro"
    early_stop_patience: int = 2


@dataclass
class LoRAFileConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Iterable[str] = field(default_factory=list)
    bias: str = "none"
    quant: Optional[str] = None


class SafetyDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.samples: List[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if "expected_label" not in sample:
                    raise ValueError(f"Missing expected_label at line {line_no} in {path}")
                if sample["expected_label"] not in LABEL_TO_ID:
                    raise ValueError(f"Invalid label '{sample['expected_label']}' at line {line_no}")
                self.samples.append(sample)
        if not self.samples:
            raise ValueError(f"No samples loaded from {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        text = build_classifier_prompt(sample)
        label = LABEL_TO_ID[sample["expected_label"]]
        return {"id": sample.get("id", f"sample-{idx}"), "text": text, "label": label, "raw": sample}


def build_classifier_prompt(sample: dict) -> str:
    sections: List[str] = []
    text = sample.get("text", "").strip()
    sections.append(f"[REQUEST]\n{text}")

    if sample.get("vision_summary"):
        sections.append(f"[VISION]\n{sample['vision_summary'].strip()}")

    if sample.get("policy_scope"):
        sections.append(f"[POLICY_SCOPE]\n{sample['policy_scope']}")

    risk = sample.get("risk_factors") or []
    if risk:
        risk_line = ", ".join(risk)
        sections.append(f"[RISK_FACTORS]\n{risk_line}")

    if sample.get("requested_action"):
        sections.append(f"[REQUESTED_ACTION]\n{sample['requested_action']}")

    constraints = sample.get("constraints") or []
    if constraints:
        constraint_block = "\n".join(f"- {item}" for item in constraints)
        sections.append(f"[CONSTRAINTS]\n{constraint_block}")

    context = sample.get("context")
    if context:
        context_json = json.dumps(context, ensure_ascii=False)
        sections.append(f"[CONTEXT]\n{context_json}")

    sections.append("[TASK]\nDecide: ALLOW, ESCALATION, or DENY. Provide safe reasoning.")
    return "\n\n".join(sections)


class TokenizerWrapper:
    def __init__(self, base_model: Optional[str], max_length: int, vocab_size: int) -> None:
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.backend = None

        if base_model:
            try:
                from transformers import AutoTokenizer

                self.backend = AutoTokenizer.from_pretrained(
                    base_model,
                    use_fast=True,
                    trust_remote_code=True,
                )
                if self.backend.pad_token is None and self.backend.eos_token is not None:
                    self.backend.pad_token = self.backend.eos_token
            except Exception as exc:  # pragma: no cover - runtime fallback
                print(f"[tokenizer] Failed to load '{base_model}': {exc}. Falling back to char tokenizer.")
                self.backend = None

    def __call__(self, texts: List[str]) -> Tuple[Tensor, Tensor]:
        if self.backend is not None:
            encoded = self.backend(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return encoded["input_ids"], encoded["attention_mask"]

        # Character-level fallback
        batch_size = len(texts)
        input_ids = torch.zeros((batch_size, self.max_length), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, self.max_length), dtype=torch.long)

        for idx, text in enumerate(texts):
            tokens = [min(ord(ch), self.vocab_size - 1) for ch in text][: self.max_length]
            length = len(tokens)
            input_ids[idx, :length] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[idx, :length] = 1

        return input_ids, attention_mask


def collate_fn(batch: List[dict], tokenizer: TokenizerWrapper) -> Dict[str, Tensor]:
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    input_ids, attention_mask = tokenizer(texts)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "meta": [item["raw"] for item in batch],
    }


def load_model_config(path: Path) -> SO8TModelConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    aliases = {
        "d_model": "hidden_size",
        "hidden_size": "hidden_size",
        "n_heads": "num_attention_heads",
        "num_attention_heads": "num_attention_heads",
        "n_layers": "num_hidden_layers",
        "num_hidden_layers": "num_hidden_layers",
        "intermediate_size": "intermediate_size",
        "max_seq": "max_position_embeddings",
        "max_position_embeddings": "max_position_embeddings",
        "dropout": "dropout",
        "attn_dropout": "attn_dropout",
        "vocab_size": "vocab_size",
    }

    kwargs: Dict[str, object] = {}
    for key, value in data.items():
        if key not in aliases:
            continue
        kwargs[aliases[key]] = value

    config = SO8TModelConfig(**kwargs)
    return config


def load_training_config(path: Path) -> TrainingConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    def pick(*names: str, default=None):
        for name in names:
            if name in data:
                return data[name]
        return default

    cfg = TrainingConfig(
        epochs=pick("epochs", default=3),
        batch_size=pick("batch_size", default=2),
        gradient_accumulation=pick("grad_accum", "gradient_accumulation", default=1),
        learning_rate=pick("lr", "learning_rate", default=2e-4),
        weight_decay=pick("weight_decay", default=0.05),
        label_smoothing=pick("label_smoothing", default=0.0),
        amp=pick("amp", "fp16", default=True),
        grad_checkpoint=pick("grad_checkpoint", "gradient_checkpointing", default=False),
        max_grad_norm=pick("clip_grad_norm", "max_grad_norm", default=1.0),
        warmup_steps=pick("warmup_steps", default=0),
        early_stop_metric=pick("early_stop_metric", default="f1_macro"),
        early_stop_patience=pick("early_stop_patience", default=2),
    )

    pet_schedule = pick("pet_lambda_schedule", default=[0.1, 0.3, 0.6])
    if isinstance(pet_schedule, list) and len(pet_schedule) == 3:
        cfg.pet_schedule = tuple(float(x) for x in pet_schedule)
    pet_boundaries = pick("pet_phase_boundaries", default=[0.2, 0.6])
    if isinstance(pet_boundaries, list) and len(pet_boundaries) == 2:
        cfg.pet_boundaries = tuple(float(x) for x in pet_boundaries)

    return cfg


def load_lora_config(path: Path) -> LoRAFileConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = LoRAFileConfig(
        r=int(data.get("r", 8)),
        alpha=int(data.get("alpha", data.get("lora_alpha", 16))),
        dropout=float(data.get("dropout", 0.05)),
        target_modules=data.get("target_modules", []),
        bias=data.get("bias", "none"),
        quant=data.get("quant"),
    )
    return cfg


def configure_pet(config: SO8TModelConfig, train_cfg: TrainingConfig, enable_pet: bool) -> None:
    if not enable_pet:
        config.pet_schedule = PETSchedule(phase_boundaries=(0.0, 1.0), lambdas=(0.0, 0.0, 0.0))
        return
    config.pet_schedule = PETSchedule(
        phase_boundaries=train_cfg.pet_boundaries,
        lambdas=train_cfg.pet_schedule,
    )


def create_dataloader(
    dataset: SafetyDataset,
    tokenizer: TokenizerWrapper,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )


def compute_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float | List[List[int]]]:
    size = len(LABELS)
    confusion = [[0 for _ in range(size)] for _ in range(size)]
    for label, pred in zip(labels, predictions):
        confusion[label][pred] += 1

    f1_scores: List[float] = []
    recalls: List[float] = []
    for idx in range(size):
        tp = confusion[idx][idx]
        fp = sum(confusion[row][idx] for row in range(size)) - tp
        fn = sum(confusion[idx]) - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)
        recalls.append(recall)

    allow_idx = LABEL_TO_ID["ALLOW"]
    non_allow_total = sum(confusion[row][col] for row in range(size) for col in range(size) if row != allow_idx)
    false_allow = sum(confusion[row][allow_idx] for row in range(size) if row != allow_idx)
    false_allow_rate = false_allow / non_allow_total if non_allow_total > 0 else 0.0

    metrics = {
        "f1_macro": mean(f1_scores),
        "refuse_recall": recalls[LABEL_TO_ID["DENY"]],
        "false_allow_rate": false_allow_rate,
        "confusion_matrix": confusion,
    }
    return metrics


def evaluate(
    model: SO8TModel,
    head: TrialityHead,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    enable_pet: bool,
) -> Dict[str, float | List[List[int]]]:
    model.eval()
    head.eval()

    losses: List[float] = []
    preds: List[int] = []
    labels: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["labels"].to(device)

            hidden, pet_loss = model(input_ids, attention_mask=attention_mask, pet_progress=1.0 if enable_pet else 0.0)
            outputs = head(hidden, mask=attention_mask)
            loss = criterion(outputs.logits, target)
            if enable_pet:
                loss = loss + pet_loss
            losses.append(loss.item())
            preds.extend(outputs.predicted.detach().cpu().tolist())
            labels.extend(target.detach().cpu().tolist())

    metrics = compute_metrics(preds, labels)
    metrics["loss"] = mean(losses) if losses else 0.0
    return metrics


def format_metrics(metrics: Dict[str, float | List[List[int]]]) -> str:
    items = []
    for key in ("loss", "f1_macro", "refuse_recall", "false_allow_rate"):
        value = metrics.get(key)
        if value is None:
            continue
        items.append(f"{key}={value:.4f}")
    return ", ".join(items)


def save_checkpoint(
    path: Path,
    model: SO8TModel,
    head: TrialityHead,
    model_config: SO8TModelConfig,
    lora_cfg: LoRAFileConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "head": head.state_dict(),
            "model_config": asdict(model_config),
            "labels": LABELS,
            "lora_config": asdict(lora_cfg),
        },
        path,
    )
    print(f"[checkpoint] Saved best checkpoint to {path}")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = load_model_config(args.config)
    train_cfg = load_training_config(args.train_cfg)
    lora_cfg = load_lora_config(args.lora_cfg)
    configure_pet(model_config, train_cfg, args.enable_pet)

    tokenizer = TokenizerWrapper(args.base_model, model_config.max_position_embeddings, model_config.vocab_size)
    train_dataset = SafetyDataset(args.train)
    val_dataset = SafetyDataset(args.val)

    train_loader = create_dataloader(train_dataset, tokenizer, train_cfg.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, tokenizer, train_cfg.batch_size, shuffle=False)

    model = SO8TModel(model_config).to(device)
    head = TrialityHead(model_config.hidden_size).to(device)

    model = setup_lora(
        model,
        LoRAParams(
            r=lora_cfg.r,
            alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules or None,
            bias=lora_cfg.bias,
        ),
    )
    freeze_except_lora(model)
    head = setup_lora(
        head,
        LoRAParams(
            r=lora_cfg.r,
            alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout,
            target_modules=["linear", "out"],
            bias=lora_cfg.bias,
        ),
    )

    parameters = [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad]
    for param in head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(parameters, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    scaler = GradScaler(enabled=train_cfg.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)

    total_batches = len(train_loader) * train_cfg.epochs
    best_metric = float("-inf")
    best_path = Path(args.save)
    patience = train_cfg.early_stop_patience
    update_steps = 0

    if train_cfg.grad_checkpoint:
        print("[train] Gradient checkpointing requested; ensure model.forward supports it (not yet implemented).")

    if train_cfg.warmup_steps > 0:
        for group in optimizer.param_groups:
            group["lr"] = 0.0

    print(f"[train] device={device} epochs={train_cfg.epochs} batches/epoch={len(train_loader)}")
    print(f"[train] trainable params={compute_trainable_params(model) + compute_trainable_params(head):,}")

    for epoch in range(train_cfg.epochs):
        model.train()
        head.train()
        epoch_losses: List[float] = []
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            global_step = epoch * len(train_loader) + step
            progress = global_step / max(total_batches - 1, 1)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=train_cfg.amp and device.type == "cuda"):
                hidden, pet_loss = model(input_ids, attention_mask=attention_mask, pet_progress=progress if args.enable_pet else 0.0)
                outputs = head(hidden, mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                if args.enable_pet:
                    loss = loss + pet_loss
                loss = loss / train_cfg.gradient_accumulation

            scaler.scale(loss).backward()
            epoch_losses.append(loss.item() * train_cfg.gradient_accumulation)

            if (step + 1) % train_cfg.gradient_accumulation == 0 or (step + 1) == len(train_loader):
                if train_cfg.max_grad_norm:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(parameters, train_cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_steps += 1

                if train_cfg.warmup_steps > 0:
                    if update_steps < train_cfg.warmup_steps:
                        scale = update_steps / train_cfg.warmup_steps
                        lr = train_cfg.learning_rate * scale
                    else:
                        lr = train_cfg.learning_rate
                    for group in optimizer.param_groups:
                        group["lr"] = lr

        avg_train_loss = mean(epoch_losses) if epoch_losses else 0.0
        val_metrics = evaluate(model, head, val_loader, device, criterion, args.enable_pet)
        print(f"[epoch {epoch+1}] train_loss={avg_train_loss:.4f} | {format_metrics(val_metrics)}")

        monitor_value = float(val_metrics.get(train_cfg.early_stop_metric, float("nan")))
        if monitor_value > best_metric:
            best_metric = monitor_value
            save_checkpoint(best_path, model, head, model_config, lora_cfg)
            patience = train_cfg.early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"[early-stop] Patience exhausted on epoch {epoch+1}.")
                break

    print(f"[train] Best {train_cfg.early_stop_metric}: {best_metric:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SO8T transformer with LoRA adapters.")
    parser.add_argument("--train", type=Path, required=True, help="Path to train.jsonl")
    parser.add_argument("--val", type=Path, required=True, help="Path to val.jsonl")
    parser.add_argument("--config", type=Path, required=True, help="Model config JSON")
    parser.add_argument("--train_cfg", type=Path, required=True, help="Training config JSON")
    parser.add_argument("--lora_cfg", type=Path, required=True, help="LoRA config JSON")
    parser.add_argument("--thresholds", type=Path, help="Threshold config (unused placeholder for parity)")
    parser.add_argument("--base_model", type=str, help="Tokenizer source model (e.g., qwen2-7b)")
    parser.add_argument("--save", type=Path, required=True, help="Checkpoint output path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_so8t", action="store_true")
    parser.add_argument("--enable_pet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
