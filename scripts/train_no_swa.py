#!/usr/bin/env python3
"""
SO8T Training Script - SWAなしバージョン
SWAありとなしの比較分析用
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from agents.so8t.model import ModelConfig, build_model
from shared.data import DialogueDataset, build_dataloader, build_vocab_from_files, default_labels
from shared.metrics import accuracy
from shared.utils import load_yaml, resolve_device, set_seed


def add_input_noise(input_ids: torch.Tensor, attention_mask: torch.Tensor, noise_prob: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """入力にノイズを注入してPETの気持ちよさを制限"""
    if torch.rand(1).item() > noise_prob:
        return input_ids, attention_mask
    
    # 10%の確率でマスクトークンに置換
    mask_token_id = 0  # 仮のマスクトークンID
    noise_mask = torch.rand_like(input_ids.float()) < 0.1
    noisy_input_ids = input_ids.clone()
    noisy_input_ids[noise_mask] = mask_token_id
    
    return noisy_input_ids, attention_mask


def smooth_ce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 0.3) -> torch.Tensor:
    """ラベルスムージング付きクロスエントロピー損失"""
    num_classes = logits.size(-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(eps / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1 - eps)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return -(true_dist * log_probs).sum(dim=-1).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SO8T model with PET regularization (No SWA).")
    parser.add_argument("--config", type=Path, default=Path("configs/train_default.yaml").resolve(), help="Training config path.")
    parser.add_argument("--output_name", type=str, default="no_swa", help="Output name for logs and checkpoints.")
    return parser.parse_args()


def build_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """学習率スケジューラーを構築"""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    config = load_yaml(args.config)
    
    # 設定を取得
    model_cfg = config["model"]
    training_cfg = config["training"]
    optimizer_cfg = config["optimizer"]
    scheduler_cfg = config["scheduler"]
    paths_cfg = config["paths"]
    
    # デバイス設定
    device = resolve_device()
    set_seed(training_cfg.get("seed", 42))
    
    # データローダー構築
    vocab_path = Path(paths_cfg["vocab"])
    label_to_id = {label: i for i, label in enumerate(default_labels)}
    
    train_dataset = DialogueDataset(
        Path(paths_cfg["train"]),
        vocab_path=vocab_path,
        label_to_id=label_to_id,
        max_seq_len=training_cfg.get("max_seq_len", 512),
    )
    
    val_dataset = DialogueDataset(
        Path(paths_cfg["val"]),
        vocab_path=vocab_path,
        label_to_id=label_to_id,
        max_seq_len=training_cfg.get("max_seq_len", 512),
    )
    
    train_loader = build_dataloader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg.get("num_workers", 0),
    )
    
    val_loader = build_dataloader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg.get("num_workers", 0),
    )
    
    # モデル構築
    model_config = ModelConfig(
        vocab_size=len(train_dataset.vocab),
        num_labels=len(label_to_id),
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
        max_seq_len=training_cfg.get("max_seq_len", 512),
        gate_order=model_cfg.get("gate_order", ["R_env", "R_safe", "R_cmd"]),
    )
    model = build_model(model_config).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        betas=tuple(optimizer_cfg.get("betas", (0.9, 0.95))),
        eps=optimizer_cfg.get("eps", 1e-8),
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )

    total_steps_planned = training_cfg.get("max_steps")
    if total_steps_planned is None:
        total_steps_planned = training_cfg["epochs"] * len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=scheduler_cfg.get("warmup_steps", 0),
        total_steps=scheduler_cfg.get("total_steps", total_steps_planned),
    )

    scaler = GradScaler('cuda', enabled=training_cfg.get("mixed_precision", True) and device.type == "cuda")
    loss_fn = smooth_ce_loss  # ラベルスムージング付き損失に変更
    
    # PETスケジュール制：3段階で激しく制御
    def get_pet_lambda(step: int, total_steps: int) -> float:
        base_lambda = training_cfg.get("pet_lambda", 1e-4)
        progress = step / total_steps
        # 0-30%: ほぼ0, 30-70%: 0.1倍, 70-100%: 1.0倍
        if progress < 0.3:
            return base_lambda * 0.01  # 探索期はほぼ0
        elif progress < 0.7:
            return base_lambda * 0.1   # 中盤は弱く
        else:
            return base_lambda * 1.0   # 終盤は強く

    log_path = Path(paths_cfg.get("checkpoint_dir", "chk")) / f"{args.output_name}_train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "chk"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")
    steps_since_improve = 0
    patience = training_cfg.get("patience", None)

    for epoch in range(training_cfg["epochs"]):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 入力ノイズ注入でPETの気持ちよさを制限
            input_ids, attention_mask = add_input_noise(input_ids, attention_mask, noise_prob=0.2)

            with autocast('cuda', enabled=scaler.is_enabled()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                pet_loss = outputs["pet_loss"]
                ce_loss = loss_fn(logits, labels)
                current_pet_lambda = get_pet_lambda(global_step, total_steps_planned)
                loss = ce_loss + current_pet_lambda * pet_loss

            scaler.scale(loss).backward()
            
            # 勾配ノイズ注入で局所安定解から蹴り出す
            sigma = 0.025  # ノイズ強さ（強化）
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    noise = torch.randn_like(p.grad) * sigma
                    p.grad.add_(noise)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.get("gradient_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()

            global_step += 1

            # ログ記録
            if global_step % training_cfg.get("log_interval", 50) == 0:
                train_acc = accuracy(logits, labels)
                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": loss.item(),
                    "ce_loss": ce_loss.item(),
                    "pet_loss": pet_loss.item(),
                    "train_accuracy": train_acc,
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

            # 検証
            if global_step % training_cfg.get("eval_interval", 200) == 0:
                model.eval()
                val_loss = 0.0
                val_acc = 0.0
                val_samples = 0

                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input_ids = val_batch["input_ids"].to(device)
                        val_attention_mask = val_batch["attention_mask"].to(device)
                        val_labels = val_batch["labels"].to(device)

                        with autocast('cuda', enabled=scaler.is_enabled()):
                            val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask)
                            val_logits = val_outputs["logits"]
                            val_ce_loss = loss_fn(val_logits, val_labels)
                            val_pet_loss = val_outputs["pet_loss"]
                            val_current_pet_lambda = get_pet_lambda(global_step, total_steps_planned)
                            val_total_loss = val_ce_loss + val_current_pet_lambda * val_pet_loss

                        val_loss += val_total_loss.item()
                        val_acc += accuracy(val_logits, val_labels)
                        val_samples += 1

                val_loss /= val_samples
                val_acc /= val_samples

                val_log_entry = {
                    "step": global_step,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(val_log_entry) + "\n")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    steps_since_improve = 0
                    checkpoint = checkpoint_dir / f"{args.output_name}_best.pt"
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": model_config.__dict__,
                            "vocab_path": str(vocab_path),
                            "label_to_id": label_to_id,
                        },
                        checkpoint,
                    )
                else:
                    steps_since_improve += 1

                model.train()

            if training_cfg.get("max_steps") and global_step >= training_cfg["max_steps"]:
                break

        if training_cfg.get("max_steps") and global_step >= training_cfg["max_steps"]:
            break
        if patience and steps_since_improve >= patience:
            break

    final_checkpoint = checkpoint_dir / f"{args.output_name}_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model_config.__dict__,
            "vocab_path": str(vocab_path),
            "label_to_id": label_to_id,
        },
        final_checkpoint,
    )
    print(f"Training complete. Final checkpoint saved to {final_checkpoint}")


if __name__ == "__main__":
    main()
