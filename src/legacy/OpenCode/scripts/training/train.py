from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR
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
    parser = argparse.ArgumentParser(description="Train the SO8T model with PET regularization.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_default.yaml").resolve(), help="Training config path.")
    return parser.parse_args()


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: Optional[int],
) -> Optional[LambdaLR]:
    if total_steps is None or total_steps <= 0:
        return None

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    pet_lambda: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            pet_loss = outputs["pet_loss"]
            loss = loss_fn(logits, labels) + pet_lambda * pet_loss

            preds = torch.argmax(logits, dim=-1)
            total_acc += accuracy(preds, labels)
            total_loss += loss.item()
            total_steps += 1
    if total_steps == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": total_loss / total_steps, "accuracy": total_acc / total_steps}


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    paths_cfg = config["paths"]
    training_cfg = config["training"]
    model_cfg = config["model"]
    runtime_cfg = config["runtime"]
    scheduler_cfg = config.get("scheduler", {})
    optimizer_cfg = config.get("optimizer", {})

    set_seed(runtime_cfg.get("seed", 42))
    device = resolve_device(runtime_cfg.get("device", "auto"))
    label_to_id, id_to_label = default_labels()

    train_path = Path(paths_cfg["train"])
    val_path = Path(paths_cfg["val"])

    vocab = build_vocab_from_files([train_path, val_path])
    vocab_path = Path(paths_cfg.get("checkpoint_dir", "chk")) / f"{paths_cfg.get('run_name', 'run')}_vocab.json"
    vocab.to_file(vocab_path)

    max_seq_len = runtime_cfg.get("max_seq_len", 96)
    train_dataset = DialogueDataset(train_path, vocab, label_to_id, max_seq_len)
    val_dataset = DialogueDataset(val_path, vocab, label_to_id, max_seq_len)

    train_loader = build_dataloader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=runtime_cfg.get("num_workers", 0),
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=runtime_cfg.get("num_workers", 0),
    )

    model_config = ModelConfig(
        vocab_size=len(vocab),
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
        num_labels=len(label_to_id),
        max_seq_len=max_seq_len,
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
    
    # SWA設定：広い谷の庸解を最終形にする
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, anneal_strategy="linear", anneal_epochs=2, swa_lr=5e-4)

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

    log_path = Path(paths_cfg.get("checkpoint_dir", "chk")) / f"{paths_cfg.get('run_name', 'run')}_train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "chk"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = 0.0
    global_step = 0
    patience = training_cfg.get("early_stopping_patience")
    steps_since_improve = 0

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

            preds = torch.argmax(logits, dim=-1)
            train_acc = accuracy(preds, labels)
            progress.set_postfix({"loss": loss.item(), "acc": train_acc})

            global_step += 1
            log_entry = {
                "step": global_step,
                "epoch": epoch + 1,
                "loss": loss.item(),
                "ce_loss": ce_loss.item(),
                "pet_loss": pet_loss.item(),
                "train_accuracy": train_acc,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_entry) + "\n")

            if training_cfg.get("log_interval") and global_step % training_cfg["log_interval"] == 0:
                progress.set_postfix({"loss": loss.item(), "acc": train_acc})

            if training_cfg.get("eval_interval") and global_step % training_cfg["eval_interval"] == 0:
                # Use the correct PET lambda for evaluation
                current_pet_lambda = get_pet_lambda(global_step, total_steps_planned)
                metrics = evaluate(model, val_loader, device, loss_fn, current_pet_lambda)
                if metrics["accuracy"] > best_val_accuracy:
                    best_val_accuracy = metrics["accuracy"]
                    steps_since_improve = 0
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": model_config.__dict__,
                            "vocab_path": str(vocab_path),
                            "label_to_id": label_to_id,
                        },
                        checkpoint_dir / f"{paths_cfg.get('run_name', 'run')}_best.pt",
                    )
                else:
                    steps_since_improve += 1
                status = {
                    "step": global_step,
                    "val_loss": metrics["loss"],
                    "val_accuracy": metrics["accuracy"],
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(status) + "\n")
                if patience and steps_since_improve >= patience:
                    print("Early stopping triggered.")
                    break

            if training_cfg.get("max_steps") and global_step >= training_cfg["max_steps"]:
                break

        if training_cfg.get("max_steps") and global_step >= training_cfg["max_steps"]:
            break
        if patience and steps_since_improve >= patience:
            break
        
        # SWA更新：70%以降のみで重みを平均化
        progress = global_step / total_steps_planned
        if progress >= 0.7:
            swa_model.update_parameters(model)
            swa_scheduler.step()

    # SWA最終更新
    swa_model.update_parameters(model)
    
    final_checkpoint = checkpoint_dir / f"{paths_cfg.get('run_name', 'run')}_final.pt"
    swa_checkpoint = checkpoint_dir / f"{paths_cfg.get('run_name', 'run')}_swa_final.pt"
    
    # 通常モデルとSWAモデルの両方を保存
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model_config.__dict__,
            "vocab_path": str(vocab_path),
            "label_to_id": label_to_id,
        },
        final_checkpoint,
    )
    
    torch.save(
        {
            "model_state_dict": swa_model.module.state_dict(),
            "config": model_config.__dict__,
            "vocab_path": str(vocab_path),
            "label_to_id": label_to_id,
        },
        swa_checkpoint,
    )
    print(f"Training complete. Final checkpoints saved to {final_checkpoint} and {swa_checkpoint}")


if __name__ == "__main__":
    main()
