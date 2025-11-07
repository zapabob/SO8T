#!/usr/bin/env python3
"""
Safety-Aware SO8T Training Script
安全判断を重視したSO8Tモデルの訓練
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime

from agents.so8t.model_safety import SafetyModelConfig, build_safety_model
from shared.data import DialogueDataset, build_dataloader, build_vocab_from_files, default_labels
from shared.data_backup import SessionCheckpointManager
from shared.utils import load_yaml, resolve_device, set_seed
from safety_losses import SafetyAwareLoss, SafetyMetrics


def smooth_ce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 0.3) -> torch.Tensor:
    """ラベルスムージング付きクロスエントロピー損失"""
    num_classes = logits.size(-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(eps / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1 - eps)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return -(true_dist * log_probs).sum(dim=-1).mean()


def add_input_noise(input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                   noise_prob: float = 0.2, mask_token_id: int = 0) -> tuple:
    """入力にノイズを注入"""
    if random.random() < noise_prob:
        # ランダムにトークンをマスク
        noise_mask = torch.rand_like(input_ids.float()) < 0.1
        input_ids = input_ids.clone()
        input_ids[noise_mask] = mask_token_id
    return input_ids, attention_mask


def create_safety_aware_dataset(dataset: DialogueDataset) -> List[Dict[str, Any]]:
    """安全を重視したデータセットを作成"""
    safety_samples = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        tokens = sample.tokens
        label = sample.label
        
        # 危険なキーワードを検出
        text = " ".join(tokens).lower()
        danger_keywords = [
            "dangerous", "unsafe", "risk", "harmful", "illegal", "unauthorized",
            "classified", "secret", "confidential", "experimental", "untested"
        ]
        
        is_dangerous = any(keyword in text for keyword in danger_keywords)
        
        # 安全ラベルを決定
        if is_dangerous:
            if label == "REFUSE":
                safety_label = 1  # REFUSE
            elif label == "ESCALATE":
                safety_label = 2  # ESCALATE
            else:
                safety_label = 2  # 危険な場合はESCALATEに強制
        else:
            safety_label = 0  # ALLOW (COMPLY)
        
        safety_samples.append({
            'tokens': tokens,
            'label': label,
            'safety_label': safety_label,
            'is_dangerous': is_dangerous,
            'scenario': sample.scenario,
            'meta': sample.meta
        })
    
    return safety_samples


def train_epoch(model, dataloader, optimizer, scaler, safety_loss_fn, device, 
                training_cfg: Dict[str, Any], global_step: int, total_steps: int,
                checkpoint_mgr: Optional[SessionCheckpointManager] = None) -> Dict[str, float]:
    """1エポックの訓練"""
    model.train()
    
    # PETスケジュール制：3段階で激しく制御
    def get_pet_lambda(step: int, total_steps: int) -> float:
        base_lambda = training_cfg.get("pet_lambda", 1e-4)
        progress = step / total_steps
        # 0-20%: ほぼ0, 20-60%: 0.1倍, 60-100%: 1.0倍
        if progress < 0.2:
            return base_lambda * 0.01  # 探索期はほぼ0
        elif progress < 0.6:
            return base_lambda * 0.1   # 中盤は弱く
        else:
            return base_lambda * 1.0   # 終盤は強く
    
    total_loss = 0.0
    total_task_loss = 0.0
    total_safety_loss = 0.0
    total_pet_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # 安全メトリクス
    all_task_logits = []
    all_safety_logits = []
    all_targets = []
    
    # オートセーブ用のタイマー
    last_autosave_time = time.time()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 入力ノイズ注入でPETの気持ちよさを制限
        input_ids, attention_mask = add_input_noise(input_ids, attention_mask, noise_prob=0.2)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', enabled=scaler.is_enabled()):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            task_logits = outputs["task_logits"]
            safety_logits = outputs["safety_logits"]
            pet_loss = outputs["pet_loss"]
            
            # 安全重視損失
            loss_dict = safety_loss_fn(
                task_logits=task_logits,
                safety_logits=safety_logits,
                targets=labels,
                pet_loss=pet_loss
            )
            
            loss = loss_dict["total_loss"]
        
        scaler.scale(loss).backward()
        
        # 勾配ノイズ注入で局所安定解から蹴り出す
        sigma = 0.025  # ノイズ強さ（強化）
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    continue
                noise = torch.randn_like(p.grad) * sigma
                p.grad.add_(noise)
        
        scaler.step(optimizer)
        scaler.update()
        
        # 統計を更新
        total_loss += loss.item()
        total_task_loss += loss_dict["task_loss"].item()
        total_safety_loss += loss_dict["safety_loss"].item()
        total_pet_loss += pet_loss.item()
        
        # 予測精度
        predictions = torch.argmax(task_logits, dim=-1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        # 安全メトリクス用データを保存
        all_task_logits.append(task_logits.detach().cpu())
        all_safety_logits.append(safety_logits.detach().cpu())
        all_targets.append(labels.detach().cpu())
        
        # 緊急保存チェック
        if checkpoint_mgr and checkpoint_mgr.check_emergency_save():
            checkpoint_mgr.emergency_save(
                model, optimizer, scaler, None,  # schedulerは後で追加
                {'epoch': 'unknown', 'step': global_step, 'batch': batch_idx}
            )
        
        # 5分間隔のオートセーブ
        current_time = time.time()
        if checkpoint_mgr and (current_time - last_autosave_time) >= 300:  # 5分 = 300秒
            checkpoint_mgr.save(
                model, optimizer, scaler, None,  # schedulerは後で追加
                {'epoch': 'unknown', 'step': global_step, 'batch': batch_idx}
            )
            last_autosave_time = current_time
        
        global_step += 1
    
    # 安全メトリクスを計算
    all_task_logits = torch.cat(all_task_logits, dim=0)
    all_safety_logits = torch.cat(all_safety_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 両系統のKPIを計算
    dual_metrics = SafetyMetrics.dual_safety_metrics(all_task_logits, all_safety_logits, all_targets)
    
    return {
        'loss': total_loss / len(dataloader),
        'task_loss': total_task_loss / len(dataloader),
        'safety_loss': total_safety_loss / len(dataloader),
        'pet_loss': total_pet_loss / len(dataloader),
        'accuracy': correct_predictions / total_samples,
        **dual_metrics  # 両系統のKPIを展開
    }


def validate_epoch(model, dataloader, safety_loss_fn, device) -> Dict[str, float]:
    """検証エポック"""
    model.eval()
    
    total_loss = 0.0
    total_task_loss = 0.0
    total_safety_loss = 0.0
    total_pet_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    all_task_logits = []
    all_safety_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            task_logits = outputs["task_logits"]
            safety_logits = outputs["safety_logits"]
            pet_loss = outputs["pet_loss"]
            
            loss_dict = safety_loss_fn(
                task_logits=task_logits,
                safety_logits=safety_logits,
                targets=labels,
                pet_loss=pet_loss
            )
            
            total_loss += loss_dict["total_loss"].item()
            total_task_loss += loss_dict["task_loss"].item()
            total_safety_loss += loss_dict["safety_loss"].item()
            total_pet_loss += pet_loss.item()
            
            predictions = torch.argmax(task_logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            all_task_logits.append(task_logits.cpu())
            all_safety_logits.append(safety_logits.cpu())
            all_targets.append(labels.cpu())
    
    # 安全メトリクスを計算
    all_task_logits = torch.cat(all_task_logits, dim=0)
    all_safety_logits = torch.cat(all_safety_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 両系統のKPIを計算
    dual_metrics = SafetyMetrics.dual_safety_metrics(all_task_logits, all_safety_logits, all_targets)
    
    # 検証用のプレフィックスを追加
    val_dual_metrics = {f'val_{k}': v for k, v in dual_metrics.items()}
    
    return {
        'val_loss': total_loss / len(dataloader),
        'val_task_loss': total_task_loss / len(dataloader),
        'val_safety_loss': total_safety_loss / len(dataloader),
        'val_pet_loss': total_pet_loss / len(dataloader),
        'val_accuracy': correct_predictions / total_samples,
        **val_dual_metrics  # 両系統のKPIを展開
    }


def main():
    parser = argparse.ArgumentParser(description="Safety-Aware SO8T Training")
    parser.add_argument("--config", type=Path, default=Path("configs/train_safety.yaml"), 
                       help="Training config path.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), 
                       help="Data directory.")
    parser.add_argument("--output_dir", type=Path, default=Path("chk"), 
                       help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no_resume", action="store_true", help="Disable auto-resume from checkpoint")
    args = parser.parse_args()
    
    # 設定を読み込み
    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    config = load_yaml(config_path)
    model_config = config["model"]
    training_cfg = config["training"]
    paths_cfg = config["paths"]
    
    # デバイス設定
    device = resolve_device()
    print(f"Using device: {device}")
    
    # シード設定
    set_seed(args.seed)
    
    # セッション管理とオートセーブ機能を初期化
    checkpoint_mgr = SessionCheckpointManager(args.output_dir)
    session_info = checkpoint_mgr.get_session_info()
    print(f"Session ID: {session_info['session_id']}")
    if session_info['has_checkpoint'] and not args.no_resume:
        print(f"Found existing checkpoint from timestamp: {session_info['latest_timestamp']}")
        print("Will resume from latest checkpoint...")
    else:
        print("Starting fresh training session...")
    
    # 語彙を読み込み
    vocab_path = args.data_dir / "vocab.json"
    if vocab_path.exists():
        from shared.vocab import Vocabulary
        vocab = Vocabulary.from_file(vocab_path)
    else:
        print("Vocabulary file not found, building from data...")
        vocab = build_vocab_from_files([
            args.data_dir / "train.jsonl",
            args.data_dir / "val.jsonl",
            args.data_dir / "test.jsonl"
        ])
        vocab.to_file(vocab_path)
    
    # データを読み込み
    print("Loading data...")
    train_dataset = DialogueDataset(
        args.data_dir / "train.jsonl",
        vocab=vocab,
        label_to_id=training_cfg["label_to_id"],
        max_seq_len=model_config["max_seq_len"],
    )
    
    val_dataset = DialogueDataset(
        args.data_dir / "val.jsonl",
        vocab=vocab,
        label_to_id=training_cfg["label_to_id"],
        max_seq_len=model_config["max_seq_len"],
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # データローダーを作成
    train_dataloader = build_dataloader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = build_dataloader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=0
    )
    
    # モデルを作成
    print("Building model...")
    safety_model_config = SafetyModelConfig(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        n_layers=model_config["n_layers"],
        n_heads=model_config["n_heads"],
        d_ff=model_config["d_ff"],
        dropout=model_config["dropout"],
        num_labels=model_config["num_labels"],
        num_safety_labels=3,  # REFUSE, ESCALATE, ALLOW
        max_seq_len=model_config["max_seq_len"],
        gate_order=model_config["gate_order"],
        safety_first=True
    )
    
    model = build_safety_model(safety_model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # オプティマイザーとスケジューラー
    optimizer = AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"]
    )
    
    def lr_lambda(step):
        warmup_steps = training_cfg["warmup_steps"]
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # 混合精度
    scaler = GradScaler(enabled=training_cfg.get("mixed_precision", True) and device.type == "cuda")
    
    # チェックポイントからの復旧
    start_epoch = 0
    global_step = 0
    if session_info['has_checkpoint'] and not args.no_resume:
        checkpoint_data = checkpoint_mgr.load_latest()
        if checkpoint_data:
            try:
                model.load_state_dict(checkpoint_data['model_state_dict'])
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                if checkpoint_data.get('scaler_state_dict'):
                    scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
                if checkpoint_data.get('scheduler_state_dict'):
                    scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                
                # エポックとステップを復元
                meta = checkpoint_data.get('meta', {})
                start_epoch = meta.get('epoch', 0)
                global_step = meta.get('step', 0)
                
                print(f"✅ Resumed from checkpoint: epoch {start_epoch}, step {global_step}")
            except Exception as e:
                print(f"❌ Failed to resume from checkpoint: {e}")
                print("Starting fresh training...")
                start_epoch = 0
                global_step = 0
    
    # 安全損失関数
    safety_loss_fn = SafetyAwareLoss(
        task_loss_weight=1.0,
        safety_loss_weight=2.0,
        pet_loss_weight=0.1,
        refuse_weight=5.0,
        escalate_weight=3.0,
        comply_weight=1.0,
        use_focal=True
    )
    
    # 訓練ループ
    print("Starting training...")
    epochs = training_cfg["epochs"]
    best_safety_score = 0.0
    total_steps = len(train_dataloader) * epochs
    
    # ログファイル
    log_file = args.output_dir / "safety_training_log.jsonl"
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # 訓練
        train_metrics = train_epoch(
            model, train_dataloader, optimizer, scaler, safety_loss_fn, 
            device, training_cfg, global_step, total_steps, checkpoint_mgr
        )
        
        # 検証
        val_metrics = validate_epoch(model, val_dataloader, safety_loss_fn, device)
        
        # スケジューラー更新
        scheduler.step()
        
        # ログを記録
        log_entry = {
            "epoch": epoch + 1,
            "step": global_step,
            **train_metrics,
            **val_metrics
        }
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # 結果を表示
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Train Safety Score: {train_metrics['combined_safety_score']:.4f}")
        print(f"Train Refuse Recall: {train_metrics['task_refuse_recall']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['val_accuracy']:.4f}")
        print(f"Val Safety Score: {val_metrics['val_combined_safety_score']:.4f}")
        print(f"Val Refuse Recall: {val_metrics['val_task_refuse_recall']:.4f}")
        
        # ベストモデルを保存
        current_safety_score = val_metrics['val_combined_safety_score']
        if current_safety_score > best_safety_score:
            best_safety_score = current_safety_score
            checkpoint_path = args.output_dir / "safety_model_best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": safety_model_config.__dict__,
                "safety_score": current_safety_score,
                "epoch": epoch + 1,
            }, checkpoint_path)
            print(f"New best safety score: {current_safety_score:.4f}")
            print(f"Model saved to: {checkpoint_path}")
    
    print(f"\nTraining completed!")
    print(f"Best safety score: {best_safety_score:.4f}")
    print(f"Logs saved to: {log_file}")


if __name__ == "__main__":
    main()
