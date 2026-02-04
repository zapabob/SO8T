#!/usr/bin/env python3
"""
Safety-Aware SO8T Training v2.0
- 安全崩壊を防ぐための統合損失関数
- Safety Score/Refuse Recallベースの早期停止
- 安全ヘッドとタスクヘッドの分離最適化
"""

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared.data_backup import SessionCheckpointManager
from shared.data import DialogueDataset, collate_batch
from shared.vocab import Vocabulary
from shared.utils import set_seed, load_yaml
from safety_losses import SafetyAwareLoss, SafetyMetrics
from agents.so8t.model_safety import build_safety_model, SafetyModelConfig


def create_safety_loss_function(config: Dict[str, Any]) -> nn.Module:
    """安全重視の統合損失関数を作成"""
    return SafetyAwareLoss(
        task_loss_weight=config.get('task_loss_weight', 1.0),
        safety_loss_weight=config.get('safety_loss_weight', 2.0),  # 安全を重視
        pet_loss_weight=config.get('pet_loss_weight', 0.1),
        safety_penalty_weight=config.get('safety_penalty_weight', 5.0),  # 危険な従順に重い罰
        escalate_reward_weight=config.get('escalate_reward_weight', 2.0)  # ESCALATEを推奨
    )


def train_epoch_safety_aware(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    safety_optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    loss_fn: SafetyAwareLoss,
    device: torch.device,
    vocab_size: int,
    checkpoint_mgr: SessionCheckpointManager = None,
    global_step: int = 0
) -> Dict[str, float]:
    """安全重視の学習エポック"""
    model.train()
    
    total_loss = 0.0
    total_task_loss = 0.0
    total_safety_loss = 0.0
    total_pet_loss = 0.0
    total_safety_penalty = 0.0
    total_escalate_reward = 0.0
    
    correct_predictions = 0
    total_samples = 0
    
    all_task_logits = []
    all_safety_logits = []
    all_targets = []
    
    last_autosave_time = time.time()
    
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 入力ノイズ注入でPETの気持ちよさを制限
        input_ids, attention_mask = add_input_noise(input_ids, attention_mask, noise_prob=0.2, vocab_size=vocab_size)
        
        # タスクヘッドの最適化
        optimizer.zero_grad()
        
        with autocast(enabled=scaler.is_enabled()):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            task_logits = outputs["task_logits"]
            safety_logits = outputs["safety_logits"]
            pet_loss = outputs["pet_loss"]
            
            # 安全重視の統合損失
            loss_dict = loss_fn(
                task_logits=task_logits,
                safety_logits=safety_logits,
                targets=labels,
                pet_loss=pet_loss
            )
            
            total_loss_val = loss_dict["total_loss"]
        
        scaler.scale(total_loss_val).backward()
        
        # 勾配ノイズ注入で局所安定解から蹴り出す
        sigma = 0.025
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    continue
                noise = torch.randn_like(p.grad) * sigma
                p.grad.add_(noise)
        
        scaler.step(optimizer)
        scaler.update()
        
        # 安全ヘッドの別最適化（一時的に無効化）
        # TODO: 安全ヘッドの分離最適化を実装
        safety_penalty = torch.tensor(0.0, device=device)
        escalate_reward = torch.tensor(0.0, device=device)
        
        # 統計を更新
        total_loss += total_loss_val.item()
        total_task_loss += loss_dict["task_loss"].item()
        total_safety_loss += loss_dict["safety_loss"].item()
        total_pet_loss += loss_dict["pet_loss"].item()
        total_safety_penalty += safety_penalty.item()
        total_escalate_reward += escalate_reward.item()
        
        # 精度計算
        predictions = torch.argmax(task_logits, dim=-1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        # ログ用データ収集
        all_task_logits.append(task_logits.detach().cpu())
        all_safety_logits.append(safety_logits.detach().cpu())
        all_targets.append(labels.detach().cpu())
        
        # 緊急保存チェック
        if checkpoint_mgr and checkpoint_mgr.check_emergency_save():
            checkpoint_mgr.emergency_save(
                model, optimizer, scaler, None,
                {'epoch': 'unknown', 'step': global_step, 'batch': batch_idx}
            )
        
        # 5分間隔のオートセーブ
        current_time = time.time()
        if checkpoint_mgr and (current_time - last_autosave_time) >= 300:
            checkpoint_mgr.save(
                model, optimizer, scaler, None,
                {'epoch': 'unknown', 'step': global_step, 'batch': batch_idx}
            )
            last_autosave_time = current_time
        
        global_step += 1
        
        # プログレスバー更新
        progress_bar.set_postfix({
            'Loss': f'{total_loss_val.item():.4f}',
            'Acc': f'{correct_predictions/total_samples:.4f}',
            'Safety': f'{loss_dict.get("safety_loss", 0):.4f}'
        })
    
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
        'safety_penalty': total_safety_penalty / len(dataloader),
        'escalate_reward': total_escalate_reward / len(dataloader),
        'accuracy': correct_predictions / total_samples,
        **dual_metrics
    }


def validate_safety_aware(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: SafetyAwareLoss,
    device: torch.device
) -> Dict[str, float]:
    """安全重視のバリデーション"""
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
        progress_bar = tqdm(dataloader, desc="Validation", unit="batch")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast(enabled=True):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                task_logits = outputs["task_logits"]
                safety_logits = outputs["safety_logits"]
                pet_loss = outputs["pet_loss"]
                
                loss_dict = loss_fn(
                    task_logits=task_logits,
                    safety_logits=safety_logits,
                    targets=labels,
                    pet_loss=pet_loss
                )
            
            total_loss += loss_dict["total_loss"].item()
            total_task_loss += loss_dict["task_loss"].item()
            total_safety_loss += loss_dict["safety_loss"].item()
            total_pet_loss += loss_dict["pet_loss"].item()
            
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
    
    dual_metrics = SafetyMetrics.dual_safety_metrics(all_task_logits, all_safety_logits, all_targets)
    
    return {
        'val_loss': total_loss / len(dataloader),
        'val_task_loss': total_task_loss / len(dataloader),
        'val_safety_loss': total_safety_loss / len(dataloader),
        'val_pet_loss': total_pet_loss / len(dataloader),
        'val_accuracy': correct_predictions / total_samples,
        **{f'val_{k}': v for k, v in dual_metrics.items()}
    }


def add_input_noise(input_ids: torch.Tensor, attention_mask: torch.Tensor, noise_prob: float = 0.1, vocab_size: int = 793) -> Tuple[torch.Tensor, torch.Tensor]:
    """入力にノイズを注入してPETの過度な安定化を防ぐ"""
    if noise_prob <= 0:
        return input_ids, attention_mask
    
    # ランダムにトークンを置換（実際の語彙サイズを使用）
    noise_mask = torch.rand_like(input_ids.float()) < noise_prob
    noise_tokens = torch.randint(1, vocab_size, input_ids.shape, device=input_ids.device)
    
    noisy_input_ids = input_ids.clone()
    noisy_input_ids[noise_mask] = noise_tokens[noise_mask]
    
    return noisy_input_ids, attention_mask


def main():
    parser = argparse.ArgumentParser(description="Safety-Aware SO8T Training v2.0")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # 設定を読み込み
    config = load_yaml(Path(args.config))
    set_seed(args.seed)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # セッション管理
    session_id = str(uuid.uuid4())[:8]
    print(f"Session ID: {session_id}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # チェックポイント管理
    checkpoint_mgr = SessionCheckpointManager(
        output_dir / "autosave",
        max_backups=10,
        session_id=session_id
    )
    
    # データセット準備
    print("Loading data...")
    vocab = Vocabulary.from_file(Path(args.data_dir) / "vocab.json")
    
    train_dataset = DialogueDataset(
        path=Path(args.data_dir) / "train.jsonl",
        vocab=vocab,
        label_to_id={"COMPLY": 0, "REFUSE": 1, "ESCALATE": 2},
        max_seq_len=config.get('max_length', 512)
    )
    
    val_dataset = DialogueDataset(
        path=Path(args.data_dir) / "val.jsonl",
        vocab=vocab,
        label_to_id={"COMPLY": 0, "REFUSE": 1, "ESCALATE": 2},
        max_seq_len=config.get('max_length', 512)
    )
    
    def collate_fn(batch):
        return collate_batch(batch, pad_index=vocab.pad_index)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # モデル構築
    print("Building model...")
    model_config = SafetyModelConfig(
        vocab_size=len(vocab),
        d_model=config.get('d_model', 256),
        n_heads=config.get('num_heads', 8),
        n_layers=config.get('num_layers', 6),
        d_ff=config.get('d_model', 256) * 4,
        dropout=config.get('dropout', 0.1),
        num_labels=3,  # COMPLY, REFUSE, ESCALATE
        num_safety_labels=3,  # ALLOW, REFUSE, ESCALATE
        max_seq_len=config.get('max_length', 512),
        gate_order=["R_env", "R_safe", "R_cmd"],
        safety_first=config.get('safety_first', True)
    )
    
    model = build_safety_model(model_config).to(device)
    
    # パラメータ数を表示
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # 損失関数
    loss_fn = create_safety_loss_function(config)
    
    # オプティマイザー（分離）
    task_params = []
    safety_params = []
    
    for name, param in model.named_parameters():
        if 'safety' in name.lower():
            safety_params.append(param)
        else:
            task_params.append(param)
    
    optimizer = torch.optim.AdamW(
        task_params,
        lr=float(config.get('learning_rate', 1e-4)),
        weight_decay=float(config.get('weight_decay', 0.01))
    )
    
    safety_optimizer = torch.optim.AdamW(
        safety_params,
        lr=float(config.get('safety_learning_rate', 1e-5)),  # 低学習率
        weight_decay=float(config.get('weight_decay', 0.01))
    )
    
    scaler = GradScaler()
    
    # 学習ループ
    print("Starting training...")
    
    best_safety_score = 0.0
    best_refuse_recall = 0.0
    patience = 0
    max_patience = config.get('patience', 3)
    
    global_step = 0
    
    for epoch in range(config.get('num_epochs', 5)):
        print(f"\nEpoch {epoch+1}/{config.get('num_epochs', 5)}")
        
        # 学習
        train_metrics = train_epoch_safety_aware(
            model, train_loader, optimizer, safety_optimizer, scaler, loss_fn, device, len(vocab), checkpoint_mgr, global_step
        )
        
        # バリデーション
        val_metrics = validate_safety_aware(model, val_loader, loss_fn, device)
        
        # ログ保存
        log_entry = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'session_id': session_id,
            **train_metrics,
            **val_metrics
        }
        
        with open(output_dir / "safety_training_log.jsonl", "a", encoding="utf-8") as f:
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
        
        # 安全ベースの早期停止
        current_safety_score = val_metrics['val_combined_safety_score']
        current_refuse_recall = val_metrics['val_task_refuse_recall']
        
        # 安全スコアまたはRefuse Recallが改善した場合
        if current_safety_score > best_safety_score or current_refuse_recall > best_refuse_recall:
            best_safety_score = max(best_safety_score, current_safety_score)
            best_refuse_recall = max(best_refuse_recall, current_refuse_recall)
            patience = 0
            
            # ベストモデルを保存
            checkpoint_path = output_dir / "safety_model_best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "safety_optimizer_state_dict": safety_optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": model_config.__dict__,
                "vocab_path": str(Path(args.data_dir) / "vocab.json"),
                "task_label_to_id": {"COMPLY": 0, "REFUSE": 1, "ESCALATE": 2},
                "safety_label_to_id": {"ALLOW": 0, "REFUSE": 1, "ESCALATE": 2},
                "safety_score": current_safety_score,
                "refuse_recall": current_refuse_recall,
                "epoch": epoch + 1,
                "global_step": global_step
            }, checkpoint_path)
            
            print(f"New best safety score: {current_safety_score:.4f}")
            print(f"New best refuse recall: {current_refuse_recall:.4f}")
            print(f"Model saved to: {checkpoint_path}")
        else:
            patience += 1
            print(f"No improvement. Patience: {patience}/{max_patience}")
            
            if patience >= max_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        global_step += len(train_loader)
    
    print(f"\nTraining completed!")
    print(f"Best safety score: {best_safety_score:.4f}")
    print(f"Best refuse recall: {best_refuse_recall:.4f}")
    print(f"Logs saved to: {output_dir / 'safety_training_log.jsonl'}")


if __name__ == "__main__":
    main()
