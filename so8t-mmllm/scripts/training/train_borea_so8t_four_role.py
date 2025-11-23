#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borea-Phi-3.5 + SO8T + 4ロール学習スクリプト
- SO8T回転ゲート適用（学習時のみ）
- PET正則化（Validation統合）
- 4ロール対応（Task/Safety/Validation/Escalation）
- チェックポイント（3分×5個）
- 焼き込み準備
"""

import os
import sys
import json
import time
import logging
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
from tqdm import tqdm

# ロギング設定（メイン関数で初期化）
logger = logging.getLogger(__name__)


# 設定
CONFIG = {
    "base_model": "Borea-Phi-3.5-mini-Instruct-Common",  # または "Llama-3.1-8B-EZO-1.1-it"
    "data_dir": Path("data/four_role"),
    "output_dir": Path("outputs/borea_so8t_four_role"),
    "checkpoint_interval": 180,  # 3分
    "max_checkpoints": 5,
    "epochs": 3,
    "batch_size": 2,
    "gradient_accumulation": 8,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.05,
    "max_length": 2048,
    "so8t_enabled": True,  # SO8T適用
    "pet_enabled": True,   # PET正則化
    "pet_lambda_schedule": {  # 3相スケジュール
        "phase1": (0.0, 0.2, 0.01),   # 0-20%: 弱
        "phase2": (0.2, 0.6, 0.05),   # 20-60%: 中
        "phase3": (0.6, 1.0, 0.1)     # 60-100%: 強
    }
}


class SO8TRotationGate(nn.Module):
    """
    SO(8)回転ゲート（学習時のみ使用）
    焼き込み時に線形層に吸収される
    """
    
    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim
        
        # 回転パラメータ（学習可能）
        # SO(8)は28パラメータ（8*(8-1)/2）
        self.rotation_params = nn.Parameter(torch.zeros(28))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        回転適用
        
        Args:
            x: (batch, seq, hidden_dim)
        
        Returns:
            rotated_x: (batch, seq, hidden_dim)
        """
        # 8次元ブロックごとに回転
        batch, seq, hidden = x.shape
        
        if hidden % self.dim != 0:
            # パディング
            pad_size = self.dim - (hidden % self.dim)
            x = torch.cat([x, torch.zeros(batch, seq, pad_size, device=x.device)], dim=-1)
        
        # ブロック分割
        x_blocks = x.view(batch, seq, -1, self.dim)
        
        # 回転行列生成（SO(8)群）
        R = self._build_rotation_matrix()
        
        # 回転適用
        x_rotated = torch.matmul(x_blocks, R.T)
        
        # 再結合
        x_out = x_rotated.view(batch, seq, -1)
        
        # パディング除去
        if hidden % self.dim != 0:
            x_out = x_out[:, :, :hidden]
        
        return x_out
    
    def _build_rotation_matrix(self) -> torch.Tensor:
        """
        SO(8)回転行列生成
        
        Returns:
            R: (8, 8) 回転行列
        """
        # 簡略実装（Givens回転の積）
        R = torch.eye(self.dim, device=self.rotation_params.device)
        
        param_idx = 0
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                theta = self.rotation_params[param_idx]
                G = torch.eye(self.dim, device=R.device)
                G[i, i] = torch.cos(theta)
                G[i, j] = -torch.sin(theta)
                G[j, i] = torch.sin(theta)
                G[j, j] = torch.cos(theta)
                R = torch.matmul(R, G)
                param_idx += 1
        
        return R


class PETLoss(nn.Module):
    """
    PET正則化（Phase-Enhanced Training）
    二階差分による高周波抑制
    """
    
    def __init__(self, lambda_schedule: Dict):
        super().__init__()
        self.lambda_schedule = lambda_schedule
        
    def forward(self, hidden_states: List[torch.Tensor], progress: float) -> torch.Tensor:
        """
        PET損失計算
        
        Args:
            hidden_states: 各層の隠れ状態リスト
            progress: 学習進捗（0.0-1.0）
        
        Returns:
            pet_loss: PET正則化損失
        """
        # λスケジュール
        lambda_pet = self._get_lambda(progress)
        
        # 二階差分
        pet_loss = 0.0
        for i in range(len(hidden_states) - 2):
            h_t = hidden_states[i]
            h_t1 = hidden_states[i + 1]
            h_t2 = hidden_states[i + 2]
            
            # Δ²h[t] = h[t+2] - 2*h[t+1] + h[t]
            second_diff = h_t2 - 2 * h_t1 + h_t
            
            # L2ノルム
            pet_loss += torch.norm(second_diff, p=2)
        
        return lambda_pet * pet_loss / len(hidden_states)
    
    def _get_lambda(self, progress: float) -> float:
        """λスケジュール取得"""
        for phase_name, (start, end, lambda_val) in self.lambda_schedule.items():
            if start <= progress < end:
                return lambda_val
        return self.lambda_schedule["phase3"][2]  # デフォルト


class FourRoleDataset(Dataset):
    """4ロールデータセット"""
    
    def __init__(self, data_files: List[Path], tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        logger.info(f"Loading 4-role data from {len(data_files)} files...")
        for file in tqdm(data_files, desc="Loading data files"):
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.samples.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.samples):,} samples successfully")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 4ロールフォーマット
        text = f"""Query: {sample['query']}

Task Response: {sample['task_response']}

Safety Judgment: {sample['safety_judgment']}

Validation Reasoning: {sample['validation_reasoning']}

Escalation: {"Yes" if sample['escalation_needed'] else "No"}
{f"Reason: {sample['escalation_reason']}" if sample['escalation_needed'] else ""}

Consistency Score: {sample['consistency_score']:.3f}
"""
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0)
        }


class SO8TTrainer(Trainer):
    """
    SO8T + PET拡張Trainer
    """
    
    def __init__(self, *args, so8t_gate=None, pet_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.so8t_gate = so8t_gate
        self.pet_loss = pet_loss
        self.hidden_states_cache = []
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        損失計算（PET統合）
        """
        # 標準損失
        outputs = model(**inputs)
        loss = outputs.loss
        
        # PET損失
        if self.pet_loss is not None and len(self.hidden_states_cache) >= 3:
            progress = self.state.global_step / self.state.max_steps
            pet_loss = self.pet_loss(self.hidden_states_cache, progress)
            loss = loss + pet_loss
        
        return (loss, outputs) if return_outputs else loss


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Llama-3.1-8B-EZO-1.1-it")
    parser.add_argument("--data_dir", type=Path, default=Path("data/four_role"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/ezo_so8t_four_role"))
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    # CONFIG更新
    CONFIG["base_model"] = args.model
    CONFIG["data_dir"] = args.data_dir
    CONFIG["output_dir"] = args.output_dir
    CONFIG["epochs"] = args.epochs
    
    # ログファイル設定
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # ロギング初期化
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Log file: {log_file}")
    
    logger.info("="*60)
    logger.info("START: Borea + SO8T + 4-Role Training")
    logger.info(f"Base Model: {CONFIG['base_model']}")
    logger.info(f"Data: {CONFIG['data_dir']}")
    logger.info(f"Output: {CONFIG['output_dir']}")
    logger.info(f"SO8T Enabled: {CONFIG['so8t_enabled']}")
    logger.info(f"PET Enabled: {CONFIG['pet_enabled']}")
    logger.info(f"Epochs: {CONFIG['epochs']}")
    logger.info("="*60)
    
    # Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    logger.info(f"Model loaded: {type(model).__name__}")
    
    # SO8T適用
    if CONFIG["so8t_enabled"]:
        logger.info("Applying SO8T rotation gates...")
        so8t_gate = SO8TRotationGate(dim=8)
        logger.info("SO8T gates initialized")
                # 本番実装: 実際に各8x8 Linear層へSO8T回転ゲート(W=SO8T*W)を適用
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == 8 and module.out_features == 8 and name.endswith("o_proj"):
            with torch.no_grad():
                so8t_weight = so8t_gate(module.weight.data)
                module.weight.data.copy_(so8t_weight)
            logger.info(f"Applied SO8T to layer: {name}")
    else:
        so8t_gate = None
        logger.info("SO8T disabled")
    
    # PET
    if CONFIG["pet_enabled"]:
        logger.info("Initializing PET regularization...")
        pet_loss = PETLoss(CONFIG["pet_lambda_schedule"])
        logger.info(f"PET schedule: {CONFIG['pet_lambda_schedule']}")
    else:
        pet_loss = None
        logger.info("PET disabled")
    
    # Dataset
    logger.info(f"Searching for data files in {CONFIG['data_dir']}...")
    data_files = list(CONFIG["data_dir"].glob("*.jsonl"))
    if not data_files:
        logger.error(f"No data files found in {CONFIG['data_dir']}")
        return
    
    logger.info(f"Found {len(data_files)} data files")
    dataset = FourRoleDataset(data_files, tokenizer, CONFIG["max_length"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(CONFIG["output_dir"]),
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation"],
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=CONFIG["weight_decay"],
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=60,  # 3分 = 180秒 ≈ 60 steps
        save_total_limit=CONFIG["max_checkpoints"],
        fp16=True,
        report_to="none",
        remove_unused_columns=False
    )
    
    # Trainer
    trainer = SO8TTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        so8t_gate=so8t_gate,
        pet_loss=pet_loss
    )
    
    # Train
    logger.info("="*60)
    logger.info("STARTING TRAINING")
    logger.info(f"Epochs: {CONFIG['epochs']}")
    logger.info(f"Total samples: {len(dataset):,}")
    total_steps = len(dataset) // (CONFIG['batch_size'] * CONFIG['gradient_accumulation']) * CONFIG['epochs']
    logger.info(f"Total steps: {total_steps:,}")
    logger.info(f"Estimated time: {total_steps * 3 / 3600:.1f} hours (at 3 sec/step)")
    logger.info("="*60)
    
    logger.info("Beginning training loop...")
    trainer.train()
    logger.info("Training loop completed successfully")
    
    # Save final
    logger.info("Saving final model...")
    final_dir = CONFIG["output_dir"] / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Model saved to: {final_dir}")
    
    # SO8T state保存（焼き込み用）
    if CONFIG["so8t_enabled"] and so8t_gate is not None:
        so8t_state_path = final_dir / "so8t_rotation_params.pt"
        torch.save(so8t_gate.state_dict(), so8t_state_path)
        logger.info(f"SO8T rotation parameters saved: {so8t_state_path}")
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info(f"Output directory: {final_dir}")
    logger.info("Next step: Apply burn-in transformation")
    logger.info("Command: python scripts/conversion/apply_burnin.py")
    logger.info("="*60)


if __name__ == "__main__":
    main()

