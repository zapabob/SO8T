#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8) Phi-3.5 Adapter Training Script
SO(8)アダプターを適用したPhi-3.5モデルの学習スクリプト

このスクリプトは以下の処理を行います：
1. Phi-3.5モデルの読み込み
2. SO(8) Compatible LoRAアダプターの注入
3. データセットでの学習
4. 標準LoRA形式への変換と保存
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import logging
import time
import argparse

# SO8T components
from so8_compatible_adapter import (
    SO8CompatibleLoRA,
    inject_so8_adapter_into_model,
    save_as_standard_lora
)

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """シンプルなテキストデータセット"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # データ読み込み
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if 'text' in item:
                    self.data.append(item['text'])

        logger.info(f"Loaded {len(self.data)} text samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        # トークナイズ
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }


def load_phi35_model(model_path: str):
    """Phi-3.5モデルの読み込み"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading Phi-3.5 model from {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def train_so8_adapter(
    model_path: str,
    dataset_path: str,
    output_path: str,
    num_epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    save_steps: int = 100,
    max_steps: int = 1000,
    rank: int = 8,
    alpha: float = 1.0
):
    """SO(8)アダプターの学習"""

    # モデル読み込み
    model, tokenizer = load_phi35_model(model_path)

    # SO(8)アダプター注入
    logger.info("Injecting SO(8) Compatible LoRA adapters...")
    injected_adapters = inject_so8_adapter_into_model(
        model,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        rank=rank,
        alpha=alpha
    )

    logger.info(f"Injected {len(injected_adapters)} SO(8) adapters")

    # データセット準備
    dataset = TextDataset(dataset_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # オプティマイザー設定
    optimizer = AdamW([
        {'params': model.parameters(), 'lr': learning_rate},
        # Lie代数パラメータはより小さな学習率で
        {'params': [adapter.lie_algebra_param for adapter in injected_adapters.values()], 'lr': learning_rate * 0.1},
    ])

    # 学習ループ
    model.train()
    global_step = 0
    total_loss = 0

    logger.info("Starting SO(8) adapter training...")

    progress_bar = tqdm(total=max_steps, desc="Training")

    for epoch in range(num_epochs):
        for batch in dataloader:
            if global_step >= max_steps:
                break

            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)

            # ラベル作成（次トークン予測）
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            # ログ出力
            if global_step % 10 == 0:
                avg_loss = total_loss / 10
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                total_loss = 0

            progress_bar.update(1)

            # チェックポイント保存
            if global_step % save_steps == 0:
                checkpoint_path = Path(output_path) / f"checkpoint_{global_step}"
                checkpoint_path.mkdir(exist_ok=True)

                # 現在のモデル状態を保存
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path / "checkpoint.pt")

                logger.info(f"Saved checkpoint at step {global_step}")

    progress_bar.close()

    # 学習完了後の標準LoRA変換と保存
    logger.info("Converting to standard LoRA format...")

    # モデルを評価モードに
    model.eval()

    # 標準LoRAとして保存
    save_as_standard_lora(model, injected_adapters, output_path)

    logger.info(f"SO(8) adapter training completed! Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SO(8) Phi-3.5 Adapter Training")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Phi-3.5 model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank (SO(8) fixed to 8)")
    parser.add_argument("--alpha", type=float, default=1.0, help="LoRA alpha")

    args = parser.parse_args()

    # 出力ディレクトリ作成
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # 学習実行
    train_so8_adapter(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        rank=args.rank,
        alpha=args.alpha
    )


if __name__ == "__main__":
    main()
