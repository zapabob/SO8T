#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡略版学習スクリプト（動作確認用）
HuggingFace標準Trainer使用、依存問題回避
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


class SimpleDataset(Dataset):
    """簡易データセット"""
    
    def __init__(self, data_files, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"[LOAD] Loading from {len(data_files)} files...")
        for file in data_files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    self.samples.append(data)
        
        print(f"[OK] Loaded {len(self.samples):,} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # テキスト取得
        if "query" in sample and "response" in sample:
            text = f"Query: {sample['query']}\nResponse: {sample['response']}"
        elif "text" in sample:
            text = sample["text"]
        else:
            text = str(sample)
        
        # トークナイズ
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


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="microsoft/phi-2")  # Phi-2使用（軽量）
    parser.add_argument("--data_dir", type=Path, default=Path("data/validated"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/simple_finetuned"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"[START] Simple Training")
    print(f"Model: {args.model}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Tokenizer
    print("[LOAD] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model
    print("[LOAD] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Dataset
    data_files = list(args.data_dir.glob("*.jsonl"))
    if not data_files:
        print(f"[ERROR] No data files found in {args.data_dir}")
        return
    
    dataset = SimpleDataset(data_files, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        report_to="none"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train
    print("\n[TRAIN] Starting training...")
    trainer.train()
    
    # Save
    print("\n[SAVE] Saving model...")
    trainer.save_model(str(args.output_dir / "final_model"))
    tokenizer.save_pretrained(str(args.output_dir / "final_model"))
    
    print(f"\n{'='*60}")
    print(f"[OK] Training completed!")
    print(f"Model saved to: {args.output_dir / 'final_model'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

