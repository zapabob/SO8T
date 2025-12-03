#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
トレーニングデバッグスクリプト
"""

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

class SimpleDataset(Dataset):
    def __init__(self, tokenizer, num_samples=10):
        self.tokenizer = tokenizer
        self.data = [
            "Hello world, this is a test.",
            "The SO(8) group has 8 dimensions.",
            "Machine learning is fascinating.",
            "PyTorch is great for deep learning.",
            "Transformers changed NLP forever."
        ] * (num_samples // 5 + 1)
        self.data = self.data[:num_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        # トークナイズ（勾配計算可能なテンソルとして）
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze()
        }

def test_training():
    print("=== Training Debug Test ===")

    # モデルとトークナイザー
    model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)

    # LoRAパラメータをトレーニング可能に
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True

    print(f"Trainable params: {model.print_trainable_parameters()}")

    # データセット
    dataset = SimpleDataset(tokenizer, num_samples=10)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # トレーニング設定
    training_args = TrainingArguments(
        output_dir="debug_output",
        num_train_epochs=1,
        max_steps=5,  # 最小ステップ
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        logging_steps=1,
        gradient_checkpointing=False,  # 一旦無効化
        optim="adamw_8bit",
        bf16=True,
        report_to=[],
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # テスト実行
    print("Starting training test...")
    try:
        trainer.train()
        print("✅ Training test successful!")
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training()

