"""
SO8T Thinking Model QLoRA Training Script

Thinking形式データ（<think>...</think><final>...</final>）でSO8TThinkingModelを
QLoRA（8bit）で効率的に訓練する。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from bitsandbytes import BitsAndBytesConfig
import yaml

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "so8t-mmllm" / "src"))

from models.so8t_thinking_model import SO8TThinkingModel
from models.safety_aware_so8t import SafetyAwareSO8TConfig
from utils.thinking_utils import (
    load_thinking_dataset,
    parse_safety_label,
    parse_verifier_label,
)


SAFETY_LABEL_MAP = {"ALLOW": 0, "ESCALATE": 1, "REFUSE": 2}


class ThinkingDataset(Dataset):
    """Thinking形式データセット"""
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # プロンプトを構築
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")
        
        # テキストを構築
        if input_text:
            text = f"指示: {instruction}\n入力: {input_text}\n出力: {output}"
        else:
            text = f"指示: {instruction}\n出力: {output}"
        
        # トークナイズ
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        
        # ラベルを作成（output部分のみを学習対象とする）
        labels = enc["input_ids"].clone()
        
        # Safetyラベル
        safety_label_str = sample.get("safety_label", "ALLOW")
        safety_label = SAFETY_LABEL_MAP.get(safety_label_str, 0)
        
        # Verifierラベル
        verifier_dict = sample.get("verifier_label", {"logical": 1.0, "faithful": 1.0})
        verifier_logical, verifier_faithful = parse_verifier_label(verifier_dict)
        
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "safety_label": torch.tensor(safety_label, dtype=torch.long),
            "verifier_logical": torch.tensor(verifier_logical, dtype=torch.float32),
            "verifier_faithful": torch.tensor(verifier_faithful, dtype=torch.float32),
        }


def collate_fn(batch):
    """バッチをまとめる"""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    safety_labels = torch.stack([b["safety_label"] for b in batch])
    verifier_logical = torch.stack([b["verifier_logical"] for b in batch])
    verifier_faithful = torch.stack([b["verifier_faithful"] for b in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "safety_labels": safety_labels,
        "verifier_logical": verifier_logical,
        "verifier_faithful": verifier_faithful,
    }


class ThinkingTrainer(Trainer):
    """Thinking形式データ用のカスタムTrainer"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        カスタム損失計算
        - Language Model損失
        - Safety損失
        - Verifier損失
        """
        labels = inputs.pop("labels")
        safety_labels = inputs.pop("safety_labels")
        verifier_logical = inputs.pop("verifier_logical")
        verifier_faithful = inputs.pop("verifier_faithful")
        
        # Forward pass
        outputs = model(**inputs, labels=labels, safety_labels=safety_labels)
        
        # 損失を取得
        total_loss = outputs["loss"]
        
        if return_outputs:
            return total_loss, outputs
        return total_loss


def load_config(config_path: Path) -> Dict[str, Any]:
    """設定ファイルをロード"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train SO8T Thinking Model with QLoRA"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Config file path (YAML)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Thinking format dataset (JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--use-redacted",
        action="store_true",
        help="Use <think> format",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    
    args = parser.parse_args()
    
    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # データセットをロード
    print(f"[INFO] Loading dataset from: {args.dataset}")
    samples = load_thinking_dataset(args.dataset)
    print(f"[INFO] Loaded {len(samples)} samples")
    
    # トークナイザーをロード
    print(f"[INFO] Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # SO8T設定
    so8t_config = SafetyAwareSO8TConfig(
        num_safety_labels=3,
        num_verifier_dims=2,  # logical, faithful
        use_verifier_head=True,
        use_strict_so8_rotation=True,
    )
    
    # モデルをロード
    print(f"[INFO] Loading model: {args.base_model}")
    model = SO8TThinkingModel(
        base_model_name_or_path=args.base_model,
        so8t_config=so8t_config,
        use_redacted_tokens=args.use_redacted,
    )
    
    # トークナイザーに特殊トークンを追加
    model.set_tokenizer(tokenizer)
    
    # 8bit量子化設定
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_use_double_quant=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )
    
    # モデルを8bitに変換
    model.base_model = model.base_model.to(device)
    model.base_model = prepare_model_for_kbit_training(model.base_model)
    
    # LoRA設定
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # LoRAを適用
    model.base_model = get_peft_model(model.base_model, lora_config)
    
    # データセットを作成
    train_dataset = ThinkingDataset(samples, tokenizer, max_length=args.max_length)
    
    # 訓練引数
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        evaluation_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="none",
    )
    
    # Trainerを作成
    trainer = ThinkingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    
    # 訓練開始
    print("[INFO] Starting training...")
    trainer.train()
    
    # モデルを保存
    print(f"[INFO] Saving model to: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("[SUCCESS] Training completed!")


if __name__ == "__main__":
    main()

