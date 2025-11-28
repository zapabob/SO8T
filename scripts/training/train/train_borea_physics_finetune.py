#!/usr/bin/env python3
"""
Borea-Phi3.5-instinct-jp Physics Fine-tuning

Realistic approach: Fine-tune Borea model with physics/mathematics knowledge
while maintaining its strong language generation capabilities.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def create_physics_lora_config():
    """Create LoRA config optimized for physics fine-tuning"""
    return LoraConfig(
        r=16,  # Smaller rank for stability
        lora_alpha=32,
        target_modules=[
            "self_attn.qkv_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def prepare_physics_dataset(tokenizer, max_samples=5000):
    """Prepare physics and mathematics focused dataset"""

    print("[DATA] Preparing physics-focused dataset...")

    # Load multiple relevant datasets
    datasets = []

    # Japanese physics/mathematics datasets
    try:
        physics_data = load_dataset("TFMC/imatrix-dataset-for-japanese-llm", split="train", streaming=True)
        datasets.append(("physics", physics_data))
    except:
        print("[WARNING] Physics dataset not available")

    # Create synthetic physics Q&A data
    physics_questions = [
        "時間はなぜ不可逆なのですか？",
        "エネルギー保存則とは何ですか？",
        "量子力学の不確定性原理を説明してください。",
        "相対性理論の特殊相対性とは何ですか？",
        "熱力学第一法則と第二法則を説明してください。",
        "SO(8)群とは何ですか？",
        "黄金比 φ はどのような場面で現れますか？",
        "数学における対称性とは何ですか？",
        "幾何学と物理学の関係について説明してください。",
        "意識とは物理的な現象ですか？"
    ]

    synthetic_data = []
    for question in physics_questions:
        # Create Q&A format
        qa_pair = f"<|user|>\n{question}<|end|>\n<|assistant|>\nこれは深い哲学的・物理的な質問です。{question}に対して、科学的・数学的な視点から考えてみましょう。"
        synthetic_data.append({"text": qa_pair})

    print(f"[DATA] Prepared {len(synthetic_data)} physics Q&A samples")

    def tokenize_function(examples):
        text = examples.get("text", "")
        if not isinstance(text, str):
            text = str(text)

        tokenized = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["input_ids"].squeeze()
        }

    # Process synthetic data
    synthetic_dataset = []
    for item in synthetic_data:
        tokenized = tokenize_function(item)
        if tokenized["input_ids"].numel() > 0:  # Valid sample
            synthetic_dataset.append(tokenized)

    print(f"[DATA] Final dataset size: {len(synthetic_dataset)} samples")

    return synthetic_dataset

def train_borea_physics_finetune(
    model_path="models/Borea-Phi-3.5-mini-Instruct-Jp",
    output_dir="models/Borea-Phi3.5-physics-finetuned",
    max_steps=500,
    learning_rate=2e-5,
    batch_size=2,
    gradient_accumulation_steps=4
):
    """Fine-tune Borea with physics knowledge"""

    print("[TRAINING] Starting Borea physics fine-tuning...")

    # Set protobuf environment
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = create_physics_lora_config()
    model = get_peft_model(model, lora_config)

    print(f"[PARAMS] Trainable parameters: {model.num_parameters(only_trainable=True):,}")

    # Prepare dataset
    dataset = prepare_physics_dataset(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=50,
        logging_steps=25,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("[TRAINING] Starting physics fine-tuning...")
    trainer.train()

    # Save final model
    final_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"[COMPLETE] Physics fine-tuned model saved to {final_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Borea-Phi3.5 with physics knowledge")
    parser.add_argument("--model-path", type=str, default="models/Borea-Phi-3.5-mini-Instruct-Jp")
    parser.add_argument("--output-dir", type=str, default="models/Borea-Phi3.5-physics-finetuned")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)

    args = parser.parse_args()

    train_borea_physics_finetune(
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()


