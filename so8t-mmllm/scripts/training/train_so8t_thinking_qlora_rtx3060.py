"""
SO8T Thinking Model QLoRA Training Script (RTX 3060 Optimized)

RTX 3060（12GB VRAM）環境向けに最適化された訓練スクリプト。
四重推論形式（Task/Safety/Policy/Final）でのSFT + Safety/Domain/Verifierヘッド訓練。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
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

# ドメインラベルマッピング
DOMAIN_LABEL_MAP = {
    "defense_public": 0,
    "aerospace": 1,
    "medical_reg": 2,
    "law_policy": 3,
    "wikipedia_ja_en": 4,
    "nsfw_adult": 5,
    "nsfw_block": 6,
    "general": 7,
}


class QuadrupleThinkingDataset(Dataset):
    """四重推論形式データセット"""
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,  # RTX 3060向けに512に制限
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
        
        # Domainラベル
        domain_label_str = sample.get("domain_label", "general")
        domain_label = DOMAIN_LABEL_MAP.get(domain_label_str, 7)
        
        # Verifierラベル
        verifier_dict = sample.get("verifier_label", {"logical": 1.0, "faithful": 1.0})
        verifier_logical, verifier_faithful = parse_verifier_label(verifier_dict)
        
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "safety_label": torch.tensor(safety_label, dtype=torch.long),
            "domain_label": torch.tensor(domain_label, dtype=torch.long),
            "verifier_logical": torch.tensor(verifier_logical, dtype=torch.float32),
            "verifier_faithful": torch.tensor(verifier_faithful, dtype=torch.float32),
        }


def collate_fn(batch):
    """バッチをまとめる"""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    safety_labels = torch.stack([b["safety_label"] for b in batch])
    domain_labels = torch.stack([b["domain_label"] for b in batch])
    verifier_logical = torch.stack([b["verifier_logical"] for b in batch])
    verifier_faithful = torch.stack([b["verifier_faithful"] for b in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "safety_labels": safety_labels,
        "domain_labels": domain_labels,
        "verifier_logical": verifier_logical,
        "verifier_faithful": verifier_faithful,
    }


class QuadrupleThinkingTrainer(Trainer):
    """四重推論形式データ用のカスタムTrainer"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        カスタム損失計算
        - Language Model損失
        - Safety損失
        - Domain損失
        - Verifier損失
        """
        labels = inputs.pop("labels")
        safety_labels = inputs.pop("safety_labels")
        domain_labels = inputs.pop("domain_labels")
        verifier_logical = inputs.pop("verifier_logical")
        verifier_faithful = inputs.pop("verifier_faithful")
        
        # Forward pass
        outputs = model(**inputs, labels=labels, safety_labels=safety_labels, output_hidden_states=True)
        
        # 損失を取得
        total_loss = outputs["loss"]
        
        # Domain損失を計算
        hidden_states = outputs.get("hidden_states")
        if hidden_states is not None and hasattr(model, 'domain_head'):
            last_hidden = hidden_states[-1]  # [B, T, H]
            
            # 最終トークンのSpinor-成分を取得
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
                _, _, h_S_minus, _ = model.split_hidden_states(last_hidden)
                pooled_S_minus = h_S_minus[batch_indices, lengths]  # [B, d_S_minus]
            else:
                _, _, h_S_minus, _ = model.split_hidden_states(last_hidden)
                pooled_S_minus = h_S_minus[:, -1, :]  # [B, d_S_minus]
            
            # Domainヘッドで予測
            domain_logits = model.domain_head(pooled_S_minus)  # [B, NUM_DOMAIN_LABELS]
            
            # Domain損失（CrossEntropy）
            domain_loss_fct = nn.CrossEntropyLoss()
            domain_loss = domain_loss_fct(domain_logits, domain_labels)
            total_loss = total_loss + 0.3 * domain_loss  # Domain損失の重み
        
        # Verifier損失：誤検知率（誤判定率）を損失関数として追加
        # 例: 論理的に正しいはず(verifier_logical=1)/間違っているはず(verifier_logical=0)に対する
        # モデルの出力(verifier予測)が閾値0.5で正誤判定し、誤分類率=損失とする

        # まず、モデルからverifier_headのロジットまたは確率を取得（verifier_head実装側でsigmoidなど使用を想定）
        verifier_output = None
        if hasattr(model, "verifier_head") and "verifier_hidden" in outputs:
            # verifier_headへの入力特徴量
            verifier_hidden = outputs["verifier_hidden"]  # [B, D]
            verifier_logits = model.verifier_head(verifier_hidden).squeeze(-1)  # [B]
            verifier_probs = torch.sigmoid(verifier_logits)
            # 二値化: 0.5閾値
            verifier_preds = (verifier_probs >= 0.5).long()
            # 誤検知率: targetと異なるものの割合
            incorrect = (verifier_preds != verifier_logical).float()
            misdetection_rate = incorrect.mean()
            # 合計損失へ加算（重みは必要に応じて調整）
            total_loss = total_loss + 0.2 * misdetection_rate  # 例: 0.2重み
        # end of 誤検知率による損失

        if return_outputs:
            return total_loss, outputs
        return total_loss

def main():
    parser = argparse.ArgumentParser(
        description="Train SO8T Thinking Model with QLoRA (RTX 3060 Optimized)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Config file path (YAML)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Quadruple thinking format dataset (JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--use-quadruple",
        action="store_true",
        help="Use quadruple thinking format",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (RTX 3060: 512 recommended)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (RTX 3060: 1 recommended)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Gradient accumulation steps (effective batch = batch_size * gradient_accumulation_steps)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
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
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
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
        use_redacted_tokens=False,
        use_quadruple_thinking=args.use_quadruple,
    )
    
    # トークナイザーに特殊トークンを追加
    model.set_tokenizer(tokenizer)
    
    # 4bit量子化設定（RTX 3060向け）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # RTX 3060では4bit推奨
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # モデルを4bitに変換
    model.base_model = model.base_model.to(device)
    model.base_model = prepare_model_for_kbit_training(model.base_model)
    
    # LoRA設定（RTX 3060向け最適化）
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
    
    # 訓練可能パラメータを表示
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # データセットを作成
    train_dataset = QuadrupleThinkingDataset(samples, tokenizer, max_length=args.max_length)
    
    # 訓練引数（RTX 3060向け最適化）
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,  # RTX 3060ではfp16推奨
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        evaluation_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="none",
        optim="paged_adamw_8bit",  # メモリ効率的なオプティマイザー
        max_grad_norm=1.0,
        warmup_steps=100,
    )
    
    # Trainerを作成
    trainer = QuadrupleThinkingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    
    # 訓練開始
    print("[INFO] Starting training...")
    print(f"[INFO] Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"[INFO] Max sequence length: {args.max_length}")
    trainer.train()
    
    # モデルを保存
    print(f"[INFO] Saving model to: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("[SUCCESS] Training completed!")


if __name__ == "__main__":
    main()

