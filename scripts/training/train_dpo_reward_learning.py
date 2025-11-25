#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPO (Direct Preference Optimization) 報酬学習スクリプト

SO8Tモデルに対してDPOを使用した報酬学習を実行
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# HuggingFaceキャッシュをDドライブに設定
os.environ["HF_HOME"] = r"D:\webdataset\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\webdataset\hf_cache\transformers"
os.environ["HF_DATASETS_CACHE"] = r"D:\webdataset\hf_cache\datasets"
os.environ["HF_HUB_CACHE"] = r"D:\webdataset\hf_cache\hub"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import yaml
from tqdm import tqdm

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm" / "src"))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_dpo_reward_learning.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PairwiseDataset(Dataset):
    """ペア比較データセット"""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
        
        logger.info(f"[DATASET] Loaded {len(self.samples)} pairwise samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")
        
        # プロンプトと回答を結合
        chosen_text = f"{prompt}\n{chosen}"
        rejected_text = f"{prompt}\n{rejected}"
        
        # トークン化
        chosen_encoded = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_encoded = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        prompt_encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length // 2,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "prompt": prompt,
            "prompt_input_ids": prompt_encoded["input_ids"].squeeze(),
            "prompt_attention_mask": prompt_encoded["attention_mask"].squeeze(),
            "chosen_input_ids": chosen_encoded["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_encoded["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_encoded["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_encoded["attention_mask"].squeeze(),
        }


class DPOTrainer(Trainer):
    """DPO（Direct Preference Optimization）トレーナー"""
    
    def __init__(
        self,
        *args,
        beta: float = 0.1,
        reference_model: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.reference_model = reference_model
        
        logger.info(f"[INIT] DPOTrainer initialized with beta={beta}")
    
    def _get_logprobs(self, model, input_ids, attention_mask):
        """対数確率を取得"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # シフトして次トークンの予測に合わせる
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        
        # 対数確率を計算
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # 各トークンの対数確率を取得
        per_token_logprobs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # マスクを適用して平均を計算
        masked_logprobs = per_token_logprobs * shift_mask
        sequence_logprob = masked_logprobs.sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)
        
        return sequence_logprob
    
    def compute_dpo_loss(
        self,
        policy_chosen_logprobs: torch.Tensor,
        policy_rejected_logprobs: torch.Tensor,
        reference_chosen_logprobs: torch.Tensor,
        reference_rejected_logprobs: torch.Tensor
    ) -> torch.Tensor:
        """DPO損失を計算"""
        # ログ確率比
        chosen_logratios = policy_chosen_logprobs - reference_chosen_logprobs
        rejected_logratios = policy_rejected_logprobs - reference_rejected_logprobs
        
        # DPO損失: -log(σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x))))
        logits = self.beta * (chosen_logratios - rejected_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        return loss
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """DPO損失を計算"""
        # ポリシーモデル（現在のモデル）の対数確率
        policy_chosen_logprobs = self._get_logprobs(
            model,
            inputs["chosen_input_ids"],
            inputs["chosen_attention_mask"]
        )
        
        policy_rejected_logprobs = self._get_logprobs(
            model,
            inputs["rejected_input_ids"],
            inputs["rejected_attention_mask"]
        )
        
        # リファレンスモデルの対数確率
        if self.reference_model is not None:
            with torch.no_grad():
                reference_chosen_logprobs = self._get_logprobs(
                    self.reference_model,
                    inputs["chosen_input_ids"],
                    inputs["chosen_attention_mask"]
                )
                
                reference_rejected_logprobs = self._get_logprobs(
                    self.reference_model,
                    inputs["rejected_input_ids"],
                    inputs["rejected_attention_mask"]
                )
        else:
            # リファレンスモデルがない場合は、ポリシーモデルをリファレンスとして使用（簡易版）
            reference_chosen_logprobs = policy_chosen_logprobs.detach()
            reference_rejected_logprobs = policy_rejected_logprobs.detach()
        
        # DPO損失を計算
        dpo_loss = self.compute_dpo_loss(
            policy_chosen_logprobs=policy_chosen_logprobs,
            policy_rejected_logprobs=policy_rejected_logprobs,
            reference_chosen_logprobs=reference_chosen_logprobs,
            reference_rejected_logprobs=reference_rejected_logprobs
        )
        
        # ログ出力
        if self.state.global_step % 10 == 0:
            logger.info(
                f"[DPO] Step {self.state.global_step}: "
                f"DPO Loss={dpo_loss.item():.4f}, "
                f"Chosen Logprob={policy_chosen_logprobs.mean().item():.4f}, "
                f"Rejected Logprob={policy_rejected_logprobs.mean().item():.4f}"
            )
        
        return (dpo_loss, model(input_ids=inputs["chosen_input_ids"])) if return_outputs else dpo_loss


def main():
    parser = argparse.ArgumentParser(
        description="Train SO8T model with DPO reward learning"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config file path (YAML)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Base model path (SFT済みモデル)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Pairwise dataset path (JSONL format)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        default=None,
        help="Reference model path (optional)"
    )
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("DPO Reward Learning")
    logger.info("="*80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output: {output_dir}")
    
    # トークナイザー読み込み
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # モデル読み込み
    logger.info("Loading model...")
    load_in_8bit = config.get("quantization", {}).get("load_in_8bit", True)
    
    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=config.get("quantization", {}).get("llm_int8_threshold", 6.0)
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # リファレンスモデル読み込み（オプション）
    reference_model = None
    if args.reference_model:
        logger.info(f"Loading reference model from {args.reference_model}...")
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.reference_model,
            quantization_config=quantization_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        reference_model.eval()
    
    # QLoRA設定
    if config.get("qlora", {}).get("enabled", True):
        model = prepare_model_for_kbit_training(model)
        qlora_config = config.get("qlora", {})
        lora_config = LoraConfig(
            r=qlora_config.get("r", 64),
            lora_alpha=qlora_config.get("lora_alpha", 128),
            target_modules=qlora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=qlora_config.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        logger.info("[OK] QLoRA applied")
    
    # データセット読み込み
    train_dataset = PairwiseDataset(
        data_path=args.dataset,
        tokenizer=tokenizer,
        max_length=config.get("data", {}).get("max_seq_length", 2048)
    )
    
    # トレーニング引数
    training_config = config.get("training", {})
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 16),
        learning_rate=training_config.get("learning_rate", 1.0e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=training_config.get("warmup_steps", 100),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 500),
        fp16=training_config.get("fp16", True),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        optim=training_config.get("optim", "paged_adamw_8bit"),
        report_to=[],
    )
    
    # DPOトレーナー
    reward_config = config.get("reward_learning", {})
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        beta=reward_config.get("beta", 0.1),
        reference_model=reference_model
    )
    
    # 学習実行
    logger.info("Starting DPO training...")
    trainer.train()
    
    # 最終モデル保存
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    logger.info(f"[SUCCESS] DPO training completed. Model saved to {final_model_dir}")


if __name__ == "__main__":
    main()
