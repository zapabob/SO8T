#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO (Proximal Policy Optimization) 報酬学習スクリプト

SO8Tモデルに対してPPOを使用した報酬学習を実行
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
        logging.FileHandler('logs/train_ppo_reward_learning.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PPOTrainer(Trainer):
    """PPO（Proximal Policy Optimization）トレーナー"""
    
    def __init__(
        self,
        *args,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        reward_model: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.reward_model = reward_model
    
    def compute_ppo_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PPO損失を計算
        
        Args:
            logprobs: 現在のポリシーの対数確率
            old_logprobs: 古いポリシーの対数確率
            rewards: 報酬
            values: 価値関数の出力
            advantages: アドバンテージ
        
        Returns:
            (policy_loss, value_loss, entropy_loss)
        """
        # ポリシー比
        ratio = torch.exp(logprobs - old_logprobs)
        
        # クリッピングされたポリシー損失
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # 価値関数損失
        value_loss = F.mse_loss(values, rewards)
        
        # エントロピーボーナス（簡易版）
        entropy_loss = -logprobs.mean()
        
        return policy_loss, value_loss, entropy_loss
    
    def compute_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generated_texts: List[str]
    ) -> torch.Tensor:
        """
        報酬を計算
        
        Args:
            input_ids: 入力ID
            attention_mask: アテンションマスク
            generated_texts: 生成されたテキストのリスト
        
        Returns:
            報酬テンソル
        """
        if self.reward_model is not None:
            # 報酬モデルを使用
            with torch.no_grad():
                reward_outputs = self.reward_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # 報酬モデルの出力から報酬を取得（実装依存）
                rewards = reward_outputs.logits[:, 0]  # 簡易版
        else:
            # 簡易報酬関数（長さベース）
            rewards = torch.tensor([
                len(text) / 100.0 for text in generated_texts
            ], device=input_ids.device)
        
        return rewards
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        損失計算（PPO）
        
        Args:
            model: モデル
            inputs: 入力データ
            return_outputs: 出力を返すかどうか
            num_items_in_batch: バッチ内のアイテム数（未使用）
        """
        # フォワードパス
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        logits = outputs.logits
        
        # 対数確率を計算
        logprobs = F.log_softmax(logits, dim=-1)
        
        # 古い対数確率（簡易版：前回の値を保存する必要がある）
        old_logprobs = logprobs.detach()
        
        # 報酬を計算（簡易版）
        generated_texts = [""] * inputs["input_ids"].size(0)  # 実際には生成テキストを使用
        rewards = self.compute_reward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generated_texts=generated_texts
        )
        
        # 価値関数の出力（簡易版：logitsの平均を使用）
        values = logits.mean(dim=-1).mean(dim=-1)
        
        # アドバンテージを計算
        advantages = rewards - values.detach()
        
        # PPO損失を計算
        policy_loss, value_loss, entropy_loss = self.compute_ppo_loss(
            logprobs=logprobs[:, -1, :].mean(dim=-1),
            old_logprobs=old_logprobs[:, -1, :].mean(dim=-1),
            rewards=rewards,
            values=values,
            advantages=advantages
        )
        
        # 総損失
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        
        # ログ出力（定期的に）
        if self.state.global_step % 10 == 0:
            logger.info(
                f"[PPO] Step {self.state.global_step}: "
                f"Policy Loss={policy_loss.item():.4f}, "
                f"Value Loss={value_loss.item():.4f}, "
                f"Entropy Loss={entropy_loss.item():.4f}, "
                f"Total Loss={total_loss.item():.4f}"
            )
        
        return (total_loss, outputs) if return_outputs else total_loss


def main():
    parser = argparse.ArgumentParser(
        description="Train SO8T model with PPO reward learning"
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
        help="Training dataset path (JSONL format)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default=None,
        help="Reward model path (optional)"
    )
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("PPO Reward Learning")
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 報酬モデル読み込み（オプション）
    reward_model = None
    if args.reward_model:
        logger.info(f"Loading reward model from {args.reward_model}...")
        reward_model = AutoModelForCausalLM.from_pretrained(
            args.reward_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        reward_model.eval()
    
    # QLoRA設定（必要に応じて）
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
    
    # データセット読み込み（簡易版：標準データセットを使用）
    from transformers import TextDataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=str(args.dataset),
        block_size=512
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
    
    # PPOトレーナー
    reward_config = config.get("reward_learning", {})
    trainer = PPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        clip_epsilon=reward_config.get("clip_epsilon", 0.2),
        value_coef=reward_config.get("value_coef", 0.5),
        entropy_coef=reward_config.get("entropy_coef", 0.01),
        reward_model=reward_model
    )
    
    # 学習実行
    logger.info("Starting PPO training...")
    trainer.train()
    
    # 最終モデル保存
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    logger.info(f"[SUCCESS] PPO training completed. Model saved to {final_model_dir}")


if __name__ == "__main__":
    main()



