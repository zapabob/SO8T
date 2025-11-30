#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borea-Phi-3.5-mini-Instruct-Jp SO8T PPO Training Pipeline
SO8T/thinkingモデル化のためのPPOトレーニング
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from tqdm import tqdm
from datetime import datetime
import argparse

# パス設定 (直接実行時用)
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root / "utils"))

# 外部ライブラリ
try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers import TrainingArguments
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from trl.core import LengthSampler
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import bitsandbytes as bnb

    # UnslothはGPU必須なので、CPUモードでは使用しない
    USE_UNSLOTH = torch.cuda.is_available()
    if USE_UNSLOTH:
        try:
            from unsloth import FastLanguageModel
            from unsloth.chat_templates import get_chat_template
        except ImportError:
            USE_UNSLOTH = False
            print("Unsloth not available, using standard transformers")
    else:
        print("CUDA not available, using standard transformers (CPU mode)")

except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install transformers trl datasets peft bitsandbytes")
    sys.exit(1)

# SO8Tモジュールインポート
try:
    from scripts.training.nkat_reward_function import NKATRewardFunction
    from scripts.inference.nkat_thermostat import NKATDynamicTemperature, create_nkat_thermostat
    from utils.checkpoint_manager import RollingCheckpointManager
except ImportError as e:
    print(f"SO8T module import error: {e}")
    sys.exit(1)

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SO8TThinkingDataset(Dataset):
    """SO8T思考トレーニング用データセット"""

    def __init__(self, dataset_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # データセット読み込み
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        logger.info(f"Loaded {len(self.data)} examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # システムプロンプト + インストラクション
        system_prompt = item.get('system', '')
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')

        # チャットフォーマット
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{instruction}\n{input_text}".strip()}
        ]

        # トークナイズ
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return {
            "query": prompt,
            "ground_truth": item.get('output', ''),
            "domain": item.get('domain', 'unknown'),
            "inference_type": item.get('inference_type', 'deduction')
        }

def create_so8t_ppo_config(args) -> PPOConfig:
    """SO8T PPO設定を作成"""

    config = PPOConfig(
        model_name="Borea-Phi-3.5-mini-Instruct-Jp",
        learning_rate=args.learning_rate,
        log_with="wandb",  # Weights & Biases
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=args.early_stopping,
        target_kl=args.target_kl,
        kl_penalty=args.kl_penalty,
        seed=args.seed,
        use_score_scaling=args.use_score_scaling,
        use_score_norm=args.use_score_norm,
        score_clip=args.score_clip,
    )

    return config

def setup_so8t_model_and_tokenizer(model_path: str = None):
    """SO8Tモデルとトークナイザーのセットアップ"""

    # GPU利用可能性チェック
    use_4bit = torch.cuda.is_available()
    device = "cuda" if use_4bit else "cpu"

    if USE_UNSLOTH and use_4bit:
        # Unsloth使用時 (GPU必須)
        if model_path and Path(model_path).exists():
            logger.info(f"Loading model from {model_path} (Unsloth)")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
        else:
            logger.info("Loading base Borea-Phi-3.5-mini-Instruct-Jp model (Unsloth)")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="Borea-Phi-3.5-mini-Instruct-Common",
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )

        # LoRA設定
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # LoRA rank
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        # チャットテンプレート設定
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="phi-3",
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        )
    else:
        # 標準transformers使用時 (CPU/GPU両対応)
        # ローカルパスが存在するか、HFモデル名が指定されている場合
        if model_path and Path(model_path).exists():
            model_name = model_path
        elif model_path and "/" in model_path:  # HFモデル名 (例: user/model-name)
            model_name = model_path
        else:
            model_name = "microsoft/Phi-3.5-mini-instruct"  # デフォルトHFモデル

        logger.info(f"Loading model {model_name} (standard transformers, {'GPU' if use_4bit else 'CPU'} mode)")

        # 量子化設定 (GPU時のみ)
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None

        # モデルとトークナイザーのロード
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if use_4bit else None,
            torch_dtype=torch.float16 if use_4bit else torch.float32,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # CPUモード時はモデルをCPUに移動
        if not use_4bit:
            model = model.to(device)

        # LoRA設定 (CPU/GPU両対応)
        if use_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # チャットテンプレート (簡易版)
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant:"

    return model, tokenizer

def create_so8t_reward_function(tokenizer):
    """SO8T報酬関数を作成"""

    reward_fn = NKATRewardFunction(tokenizer=tokenizer)

    # NKAT Thermostat統合
    thermostat_controller = create_nkat_thermostat(tokenizer)
    reward_fn.thermostat = thermostat_controller.get_logits_processor()

    return reward_fn

def train_so8t_ppo(args):
    """SO8T PPOトレーニングメイン関数"""

    logger.info("=== SO8T PPO Training Started ===")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Model: {args.model_path or 'Borea-Phi-3.5-mini-Instruct-Jp'}")
    logger.info(f"Execution ID: {getattr(args, 'execution_id', 'manual')}")

    # モデルとトークナイザー準備
    model, tokenizer = setup_so8t_model_and_tokenizer(args.model_path)

    # データセット準備
    dataset = SO8TThinkingDataset(args.dataset_path, tokenizer)

    # PPO設定
    ppo_config = create_so8t_ppo_config(args)

    # 報酬関数
    reward_fn = create_so8t_reward_function(tokenizer)

    # PPOトレーナー
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,  # 参照モデルなし（メモリ節約）
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=lambda data: {
            'input_ids': [torch.tensor(tokenizer.encode(item['query'])) for item in data],
            'attention_mask': [torch.ones(len(tokenizer.encode(item['query']))) for item in data]
        }
    )

    # チェックポイントマネージャー
    checkpoint_base_dir = getattr(args, 'checkpoint_dir', args.output_dir)
    checkpoint_manager = RollingCheckpointManager(
        Path(checkpoint_base_dir),
        max_keep=5,
        save_interval_sec=180  # 3分ごと
    )

    # トレーニングループ
    logger.info("Starting PPO training...")

    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        for step, batch in enumerate(tqdm(ppo_trainer.dataloader, desc=f"Epoch {epoch+1}")):
            # クエリ
            queries = batch['input_ids']

            # 応答生成
            response_kwargs = {
                'max_new_tokens': 512,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'pad_token_id': tokenizer.eos_token_id,
            }

            # NKAT Thermostat適用
            if hasattr(reward_fn, 'thermostat'):
                response_kwargs['logits_processor'] = [reward_fn.thermostat]

            responses = ppo_trainer.generate(queries, **response_kwargs)

            # 報酬計算
            rewards = []
            for query, response in zip(queries, responses):
                # テキストにデコード
                query_text = tokenizer.decode(query.squeeze(), skip_special_tokens=True)
                response_text = tokenizer.decode(response.squeeze()[len(query.squeeze()):], skip_special_tokens=True)

                # 報酬計算
                reward = reward_fn.calculate_reward(query_text, response_text)
                rewards.append(torch.tensor(reward))

            rewards = torch.stack(rewards).to(ppo_trainer.accelerator.device)

            # PPOステップ
            train_stats = ppo_trainer.step(queries, responses, rewards)

            # ログ出力
            if step % args.log_interval == 0:
                logger.info(f"Step {step}: Reward = {rewards.mean().item():.4f}")
                logger.info(f"PPO Stats: {train_stats}")

        # エポック終了時のチェックポイント保存
        checkpoint_manager.save_checkpoint(
            model,
            tokenizer,
            step_info=f"epoch_{epoch+1}"
        )

    # 最終モデル保存
    final_path = Path(args.output_dir) / "final_model"
    logger.info(f"Saving final model to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    logger.info("=== SO8T PPO Training Completed ===")

def main():
    parser = argparse.ArgumentParser(description="SO8T PPO Training Pipeline")

    # データセット設定
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to SO8T thinking dataset (JSONL)")
    parser.add_argument("--execution_id", type=str, default=None,
                       help="Execution ID for logging and checkpointing")
    parser.add_argument("--model_path", type=str, default="microsoft/Phi-3.5-mini-instruct",
                       help="Path to pre-trained model or HuggingFace model name")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                       help="Checkpoint directory (default: output_dir/checkpoints)")

    # トレーニング設定
    parser.add_argument("--output_dir", type=str, default="outputs/so8t_ppo",
                       help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=1,
                       help="Mini batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1.41e-5,
                       help="Learning rate")

    # PPO設定
    parser.add_argument("--ppo_epochs", type=int, default=4,
                       help="PPO epochs")
    parser.add_argument("--target_kl", type=float, default=0.1,
                       help="Target KL divergence")
    parser.add_argument("--kl_penalty", type=str, default="kl",
                       help="KL penalty type")

    # その他設定
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Logging interval")
    parser.add_argument("--early_stopping", type=bool, default=False,
                       help="Enable early stopping")
    parser.add_argument("--use_score_scaling", type=bool, default=False,
                       help="Use score scaling")
    parser.add_argument("--use_score_norm", type=bool, default=True,
                       help="Use score normalization")
    parser.add_argument("--score_clip", type=float, default=None,
                       help="Score clipping threshold")

    args = parser.parse_args()

    # 出力ディレクトリ作成
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # トレーニング実行
    train_so8t_ppo(args)

if __name__ == "__main__":
    main()
